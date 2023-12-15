from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Conv1D, Conv2D, Conv2DTranspose, BatchNormalization, Activation, Reshape, Layer, Input, Add, Concatenate, UpSampling2D, Dense, Dropout, MaxPooling2D, Flatten, Multiply, Lambda
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import Sequence
import os
import numpy as np
from PIL import Image, ImageFilter
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
import tensorflow as tf
import tensorflow_addons as tfa
import cv2 as cv
import random
from tensorflow.keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt 
import tensorflow.keras.backend as backend
from tensorflow.keras.initializers import RandomNormal
from skimage.util.noise import random_noise
from tensorflow.python.keras.backend import binary_crossentropy


seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

DEBUG = False

image_size = 64
output_image_size = image_size

if DEBUG:
    batch_size = 24
    latent_space_size = 32
    num_filters = 8
    epochs = 10
else :
    batch_size = 32
    latent_space_size = 128
    num_filters = 8
    epochs = 5


NUM_BODY_PARTS = 1
NUM_COLORS = 3
NUM_OUTPUT_CHANNELS = 1 + NUM_COLORS

is_training = True

train_folder = r"C:\Users\sraimund\Pictorial-Maps-Autoencoder\train\heads_input"
test_folder = r"C:\Users\sraimund\Pictorial-Maps-Autoencoder\test\heads_input"
weights_folder = r"C:\Users\sraimund\Pictorial-Maps-Autoencoder"

weights_file_path = os.path.join(weights_folder, "autoencoders.hdf5")

class DataGenerator(Sequence):

    def __init__(self, image_folder, preprocess_input_image, batch_size):
        image_names = os.listdir(image_folder)  
        
        image_paths = list(map(lambda image_name: os.path.join(image_folder, image_name), image_names))
        
        if DEBUG:
            image_paths = image_paths[:batch_size]
        
        self.file_names = image_paths
        self.preprocess_input_image = preprocess_input_image
        self.batch_size = batch_size
        
        self.on_epoch_end()

    
    def __len__(self):
        return int(np.floor(len(self.file_names) / self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        file_names_temp = [self.file_names[k] for k in indexes]
        X, Y = self.__data_generation(file_names_temp, indexes)

        return X, Y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_names))
        np.random.shuffle(self.indexes)


    def __data_generation(self, image_file_paths, indexes):
        source = np.empty((self.batch_size, image_size, image_size, NUM_COLORS * NUM_BODY_PARTS + 2), dtype=np.float32) # NUM_COLORS * NUM_BODY_PARTS + 2
        target = np.zeros((self.batch_size, output_image_size, output_image_size, NUM_OUTPUT_CHANNELS), dtype=np.float32)


        for i, image_input_file_path in enumerate(image_file_paths):
            image_output_file_path = image_input_file_path.replace("heads_input", "heads_output")
            mask_file_path = image_input_file_path.replace("heads_input", "heads_masks").replace(".jpg", ".png")
            uv_file_path = image_input_file_path.replace("heads_input", "heads_uv").replace(".jpg", ".npy")
                        
            if (self.preprocess_input_image == preprocess_train_image):
                rotation_angle = 0 # random.randint(-45, 45)
                color_shift = [random.randint(0, 255), random.randint(0, 180), random.randint(0, 255)]
                oilyness = random.randint(0, 1)
                blur_radius = random.randint(0, 1)
                
                # image_input_file_path = image_output_file_path
                
            else : # test images
                rotation_angle = 0
                color_shift = [0, 0, 0]
                blur_radius = 0
                oilyness = 0
            
            input_image_np = self.preprocess_input_image(image_input_file_path, rotation_angle, color_shift, blur_radius, oilyness)            
            output_image_np = preprocess_output_image(image_output_file_path, rotation_angle, color_shift)
            binary_mask_np, body_parts_mask_np = preprocess_body_part_mask(mask_file_path, rotation_angle)
            uv_map_np = np.load(uv_file_path)
            
            # Image.fromarray(((input_image_np + 1) * 127.5).astype(np.uint8)).show()
            # Image.fromarray((output_image_np * 255).astype(np.uint8)).show()
            # Image.fromarray((binary_mask_np[..., 0] * 255).astype(np.uint8)).show()
            
            source_image_np = np.concatenate([input_image_np * np.concatenate([binary_mask_np, binary_mask_np, binary_mask_np], axis=-1), uv_map_np], axis=-1)
            target_image_np = np.concatenate([binary_mask_np, output_image_np], axis=2)
            
            source[i,] = source_image_np
            target[i,] = target_image_np          
        
        return source, target


def preprocess_train_image(file_path, rotation_angle=0, color_shift=[0, 0, 0], blur_radius=0, oilyness=0):
    image = Image.open(file_path)
    
    image = image.rotate(rotation_angle, Image.CUBIC)
    image = oilpaint_image(image, oilyness)
    image = image.resize((image_size, image_size), Image.NEAREST)
     
    image = blur_image(image, blur_radius)

    image = apply_noises(image)
    image = apply_colored_salt_and_pepper_noise(image, 0.3)
    
    # image.show()
    
    image_np = np.asarray(image, dtype=np.uint8)
    
    image_rgb_shifted_np = shift_color(image_np, color_shift)
    return image_rgb_shifted_np # / 127.5 - 1


def preprocess_test_image(file_path, rotation_angle=0, color_shift=[0, 0, 0], blur_radius=0, oilyness=0):
    image = Image.open(file_path)
    # image = blur_image(image, 1)
    # image = image.resize((image_size, image_size), Image.NEAREST)
    resized_image_np = np.asarray(image, dtype=np.uint8)
    # preprocessed_image = imagenet_utils.preprocess_input(resized_image_np)
    image_rgb_shifted_np = shift_color(resized_image_np, color_shift)

    return image_rgb_shifted_np # / 127.5 - 1


def preprocess_body_part_mask(mask_file_path, rotation_angle=0):
    mask = Image.open(mask_file_path)
    mask = mask.rotate(rotation_angle, Image.NEAREST, fillcolor=255)
    
    mask = mask.getchannel(0)
    # mask = mask.resize((output_image_size, output_image_size), Image.NEAREST)
    mask_np = np.asarray(mask)
    
    binary_mask_np = (mask_np == 255).astype(np.uint8).astype(np.float32)
    binary_mask_np = np.expand_dims(binary_mask_np, -1)
    
    body_parts_mask_np = np.zeros((image_size, image_size, NUM_BODY_PARTS))
    body_parts_mask_np[..., 0] = (mask_np == 255).astype(np.uint8)
    
    return binary_mask_np, body_parts_mask_np


def preprocess_output_image(image_file_path, rotation_angle, color_shift=[0, 0, 0]):
    image = Image.open(image_file_path)
    image = image.rotate(rotation_angle, Image.CUBIC)
    # image.show()
    
    # image = image.resize((output_image_size, output_image_size), Image.NEAREST)
    resized_image_np = np.asarray(image)

    image_rgb_shifted_np = shift_color(resized_image_np, color_shift)
    return image_rgb_shifted_np


def blur_image(image, radius):
    if (radius == 0):
        return image
    
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius = radius))
    
    return blurred_image


def oilpaint_image(image, oilyness):
    if (oilyness == 0):
        return image
    
    image_np = np.asarray(image, dtype=np.uint8)
    
    image_np = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)
    
    dynRatio = 10 # random.randint(1, 10)
    
    image_np = cv.xphoto.oilPainting(image_np, oilyness, dynRatio)
    image_np = cv.cvtColor(image_np, cv.COLOR_BGR2RGB)
    
    return Image.fromarray(image_np.astype(np.uint8))


# source: https://stackoverflow.com/questions/67448555/python-opencv-how-to-change-hue-in-hsv-channels/67452492#67452492
def shift_color(image_np, color_shift):
    # return image_np
    
    hsv = cv.cvtColor(image_np, cv.COLOR_RGB2HSV)
    hue = hsv[:,:,0]
    saturation = hsv[:,:,1]
    value = hsv[:,:,2]
    
    shifted_saturation = saturation
    shifted_hue = (hue + color_shift[1]) % 180
    shifted_value = value
    
    image_color_shifted_np = cv.merge([np.cos(shifted_hue/90*np.pi) * (shifted_saturation/255), np.sin(shifted_hue/90*np.pi) * (shifted_saturation/255), 2*(shifted_value/255)-1])
    
    return image_color_shifted_np

# 3-channel RGB array (0...255)
def apply_colored_salt_and_pepper_noise(image, percentage):
    arr = np.array(image)
    
    (w, h, d) = arr.shape
    
    random_pixels = np.random.choice([0, 1], size=(w, h), p=[1 - percentage, percentage])
    
    random_pixels_rgb = np.tile(np.expand_dims(random_pixels, -1), (1, 1, 3))
    
    random_colors = np.floor(np.random.rand(w, h, d) * 255)
    # implicit modulo operation to 255
    noisy_image = Image.fromarray((arr + random_pixels_rgb * random_colors).astype(np.uint8))
    
    return noisy_image


def apply_noises(image):
    arr = np.array(image)
    
    modes = ["gaussian"] # , "poisson", "speckle"
    # random.shuffle(modes)
    
    for mode in modes:
        arr = random_noise(arr, mode=mode)
    
    noisy_image = Image.fromarray(np.uint8(arr * 255))
    
    return noisy_image
    


class Conv1DBatch(Layer):
    
    def __init__(self, num_filters, kernel_size, strides=1, activation = LeakyReLU(), activity_regularizer=None, *args, **kwargs):
        super(Conv1DBatch, self).__init__(*args, **kwargs)
    
        self.conv = Conv1D(num_filters, kernel_size, strides=strides, padding = 'same', kernel_initializer = 'he_normal', activity_regularizer=activity_regularizer)
        self.bn = BatchNormalization(trainable=False)
        self.activation = Activation(activation)
    
    def call(self, x):
        x = self.conv(x)
        if (batch_size > 1):
            x = self.bn(x)   
        x = self.activation(x)
     
        return x
    
class Conv2DBatch(Layer):
    
    def __init__(self, num_filters, kernel_size, strides=(1, 1), activation = LeakyReLU(), dilation_rate=(1, 1), *args, **kwargs):
        super(Conv2DBatch, self).__init__(*args, **kwargs)
    
        self.conv = Conv2D(num_filters, kernel_size, strides=strides, dilation_rate=dilation_rate, padding = 'same', kernel_initializer = 'he_normal')
        self.bn = BatchNormalization(trainable=False)
        self.activation = Activation(activation)
    
    def call(self, x):
        x = self.conv(x)
        
        if (batch_size > 1):
            x = self.bn(x) 
        x = self.activation(x)
               
        return x


class Conv2DTransposeBatch(Layer):
    
    def __init__(self, num_filters, kernel_size, strides=(1, 1), activation = LeakyReLU(), *args, **kwargs):
        super(Conv2DTransposeBatch, self).__init__(*args, **kwargs)
    
        self.conv = Conv2DTranspose(num_filters, kernel_size, strides=strides, padding = 'same', kernel_initializer = 'he_normal')
        self.bn = BatchNormalization(trainable=False)
        self.activation = Activation(activation)
    
    def call(self, x):
        x = self.conv(x)
        
        if (batch_size > 1):
            x = self.bn(x) 
        x = self.activation(x)
        
        return x


def down(num_filters, layer):
    x = Conv2DBatch(num_filters * 2, 3, strides = (2,2))(layer)
    x = Conv2DBatch(num_filters * 2, 3)(x)
    x = Conv2DBatch(num_filters * 2, 3)(x)
    
    return x

def up(num_filters, layer):
    deconv = Conv2DTransposeBatch(num_filters, 4, strides = (2,2))(layer)
    deconv = Conv2DBatch(num_filters, 3)(deconv)
    deconv = Conv2DBatch(num_filters, 3)(deconv)
    
    return deconv

def up_sample(num_filters, layer, other_layer):
    deconv = UpSampling2D(interpolation="bilinear")(layer)
    deconv = Conv2DBatch(num_filters, 3)(deconv)
    deconv = Conv2DBatch(num_filters, 3)(deconv)
    
    return deconv


def up_add(num_filters, layer, other_layer):
    deconv = Conv2DTransposeBatch(num_filters, 4, strides = (2,2))(layer)
    
    other_layer = up_down(other_layer, num_filters)
    
    deconv = Add()([deconv, other_layer])
    deconv = Conv2DBatch(num_filters, 3)(deconv)   
    
    return deconv


def custom_loss(ytrue, ypred):
    return ytrue[..., 0] * mean_squared_error(ytrue[..., 1:3], ypred[..., 1:3]) + ytrue[..., 0] * mean_squared_error(ytrue[..., 3:4], ypred[..., 3:4])


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
  """A residual block.

  Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.

  Returns:
    Output tensor for the residual block.
  """
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  if conv_shortcut:
    shortcut = layers.Conv2D(
        4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
  else:
    shortcut = x

  x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
  x = layers.Activation("relu", name=name + '_1_relu')(x)

  x = layers.Conv2D(
      filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
  x = layers.Activation("relu", name=name + '_2_relu')(x)

  x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

  x = layers.Add(name=name + '_add')([shortcut, x])
  x = layers.Activation("relu", name=name + '_out')(x)
  return x


def stack1(x, filters, blocks, stride1=2, name=None):
  """A set of stacked residual blocks.

  Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    blocks: integer, blocks in the stacked blocks.
    stride1: default 2, stride of the first layer in the first block.
    name: string, stack label.

  Returns:
    Output tensor for the stacked blocks.
  """
  x = block1(x, filters, stride=stride1, name=name + '_block1')
  for i in range(2, blocks + 1):
    x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
  return x



def autoencoder(x_in, part):
    conv1 = stack1(x_in, num_filters * 1, 1, stride1=1, name='part%s_conv1' % part)
    conv2 = stack1(conv1, num_filters * 1, 1, stride1=2, name='part%s_conv2' % part)
    conv3 = stack1(conv2, num_filters * 2, 1, stride1=2, name='part%s_conv3' % part)
    conv4 = stack1(conv3, num_filters * 4, 1, stride1=2, name='part%s_conv4' % part)
    conv5 = stack1(conv4, num_filters * 8, 1, stride1=2, name='part%s_conv5' % part)
    
    x = Conv2DBatch(num_filters * 8, 3)(conv5)
    x = Reshape((1, 4 * 4 * num_filters * 8))(x)
    x = Conv1DBatch(latent_space_size, 1, activity_regularizer=regularizers.l1(10e-5), name="latent_space_%s" % part)(x)
    x = Conv1DBatch(4 * 4 * num_filters * 4, 1)(x)
    x = Reshape((4, 4, num_filters * 4))(x)
    x = Conv2DBatch(num_filters * 16, 3)(x)

    up4 = up(num_filters * 8, x)
    up3 = up(num_filters * 4, up4)
    up2 = up(num_filters * 2, up3)
    up1 = up(num_filters * 1, up2)
    
    x_out = Conv2D(1, kernel_size=(1, 1), padding="same", 
                activation='tanh', name="final_conv1_%s" % part, kernel_initializer = 'he_normal')(up1)
 
    return x_out


class Latent_Code(tf.keras.layers.Layer):
    
    def __init__(self, *args, **kwargs):
        super(Latent_Code, self).__init__(*args, **kwargs)
        
        self.latent_code = self.add_weight("latent_code", shape=[1804, latent_space_size], 
                                           initializer=RandomNormal(mean=0.0, stddev=0.01), regularizer=regularizers.L2(0.000001))  
        
    def call(self, shape_index):
        latent_code = tf.gather_nd(self.latent_code, shape_index)
        latent_code = tf.expand_dims(latent_code, 1)
        
        return latent_code

    

def medical(x, part):
    conv1 = Conv2DBatch(num_filters * 8, 3)(x)
    pool1 = MaxPooling2D()(conv1)
    conv2 = Conv2DBatch(num_filters * 4, 3)(pool1)
    pool2 = MaxPooling2D()(conv2)
    conv3 = Conv2DBatch(num_filters * 2, 3)(pool2)
    up2 = UpSampling2D()(conv3)
    conv2 = Conv2DBatch(num_filters * 4, 3)(up2)
    up1 = UpSampling2D()(conv2)
    conv1 = Conv2DBatch(num_filters * 8, 3)(up1)
    
    x_out = Conv2D(3, kernel_size=(1, 1), padding="same", 
                activation='tanh', name="final_conv2%s" % part, kernel_initializer = 'he_normal')(conv1)
    
    return x_out
    
    
image_shape = (image_size, image_size, NUM_COLORS * NUM_BODY_PARTS + 2)
x_in = Input(image_shape)

x_out1 = autoencoder(x_in[..., (NUM_COLORS - 1) * NUM_BODY_PARTS:], "a")
x_out3 = medical(x_in[..., :(NUM_COLORS - 1) * NUM_BODY_PARTS], "b")

x_out = Concatenate(-1)([x_out3, x_out1])

model = Model(inputs=x_in, outputs=x_out)
model.summary()
model.compile(optimizer = Adam(0.0001), loss = custom_loss)

from tensorflow.keras.utils import plot_model
plot_file_path = os.path.join(weights_folder, "model.png")
plot_model(model, to_file=plot_file_path, show_shapes=True)

# intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("latent_space").output)
# intermediate_layer_model1 = Model(inputs=model.input, outputs=model.get_layer("final_conv1_a").output)
# intermediate_layer_model2 = Model(inputs=model.input, outputs=model.get_layer("final_conv2a").output)

def predict():
    """
    if DEBUG:
        image_folders = [train_folder]
        num_images = batch_size
    else :
    """
    image_folders = [train_folder, test_folder]
    num_images = 60
    
    
    for image_folder in image_folders:
        for image_index, image_name in enumerate(os.listdir(image_folder)[:num_images]):
            image_output_folder = image_folder.replace("heads_input", "results")
            image_file_path = os.path.join(image_folder, image_name)
            
            if (image_folder == train_folder):
                input_image_np = preprocess_train_image(image_file_path, rotation_angle=0, color_shift=[0, 0, 0], blur_radius=0, oilyness=0)
                # Image.fromarray(((input_image_np + 1) * 127.5).astype(np.uint8)).show()
            else:
                input_image_np = preprocess_test_image(image_file_path)
                
            body_parts_mask_path = image_file_path.replace("heads_input", "heads_masks").replace(".jpg", ".png")
            binary_mask_np, body_parts_mask_np = preprocess_body_part_mask(body_parts_mask_path)
            
            uv_map_path = image_file_path.replace("heads_input", "heads_uv").replace(".jpg", ".npy")
            uv_map_np = np.load(uv_map_path)
 
            input_image_np = np.concatenate([input_image_np * np.concatenate([binary_mask_np, binary_mask_np, binary_mask_np], axis=-1), uv_map_np], axis=-1)
            source_image_np = np.expand_dims(input_image_np, axis=0)
            
            result = model.predict(source_image_np)[0][..., 1:]
            
            hue1 = (np.arctan2(result[..., 1], result[..., 0]) / np.pi * 90) % 180
            saturation1 = np.sqrt((result[..., 0])**2 + (result[..., 1])**2) * 255
            value1 = (result[..., 2] * 0.5 + 0.5) * 255

            result = np.stack([hue1, saturation1, value1], axis=-1)
            result = cv.cvtColor(result.astype(np.uint8), cv.COLOR_HSV2RGB)
            binary_mask_rgb_np = np.concatenate([binary_mask_np, binary_mask_np, binary_mask_np], axis=-1)
            result = result * binary_mask_rgb_np + (1 - binary_mask_rgb_np) * 255 
            result_image = Image.fromarray(np.uint8(result), "RGB")
            
            result_image.save(os.path.join(image_output_folder, image_name.replace(".jpg", ".png"))) 

            result_image = Image.fromarray(np.uint8(hue1 / 180 * 255 * binary_mask_np[..., 0]), "L")
            result_image.save(os.path.join(image_output_folder, image_name.replace(".jpg", "_hue.png")))
            
            result_image = Image.fromarray(np.uint8(saturation1 * binary_mask_np[..., 0]), "L")
            result_image.save(os.path.join(image_output_folder, image_name.replace(".jpg", "_saturation.png")))
            
            result_image = Image.fromarray(np.uint8(value1 * binary_mask_np[..., 0]), "L")
            result_image.save(os.path.join(image_output_folder, image_name.replace(".jpg", "_value.png")))

            

if (is_training):
    # model.load_weights(weights_file_path)
    
    training_data_gen = DataGenerator(train_folder, preprocess_train_image, batch_size)
    test_data_gen = DataGenerator(test_folder, preprocess_test_image, 12)
    
    for i in range(1000):
        print ("Loop", i)
        
        model.fit(
            training_data_gen,
            epochs = epochs,
            validation_data=test_data_gen,
            verbose=2
        )
        
        predict()
        model.save_weights(weights_file_path)
        
else :
    model.load_weights(weights_file_path)
    predict()
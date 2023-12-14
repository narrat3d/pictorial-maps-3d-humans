from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Conv2D, Conv2DTranspose, BatchNormalization, Activation, Layer, Input, Add
from tensorflow.keras.utils import Sequence
import os
import numpy as np
from PIL import Image
from tensorflow.keras.optimizers import Adam
from keras.losses import mean_squared_error
from mathutils import Color
import cv2 as cv
import random
import math


image_size = 128
output_image_size = image_size
batch_size = 16
epochs = 1
num_filters = 8
NUM_BODY_PARTS = 6
NUM_COLORS = 3

is_training = True

train_folder = r"E:\CNN\masks\data\figures_pictorial\train_real\images"
train_folder2 = r"E:\CNN\masks\data\figures_pictorial\train_pictorial\images"
test_folder = r"E:\CNN\masks\data\figures_pictorial\test_real\images"
test_folder2 = r"E:\CNN\masks\data\figures_pictorial\test_pictorial\images"
weights_folder = r"E:\CNN\masks\data\figures_pictorial"

weights_file_path = os.path.join(weights_folder, "weights_uv.hdf5")


class DataGenerator(Sequence):

    def __init__(self, image_folder, image_real_folder, preprocess_input_image, batch_size):
        image_names = os.listdir(image_folder)  
        image_paths = list(map(lambda image_name: os.path.join(image_folder, image_name), image_names))
        
        self.file_names = image_paths
        
        image_real_names = os.listdir(image_real_folder)  
        image_real_paths = list(map(lambda image_name: os.path.join(image_real_folder, image_name), image_real_names))
        
        self.file_real_names = image_real_paths        
        
        self.preprocess_input_image = preprocess_input_image
        self.batch_size = batch_size
        
        self.on_epoch_end()

    
    def __len__(self):
        return int(np.floor(len(self.file_names) / self.batch_size))


    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        file_names_temp = [self.file_names[k] for k in indexes]
        file_names_real = [self.file_real_names[k] for k in indexes]
        X, Y = self.__data_generation(file_names_temp, file_names_real)

        return X, Y


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_names))
        np.random.shuffle(self.indexes)
        
        self.indexes_real = np.arange(len(self.file_real_names))
        np.random.shuffle(self.indexes_real)
        
        self.indexes_real = self.indexes_real[:len(self.indexes)]


    def __data_generation(self, image_file_paths, image_real_file_paths):
        source = np.empty((self.batch_size * 2, image_size, image_size, NUM_COLORS * NUM_BODY_PARTS + 1), dtype=np.float32)
        target = np.zeros((self.batch_size * 2, output_image_size, output_image_size, 4 + 2), dtype=np.float32)

        merged_file_paths = image_file_paths + image_real_file_paths 
        np.random.shuffle(merged_file_paths)

        for i, image_file_path in enumerate(merged_file_paths):
            if (self.preprocess_input_image == preprocess_train_image):
                rotation_angle = random.randint(-20, 20)
                color_shift = random.randint(0, 179)
            else :
                rotation_angle = 0
                color_shift = 0
            
            input_image_np = self.preprocess_input_image(image_file_path, rotation_angle, color_shift)            
            output_image_np = preprocess_output_image(image_file_path, rotation_angle, color_shift)
            binary_mask_np, body_parts_mask_np = preprocess_body_part_mask(image_file_path, rotation_angle)
            
            if (image_file_path.find("real") != -1):
                conditional_map = np.ones((image_size, image_size, 1), np.float32)
                uv_map_np = load_uv_map(image_file_path, rotation_angle)
            else :
                conditional_map = np.zeros((image_size, image_size, 1), np.float32)
                uv_map_np = np.zeros((image_size, image_size, 2), np.float32)
            
            # Image.fromarray(((input_image_np + 1) * 127.5).astype(np.uint8)).show()
            # Image.fromarray((output_image_np * 255).astype(np.uint8)).show()
            # Image.fromarray((body_parts_mask_np[..., 0] * 255).astype(np.uint8)).show()
            # Image.fromarray((binary_mask_np[..., 0] * 255).astype(np.uint8)).show()
            # calculate_uv_map(uv_map_np).show()
            
            source_image_np = np.concatenate([conditional_map, 
                                              input_image_np[..., 0:1] * body_parts_mask_np,
                                              input_image_np[..., 1:2] * body_parts_mask_np,
                                              input_image_np[..., 2:3] * body_parts_mask_np], axis=2)
            target_image_np = np.concatenate([binary_mask_np, output_image_np, uv_map_np], axis=2)
            
            source[i,] = source_image_np
            target[i,] = target_image_np

        return source, target


def shift_color(image_np, hue_shift):
    hsv = cv.cvtColor(image_np, cv.COLOR_RGB2HSV)
    hue = hsv[:,:,0]
    saturation = hsv[:,:,1]
    value = hsv[:,:,2]
    
    shifted_hue = (hue + hue_shift) % 180
    
    image_color_shifted_np = cv.merge([shifted_hue, saturation, value])
    image_color_shifted_np = cv.cvtColor(image_color_shifted_np, cv.COLOR_HSV2RGB)
    
    return image_color_shifted_np


def preprocess_train_image(file_path, rotation_angle=0, color_shift=0):
    image = Image.open(file_path)
    # image = image.convert('L').convert('RGB')
    image = image.rotate(rotation_angle, Image.NEAREST, fillcolor=(255, 255, 255))
    
    resized_image = image.resize((image_size, image_size), Image.NEAREST)
    resized_image_np = np.asarray(resized_image, dtype=np.uint8)
        
    image_rgb_shifted_np = shift_color(resized_image_np, color_shift)
    return image_rgb_shifted_np / 127.5 - 1


def preprocess_test_image(file_path, rotation_angle=0, color_shift=0):    
    image = Image.open(file_path)
    # image = image.convert('L').convert('RGB')
    
    resized_image = image.resize((image_size, image_size), Image.NEAREST)
    resized_image_np = np.asarray(resized_image, dtype=np.uint8)
    # preprocessed_image = imagenet_utils.preprocess_input(resized_image_np)

    return resized_image_np / 127.5 - 1


def preprocess_body_part_mask(image_file_path, rotation_angle=0):
    mask_file_path = image_file_path.replace("images", "masks").replace(".jpg", ".png")
    mask = Image.open(mask_file_path)
    mask = mask.rotate(rotation_angle, Image.NEAREST, fillcolor=255)
    
    mask = mask.getchannel(0)
    mask = mask.resize((output_image_size, output_image_size), Image.NEAREST)
    mask_np = np.asarray(mask)
    
    binary_mask_np = (1 - (mask_np == 255).astype(np.uint8)).astype(np.float32)
    binary_mask_np = np.expand_dims(binary_mask_np, -1)
    
    body_parts_mask_np = np.zeros((image_size, image_size, NUM_BODY_PARTS))
    
    for body_part_index in range(NUM_BODY_PARTS):
        body_parts_mask_np[..., body_part_index] = (mask_np == body_part_index).astype(np.uint8)
    
    return binary_mask_np, body_parts_mask_np


def load_uv_map(image_file_path, rotation_angle=0):
    uv_file_path = image_file_path.replace("images", "uv").replace(".jpg", ".npy")
    
    uv_map_np = np.load(uv_file_path)
    
    nans = np.isnan(uv_map_np)
    uv_map_np[nans] = 0.0

    uv_map_np = np.flip(np.swapaxes(uv_map_np, 0, 1), 0)
    
    rotated_uv_map = np.stack([
        np.array(Image.fromarray(uv_map_np[..., 0] * 255).rotate(rotation_angle, Image.NEAREST)) / 255,
        np.array(Image.fromarray(uv_map_np[..., 1] * 255).rotate(rotation_angle, Image.NEAREST)) / 255
    ], axis=-1)
    
    return rotated_uv_map
    

def preprocess_output_image(image_file_path, rotation_angle, color_shift=0):
    image = Image.open(image_file_path)
    # image = image.convert('L').convert('RGB')
    image = image.rotate(rotation_angle, Image.NEAREST, fillcolor=(255, 255, 255))
    
    resized_image = image.resize((output_image_size, output_image_size), Image.NEAREST)
    resized_image_np = np.asarray(resized_image)

    image_rgb_shifted_np = shift_color(resized_image_np, color_shift)
    return image_rgb_shifted_np / 255.0
    

class Conv1DBatch(Layer):
    
    def __init__(self, num_filters, kernel_size, strides=1, activation = "relu", *args, **kwargs):
        super(Conv1DBatch, self).__init__(*args, **kwargs)
    
        self.conv = Conv1D(num_filters, kernel_size, strides=strides, padding = 'same', kernel_initializer = 'he_normal')
        self.bn = BatchNormalization(trainable=False)
        self.activation = Activation(activation)
    
    def call(self, x):
        x = self.conv(x)
        if (batch_size > 1):
            x = self.bn(x)   
        x = self.activation(x)
     
        return x
    
class Conv2DBatch(Layer):
    
    def __init__(self, num_filters, kernel_size, strides=(1, 1), activation = "relu", *args, **kwargs):
        super(Conv2DBatch, self).__init__(*args, **kwargs)
    
        self.conv = Conv2D(num_filters, kernel_size, strides=strides, padding = 'same', kernel_initializer = 'he_normal')
        self.bn = BatchNormalization(trainable=False)
        self.activation = Activation(activation)
    
    def call(self, x):
        x = self.conv(x)
        
        if (batch_size > 1):
            x = self.bn(x) 
        x = self.activation(x)
               
        return x


class Conv2DTransposeBatch(Layer):
    
    def __init__(self, num_filters, kernel_size, strides=(1, 1), activation = "relu", *args, **kwargs):
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
    
    return x


def up(num_filters, layer, other_layer):
    deconv = Conv2DTransposeBatch(num_filters, 4, strides = (2,2))(layer)
    deconv = Conv2DBatch(num_filters, 3)(deconv)
    deconv = Add()([deconv, other_layer])
    deconv = Conv2DBatch(num_filters, 3)(deconv)
    # deconv = Conv2DBatch(num_filters, 3)(deconv)
    
    return deconv


def custom_loss(ytrue, ypred):
    return ytrue[..., 0] * mean_squared_error(ytrue[..., 1:], ypred[..., 1:])

image_shape = (image_size, image_size, NUM_COLORS * NUM_BODY_PARTS + 1)


x_in = Input(image_shape)
conv1 = Conv2DBatch(num_filters * 1, 3)(x_in)
conv2 = down(num_filters * 1, conv1) # 64
conv3 = down(num_filters * 2, conv2) # 32
conv4 = down(num_filters * 4, conv3) # 16
conv5 = down(num_filters * 8, conv4) # 8


up4 = up(num_filters * 8, conv5, conv4)
up3 = up(num_filters * 4, up4, conv3)
up2 = up(num_filters * 2, up3, conv2)
up1 = up(num_filters * 1, up2, conv1)

final_conv = Conv2D(4 + 2, kernel_size=(1, 1), padding="same", 
                    activation='sigmoid', name="final_conv")(up1)
                    
model = Model(inputs=x_in, outputs=final_conv)
model.summary()
model.compile(optimizer = Adam(), loss = custom_loss)
"""
from tensorflow.keras.utils import plot_model
plot_file_path = os.path.join(weights_folder, "model.png")
plot_model(model, to_file=plot_file_path, show_shapes=True)
"""
def calculate_uv_map(arr):
    def uv2rgb(uv):
        [u, v] = uv
        
        if (u == 0 and v == 0):
            return [0., 0., 0.]
        
        u_norm = u - 0.5
        v_norm = v - 0.5
        
        angle = math.atan2(v_norm, u_norm) + math.pi
        hue = angle / (2 * math.pi)
    
        # source: https://stackoverflow.com/questions/13211595/how-can-i-convert-coordinates-on-a-circle-to-coordinates-on-a-square
        circle_u = u_norm * math.sqrt(1 - 0.5 * v_norm**2)
        circle_v = v_norm * math.sqrt(1 - 0.5 * u_norm**2)
        
        length = math.sqrt(circle_u**2 + circle_v**2)
        saturation = length * 2
        
        c = Color()
        c.hsv = (hue, saturation, 1.0)
        
        return [c.r, c.g, c.b]
    
    uv_map_colored = np.apply_along_axis(uv2rgb, 2, arr)

    img = Image.fromarray(np.uint8(uv_map_colored * 255))
    
    return img

def predict():    
    for [image_folder, preprocess_input_image] in [
        [train_folder, preprocess_train_image], 
        [test_folder, preprocess_test_image]
    ]:
        for image_name in os.listdir(image_folder):
            image_file_path = os.path.join(image_folder, image_name)
            
            input_image_np = preprocess_input_image(image_file_path)
            binary_mask_np, body_parts_mask_np = preprocess_body_part_mask(image_file_path)
            
            conditional_map = np.ones((image_size, image_size, 1), np.float32)
            source_image_np = np.concatenate([conditional_map, 
                                              input_image_np[..., 0:1] * body_parts_mask_np,
                                              input_image_np[..., 1:2] * body_parts_mask_np,
                                              input_image_np[..., 2:3] * body_parts_mask_np], axis=2)
            source_image_np = np.expand_dims(source_image_np, axis=0)
                   
            result = model.predict(source_image_np)[0]
            
            # rgb = result[..., 1:4] * np.concatenate([binary_mask_np, binary_mask_np, binary_mask_np], axis=-1)
            # result_image = Image.fromarray(np.uint8(rgb * 255), "RGB")
            # image_output_folder = image_folder.replace("images", "images_encoded")
            # result_image.save(os.path.join(image_output_folder, image_name.replace(".jpg", ".png")))
            
            uv = result[..., 4:] * np.concatenate([binary_mask_np, binary_mask_np], axis=-1)
            uv_image = calculate_uv_map(uv)
            uv_output_folder = image_folder.replace("images", "uv_predicted")
            uv_image.save(os.path.join(uv_output_folder, image_name.replace(".jpg", ".png")))
            
            np.save(os.path.join(uv_output_folder, image_name.replace(".jpg", ".npy")), uv)


if (False):
    # model.load_weights(weights_file_path)
    
    training_data_gen = DataGenerator(train_folder, train_folder2, preprocess_train_image, batch_size)
    test_data_gen = DataGenerator(test_folder, test_folder, preprocess_test_image, 12)
    
    for i in range(300):
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
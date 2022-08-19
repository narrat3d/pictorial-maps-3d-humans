from config import NUM_FILTERS, IMAGE_SIZE, IMAGE_SIZE_HALF
from utils import Conv1DBatch, Conv2DBatch, Conv2DTransposeBatch, InterpolateBilinear

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate, Conv1D, UpSampling2D, Conv2D


def get_inputs(num_pose_points):
    image_2d = Input((IMAGE_SIZE, IMAGE_SIZE, 1))
    sdf_coords = Input((None, 3), dtype=tf.float32, name="sdf_coords")
    pose_coords = Input((num_pose_points, 3), dtype=tf.float32)
    
    sdf_coords_normalized = -1 + sdf_coords / IMAGE_SIZE_HALF
    
    return image_2d, sdf_coords, sdf_coords_normalized, pose_coords


def get_coords_shape(coords):
    input_shape = tf.shape(coords)
    batch_size = input_shape[0]
    num_samples = input_shape[1]
    
    return batch_size, num_samples


def encode_coords(coords):
    coords = Conv1DBatch(16, 1)(coords)
    coords = Conv1DBatch(64, 1)(coords)
    coords = Conv1DBatch(128, 1)(coords)
    
    return coords


def split_coords(xyz):
    xy = tf.stack([xyz[:, :, 0], xyz[:, :, 1]], axis=2)
    xz = tf.stack([xyz[:, :, 0], xyz[:, :, 2]], axis=2)
    yz = tf.stack([xyz[:, :, 1], xyz[:, :, 2]], axis=2)
    
    return xy, xz, yz
 

def encode_image(image_2d, batch_size, global_features=True):
    conv1 = Conv2DBatch(NUM_FILTERS, 3)(image_2d)

    # image_size = 32
    pool1 = Conv2DBatch(NUM_FILTERS * 2, 3, strides = (2,2))(conv1)
    conv2 = Conv2DBatch(NUM_FILTERS * 2, 3)(pool1)

    # image_size = 16
    pool2 = Conv2DBatch(NUM_FILTERS * 4, 3, strides = (2,2))(conv2)
    conv3 = Conv2DBatch(NUM_FILTERS * 4, 3)(pool2)
    
    # image_size = 8
    pool3 = Conv2DBatch(NUM_FILTERS * 8, 3, strides = (2,2))(conv3)
    conv4 = Conv2DBatch(NUM_FILTERS * 8, 3)(pool3)

    # image_size = 4
    pool4 = Conv2DBatch(NUM_FILTERS * 16, 3, strides = (2,2))(conv4)
    conv5 = Conv2DBatch(NUM_FILTERS * 16, 3)(pool4)

    # image_size = 2
    pool5 = Conv2DBatch(NUM_FILTERS * 32, 3, strides = (2,2))(conv5)
    conv6 = Conv2DBatch(NUM_FILTERS * 32, 3)(pool5)
 
    if (not global_features):
        # image_size = 1
        pool6 = Conv2DBatch(NUM_FILTERS * 64, 3, strides = (2,2))(conv6)
        conv7 = Conv2DBatch(NUM_FILTERS * 64, 3)(pool6)       
        
        return [conv7, conv6, conv5, conv4, conv3, conv2, conv1] 
 
     
    global_features = tf.reshape(conv6, (batch_size, 1, NUM_FILTERS * 32 * 2 * 2))
    global_features = Conv1D(256, 1, activation="relu", kernel_initializer = 'he_normal')(global_features)

    return global_features, [conv5, conv4, conv3, conv2, conv1]


def get_local_features_from_image(feature_maps, coords_2d, batch_size, num_samples):
    [conv5, conv4, conv3, conv2, conv1] = feature_maps
    
    conv5_up = UpSampling2D(size=(16,16), interpolation="bilinear")(conv5)
    conv4_up = UpSampling2D(size=(8,8), interpolation="bilinear")(conv4)
    conv3_up = UpSampling2D(size=(4,4), interpolation="bilinear")(conv3) 
    conv2_up = UpSampling2D(size=(2,2), interpolation="bilinear")(conv2)
    
    conv_stack = Concatenate()([conv5_up, conv4_up, conv3_up, conv2_up, conv1])
    
    conv_stack = InterpolateBilinear()(conv_stack, coords_2d)
    conv_stack = tf.reshape(conv_stack, (batch_size, num_samples, NUM_FILTERS*(1 + 2 + 4 + 8 + 16)))
    
    return conv_stack


def decode_features(features, name=None):
    features = Conv1DBatch(512, 1)(features)
    features = Conv1DBatch(256, 1)(features)
    
    sdf = Conv1D(1, 1, name=name, padding="same", kernel_initializer = 'he_normal')(features)
    
    return sdf
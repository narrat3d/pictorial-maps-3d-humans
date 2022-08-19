import json
import tensorflow as tf
from mathutils import Vector
import numpy as np
from PIL import Image
from config import NUM_BODY_PARTS, IMAGE_SIZE, INCLUDE_BODY_PARTS
import os


def calc_vec(p2, p1):
    [x1, _, z1] = p1
    [x2, _, z2] = p2
    
    vec = Vector((x2 - x1, z2 - z1))
    vec.normalize()
    
    return vec
    

def resize_array(arr, new_size):
    num_dims = len(arr.shape)
    
    if (num_dims == 2): # width x height
        arr = arr[tf.newaxis, ..., tf.newaxis]
    elif (num_dims == 3): # width x height x channels
        arr = arr[tf.newaxis, ...]
    # else batch x width x height x channels    
    
    tensor = tf.constant(arr)
    resized_tensor = tf.image.resize(tensor, new_size)
    resized_arr = resized_tensor.numpy()
    
    if (num_dims == 2):
        result = resized_arr[0, ..., 0]
    elif (num_dims == 3):
        result = resized_arr[0, ...]
                 
    return result
  

def load_from_cache(file_path, image_size):
    # if (file_path in cache):
    #     return cache[file_path]
    
    arr = np.load(file_path)["arr_0"]
    
    if (image_size == None):
        image_size = arr.shape[0]
    else :
        arr = resize_array(arr, (image_size, image_size))    
    # cache[file_path] = arr
    
    return arr, image_size


def load_depth_map(depth_map_path, image_size=None):
    depth_map, image_size = load_from_cache(depth_map_path, image_size)
    
    depth_map *= -1
    
    min_depth = np.nanmin(depth_map)
    max_depth = np.nanmax(depth_map)
    
    # normalize
    depth_map_normalized = (depth_map - min_depth) / (max_depth - min_depth)
    
    depth_map = np.nan_to_num(depth_map_normalized) 
    
    depth_map = np.reshape(depth_map, (1, image_size, image_size, 1))

    return depth_map, depth_map_normalized


def load_uv_map(path, image_size=None):
    uv_map, image_size = load_from_cache(path, image_size)
    u_map = uv_map[..., 0]
    nans = np.isnan(u_map)
              
    uv_map = np.nan_to_num(uv_map)
                    
    mask = np.copy(u_map)
    mask[nans] = 0.0
    mask[~nans] = 1.0
    
    mask = np.expand_dims(mask, axis=-1)  
    
    uv_map_and_mask = np.concatenate([mask, uv_map], axis=2)
    uv_map_and_mask = np.expand_dims(uv_map_and_mask, axis=0)
    
    return uv_map_and_mask


def load_body_parts_mask(path, image_size=None):
    body_parts_image = Image.open(path)
    body_parts_image = body_parts_image.getchannel(0)
    
    if (image_size == None):
        image_size = body_parts_image.width
    else :
        body_parts_image = body_parts_image.resize((image_size, image_size), resample=Image.NEAREST)
        
    body_parts_image_np = np.flip(np.swapaxes(np.asarray(body_parts_image), 0, 1), 1) 
    
    body_parts_mask_np = np.zeros((image_size, image_size, NUM_BODY_PARTS))
    
    for pose_index in range(NUM_BODY_PARTS):
        body_parts_mask_np[:,:,pose_index] = (body_parts_image_np == pose_index).astype(np.uint8)
    
    body_parts_mask_np = np.reshape(body_parts_mask_np, (1, image_size, image_size, NUM_BODY_PARTS))
    
    return body_parts_mask_np


def load_texture(path, image_size=None):
    texture_image = Image.open(path)
    
    if (image_size != None):
        texture_image = texture_image.resize((image_size, image_size), resample=Image.NEAREST)
        
    texture_arr = np.flip(np.swapaxes(np.asarray(texture_image), 0, 1), 1)
    
    return texture_arr


def load_input_data(subfolder_path, view):
    depth_map_path = os.path.join(subfolder_path, "depth_%s.npz" % view)
    depth_map, depth_map_normalized = load_depth_map(depth_map_path, IMAGE_SIZE)
    
    if INCLUDE_BODY_PARTS:
        body_parts_mask_path = os.path.join(subfolder_path, "body_parts_%s_mask.png" % view)
        body_parts_mask = load_body_parts_mask(body_parts_mask_path, IMAGE_SIZE)
        depth_map = depth_map * body_parts_mask
    
    return depth_map, depth_map_normalized


def load_output_data(subfolder_path, view):
    uv_map_path = os.path.join(subfolder_path, "uv_%s.npz" % view)
    uv_map = load_uv_map(uv_map_path, IMAGE_SIZE)
    
    return uv_map
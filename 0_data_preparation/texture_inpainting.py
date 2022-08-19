'''
prepares the data for coordinate-based inpainting
adds a grey rectangular frame below the figure image (to resemble the original data).
converts the UV coordinates
'''
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import math
from mathutils import Color
import os
import argparse

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

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


def convert_uv(arr, size):
    arr[:,:,1] = 1 - arr[:,:,1]
    arr = resize_array(arr, size)
    arr = arr * 2 - 1
    arr = np.swapaxes(arr, 0, 1)
    arr = np.flip(arr, 0)
    
    return arr


def convert(input_folder, inpainting_folder):
    for folder in ["source_img", "source_uv", "target_uv", "target_img"]:
        folder_path = os.path.join(inpainting_folder, folder)
        os.makedirs(folder_path, exist_ok=True)

    size = (256, 256)
    texture_path = os.path.join(input_folder, "body_parts_front_cropped_texture.png")
    
    uv_source_path = os.path.join(input_folder, "uv_front_cropped.npy")
    uv_source_arr = np.load(uv_source_path)
    uv_mask_arr = np.isnan(uv_source_arr[..., 1])
    uv_mask_arr = np.swapaxes(uv_mask_arr, 0, 1)
    uv_mask_arr = np.flip(uv_mask_arr, 0)
    
    folder_name = os.path.basename(input_folder)

    white_bg = Image.new("RGB", size, (255, 255, 255))
    grey_bg = Image.new("RGB", [math.floor(size[0]*0.7), size[1]], (230, 230, 230))
    
    white_bg.paste(grey_bg, box=(math.floor(size[0]*0.15), 0))
    
    texture_img = Image.open(texture_path)
    texture_img = texture_img.resize(size)
    
    texture_img_arr = np.array(texture_img)
    texture_img_arr[uv_mask_arr] = (255, 255, 255, 0)
    texture_img = Image.fromarray(texture_img_arr)
        
    body_mask = texture_img.getchannel(3)
    texture_img = texture_img.convert("RGB")
    
    white_bg.paste(texture_img, mask=body_mask)
    white_bg.save(os.path.join(inpainting_folder, "source_img", "%s.jpg" % folder_name))

    uv_source_arr = convert_uv(uv_source_arr, size)
    np.save(os.path.join(inpainting_folder, "source_uv", "%s.npy" % folder_name), uv_source_arr)
    
    for view in ["front_cropped", "front", "left", "back", "right"]:
        uv_target_path = os.path.join(input_folder, "uv_%s.npy" % view)
        uv_target_arr = np.load(uv_target_path)
        uv_target_arr = convert_uv(uv_target_arr, size)
        np.save(os.path.join(inpainting_folder, "target_uv", "%s_%s.npy" % (folder_name, view)), uv_target_arr)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_folder", default=r"E:\CNN\implicit_functions\characters\out")
    parser.add_argument("--tmp_folder", default= r"E:\CNN\implicit_functions\characters\tmp")
    parser.add_argument("--sub_folders", default="")
    args = parser.parse_args()
    
    root_folder = args.out_folder
    inpainting_folder = os.path.join(args.tmp_folder, "textures")
    os.makedirs(inpainting_folder, exist_ok=True)
    subfolder_names = os.listdir(root_folder) if args.sub_folders == "" else args.sub_folders.split("|")
        
    for subfolder in subfolder_names:
        subfolder_path = os.path.join(root_folder, subfolder)
        convert(subfolder_path, inpainting_folder)
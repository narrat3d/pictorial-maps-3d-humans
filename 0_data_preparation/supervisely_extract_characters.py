'''
crops images and masks to a square bounding box containing the figure as well as 
transforms the skeleton points. the code is taken from an instance segmentation project, 
therefore it is a bit more complicated than needed
'''
import os
import json
from PIL import Image
import tensorflow as tf
import numpy as np
import argparse
import shutil
import math

parser = argparse.ArgumentParser()
parser.add_argument("--tmp_folder", default=r"E:\CNN\implicit_functions\characters\tmp")
parser.add_argument("--out_folder", default=r"E:\CNN\implicit_functions\characters\output")
parser.add_argument("--sub_folders", default="")
args = parser.parse_args()

input_folder = args.tmp_folder
output_folder = args.out_folder
subfolder_names = None if args.sub_folders == "" else args.sub_folders.split("|")

target_size = (600, 600)


def mkdir_if_not_exists(path):
    return os.makedirs(path, exist_ok=True)


def shift_keypoints(keypoints, x, y):
    keypoints_shifted = {}
    
    for key, coords in keypoints.items():
        keypoints_shifted[key] = [coords[0] + x, coords[1] + y]
        
    return keypoints_shifted


def scale_keypoints(keypoints, old_size, new_size):
    keypoints_shifted = {}
    
    for key, coords in keypoints.items():
        keypoints_shifted[key] = [
            coords[0] / old_size[0] * new_size[0], 
            coords[1] / old_size[1] * new_size[1]
        ]
       
    return keypoints_shifted


def rect_to_square(image):
    mode = image.mode
    
    if (mode == "L"):
        color = 255
    elif (mode == "RGB"):
        color = (255, 255, 255)
    elif (mode == "RGBA"):
        color = (255, 255, 255, 0)
    
    if (image.width > image.height):
        square_image = Image.new(mode, (image.width, image.width), color)
        offset = (0, round((image.width - image.height) / 2))
        square_image.paste(image, offset)
    else :
        square_image = Image.new(mode, (image.height, image.height), color)
        offset = (round((image.height - image.width) / 2), 0)
        square_image.paste(image, offset)    
    
    return [square_image, offset]


def filter_keypoints(keypoints, width, height):
    for key, coords in zip(list(keypoints.keys()), list(keypoints.values())):
        if (coords[0] < 0 or coords[0] > width or coords[1] < 0 or coords[1] > height):
            del keypoints[key]


def get_body_part_instances(mask_file_path, keypoint_file_path):
    body_part_instances = []
    bboxes = []
    instance_mask_images = []
    
    mask_map_image = Image.open(mask_file_path)
    instances_image = mask_map_image.getchannel(0)
    body_parts_image = mask_map_image.getchannel(1)
    
    keypoints = json.load(open(keypoint_file_path))
    number_of_characters = len(keypoints)
    
    for i in range(number_of_characters):
        instance_mask_image = Image.eval(instances_image, lambda x: ((x == i) and 255) or 0) 

        instance_body_parts_image = Image.new("L", body_parts_image.size, 255)
        instance_body_parts_image.paste(body_parts_image, instance_mask_image)
        
        body_part_instances.append(instance_body_parts_image)
        bboxes.append(instance_mask_image.getbbox())
        instance_mask_images.append(instance_mask_image)
        
    return (body_part_instances, keypoints, bboxes, instance_mask_images)


def resize_array(arr, new_size):
    arr = arr[tf.newaxis, ...]  
    
    tensor = tf.constant(arr)
    resized_tensor = tf.image.resize(tensor, new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    resized_arr = resized_tensor.numpy()
    
    result = resized_arr[0, ...]
           
    return result


def match_instances(image, gt_masks, bboxes, instance_mask_images, keypoints, figure_output_folder):
    for instance_number, matched_gt_mask in enumerate(gt_masks):
        bbox = bboxes[instance_number]
        matched_keypoints = keypoints[instance_number]
        instance_mask_image = instance_mask_images[instance_number]
        
        image_crop = image.crop(bbox)
        detected_mask_crop = instance_mask_image.crop(bbox)
        
        image_crop_masked = Image.new("RGB", image_crop.size, (255, 255, 255))
        image_crop_masked.paste(image_crop, detected_mask_crop)        
        
        gt_mask_crop = matched_gt_mask.crop(bbox)
        alpha_channel = gt_mask_crop.point(lambda p: 255 if p < 14 else 0)
        # just for compatibility purposes (Blender seems to export RGB PNGs)
        gt_mask_crop = Image.merge("RGB", (gt_mask_crop, gt_mask_crop, gt_mask_crop))
        
        [r, g, b] = image_crop_masked.split()
        image_crop_masked = Image.merge("RGBA", (r, g, b, alpha_channel))
        
        shifted_keypoints = shift_keypoints(matched_keypoints, -bbox[0], -bbox[1])
        filter_keypoints(shifted_keypoints, image_crop.width, image_crop.height)
        
        [image_crop_masked, offset] = rect_to_square(image_crop_masked)
        [gt_mask_crop, offset] = rect_to_square(gt_mask_crop)
        shifted_keypoints = shift_keypoints(shifted_keypoints, offset[0], offset[1])
        scaled_keypoints = scale_keypoints(shifted_keypoints, image_crop_masked.size, target_size)
        

        image_crop_masked_np = np.array(image_crop_masked, np.float32)
        image_crop_masked_alpha_np = image_crop_masked_np[:, :, 3:]
        image_crop_masked_alpha_np[image_crop_masked_alpha_np == 0.] = np.nan
        image_crop_masked_np = image_crop_masked_np * np.concatenate([image_crop_masked_alpha_np / 255.] * 4, axis=2)
        image_crop_masked_np = resize_array(image_crop_masked_np, target_size)
        image_crop_masked_np[:, :, 0:3][np.isnan(image_crop_masked_np[:, :, 0:3])] = 255.
        image_crop_masked_np[:, :, 3][np.isnan(image_crop_masked_np[:, :, 3])] = 0.
        image_crop_masked = Image.fromarray(np.uint8(image_crop_masked_np))
        
        gt_mask_crop_np = np.array(gt_mask_crop, np.float32)
        gt_mask_crop_np = gt_mask_crop_np * image_crop_masked_alpha_np / 255.
        gt_mask_crop_np = resize_array(gt_mask_crop_np, target_size)
        gt_mask_crop_np[np.isnan(gt_mask_crop_np)] = 255.
        gt_mask_crop = Image.fromarray(np.uint8(gt_mask_crop_np))

        image_crop_masked.save(os.path.join(figure_output_folder, "body_parts_front_cropped_texture.png"))      
        gt_mask_crop.save(os.path.join(figure_output_folder, "body_parts_front_cropped_mask.png"))
        json.dump(scaled_keypoints, open(os.path.join(figure_output_folder, "skeleton_2d.json"), "w"))


def extract_characters(input_folder, output_folder):
    image_input_folder = os.path.join(input_folder, "images")
    mask_input_folder = os.path.join(input_folder, "masks")
    keypoint_input_folder = os.path.join(input_folder, "keypoints")
    
    mkdir_if_not_exists(output_folder)
    
    image_names = os.listdir(image_input_folder)

    for image_name in image_names:
        image_name_without_ext = os.path.splitext(image_name)[0]

        if subfolder_names != None and not image_name_without_ext in subfolder_names:
            continue
    
        print(image_name)
        image_file_path = os.path.join(image_input_folder, image_name)
        image = Image.open(image_file_path)
    
        mask_file_path = os.path.join(mask_input_folder, image_name_without_ext + ".png")
        keypoint_file_path = os.path.join(keypoint_input_folder, image_name_without_ext + ".json")
        
        figure_output_folder = os.path.join(output_folder, image_name_without_ext)
        mkdir_if_not_exists(figure_output_folder)
        
        (body_part_instances, keypoint_instances, bboxes, instance_mask_images) = get_body_part_instances(mask_file_path, keypoint_file_path)        
        match_instances(image, body_part_instances, bboxes, instance_mask_images, keypoint_instances, figure_output_folder)
        

if __name__ == '__main__':
    extract_characters(input_folder, output_folder)
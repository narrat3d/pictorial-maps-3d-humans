import os
from PIL import Image
import numpy as np
from texture_inpainting import resize_array
import shutil
import json
import math
from mathutils import Color

IMAGE_SIZE = 256
image_size = 64


BODY_PARTS = {
    "torso": 0,
    "head": 1,
    "right_arm": 2,
    "right_leg": 3,
    "left_leg": 4,
    "left_arm": 5
}

BODY_PART_MAPPING = [
    "torso", "head", 
    "right_arm", "right_arm", "right_arm", 
    "right_leg", "right_leg", "right_leg", 
    "left_leg", "left_leg", "left_leg", 
    "left_arm", "left_arm", "left_arm"
]

mapping = list(map(lambda part_name: BODY_PARTS[part_name], BODY_PART_MAPPING)) + (255 - len(BODY_PART_MAPPING)) * [np.nan] + [255]
mapping = np.array(mapping)

'''
copy textured figures from SMPL-X to training folders and inferred figures to test folders
'''
def transfer_training_data(input_folder, output_folder, image_type, view):
    for figure_folder in os.listdir(input_folder):        
        output_file_name = figure_folder + "_" + view
        mask_path = os.path.join(input_folder, figure_folder, f"body_parts_{view}_mask.png")
        
        if (not os.path.exists(mask_path)):
            continue
        
        mask = Image.open(mask_path)
        mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        mask_np = np.array(mask)

        mapped_mask_np = mapping[mask_np]
        mapped_mask_np = Image.fromarray(mapped_mask_np.astype(np.uint8))
        mapped_mask_np.save(os.path.join(output_folder, "masks_%s" % image_type, output_file_name + ".png"))
        
        binary_mask_np = 1 - (mask_np == 255).astype(np.uint8)
        
        original_image_path = os.path.join(input_folder, figure_folder, f"body_parts_{view}_texture.png")
        original_image = Image.open(original_image_path)
        original_image = original_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        original_image_np = np.array(original_image)[..., 0:3]
        
        original_image_np = original_image_np * binary_mask_np + (1 - binary_mask_np) * 255
        
        original_image = Image.fromarray(original_image_np)
        original_image.save(os.path.join(output_folder, "images_%s_output" % image_type, output_file_name + ".jpg"))

        if (image_type == "pictorial"):
            inpainted_image_path = os.path.join(input_folder, figure_folder, "body_parts_%s_texture.png" % view.replace("_cropped", ""))
            
            inpainted_image = Image.open(inpainted_image_path)
            inpainted_image = inpainted_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
            inpainted_image_np = np.array(inpainted_image)[..., 0:3]
    
            cropped_inpainted_image_np = inpainted_image_np * binary_mask_np + (1 - binary_mask_np) * 255
            cropped_inpainted_image = Image.fromarray(cropped_inpainted_image_np.astype(np.uint8))
            cropped_inpainted_image.save(os.path.join(output_folder, "images_%s_input" % image_type, output_file_name + ".jpg"))
        
        if (image_type == "real"):    
            uv_file_name = f"uv_{view}.npz"
        else :
            uv_file_name = f"uv_{view}.npy"
        
        uv_file_path = os.path.join(input_folder, figure_folder, uv_file_name)
        uv_map = np.load(uv_file_path)
        
        if (image_type == "real"):
            uv_map = uv_map["arr_0"]
        
        uv_map_resized = resize_array(uv_map, (IMAGE_SIZE, IMAGE_SIZE))
        
        np.save(os.path.join(output_folder, "uv", output_file_name + ".npy"), uv_map_resized)
        
        

# for view in ["front_cropped", "front", "back", "right", "left"]:
    # transfer_training_data(r"C:\Users\raimund\Downloads\tmp2", r"E:\CNN\masks\data\textures_data\train_real", "real", view)
    # transfer_training_data(r"E:\Repositories\pictorial-maps-3d-humans\data\out", r"E:\CNN\masks\textures_data\test_pictorial2", "pictorial", view)

    
"""
# filter by number of body parts

for mask_name in os.listdir(r"E:\CNN\masks\data\figures\real\masks"):
    mask = Image.open(os.path.join(r"E:\CNN\masks\data\figures\real\masks", mask_name))
    mask = mask.getchannel(0)
    
    hist = mask.histogram()
    # num_body_parts = np.sum((np.array(hist) != 0).astype(np.uint8)) - 1
    # print(num_body_parts)
   
    if (hist[1] > 0): # num_body_parts == 6
        shutil.copy(os.path.join(r"E:\CNN\masks\data\figures\real\images", mask_name.replace(".png", ".jpg")), 
                    r"E:\Repositories\cartoonize\test_code\test_images")
"""

'''
display a UV image file
'''
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

# uv_map = np.load(r"E:\CNN\masks\data\figures_pictorial\test\heads_uv\braun_hogenberg_iii_4_b1_back.npy")
# uv_map = np.swapaxes(np.flip(uv_map, 1), 0, 1)
# calculate_uv_map(uv_map).save(r"E:\CNN\masks\data\figures_pictorial\1.png")


'''
extract head images, masks and UV coordinates from whole body
'''

data_type = "train" # "test"

for image_name in os.listdir(r"E:\CNN\masks\data\textures_data\test_real\images"):
    input_image_path = os.path.join(r"E:\CNN\masks\data\textures_data\test_real\images", image_name)
    output_image_path = os.path.join(r"E:\CNN\masks\data\textures_data\train_pictorial\images", image_name)
    mask_path = os.path.join(r"E:\CNN\masks\data\textures_data\test_real\masks", image_name.replace(".jpg", ".png"))
    keypoints_path = os.path.join(r"E:\CNN\masks\data\textures_data\test_real\keypoints", image_name.replace(".jpg", ".json"))
    uv_path = os.path.join(r"E:\CNN\masks\data\textures_data\train_pictorial\uv", image_name.replace(".jpg", ".npy"))
    
    current_view = None
    
    # for folder 'test_pictorial2'
    for view in ["front_cropped", "front", "left", "right", "back"]:
        ending = "_%s.jpg" % view
        
        if (image_name.find(ending) != -1):
            image_name_without_view = image_name.replace(ending, "")
            current_view = view
            break

    with open(keypoints_path) as keypoints_file:
        keypoints = json.load(keypoints_file)

    neck_point = keypoints.get("8")
    head_point = keypoints.get("9")
    
    if (neck_point == None or head_point == None):
        continue

    # rotate 3D keypoints in folder 'test_pictorial2' 
    if (current_view in [None, "front_cropped", "front"]):
        angle = -math.atan2(neck_point[0] - head_point[0], neck_point[1] - head_point[1]) / math.pi * 180
    elif (current_view == "back"):
        angle = math.atan2(neck_point[0] - head_point[0], neck_point[1] - head_point[1]) / math.pi * 180
    elif (current_view == "right"):
        angle = -math.atan2(neck_point[2] - head_point[2], neck_point[1] - head_point[1]) / math.pi * 180
    elif (current_view == "left"):
        angle = math.atan2(neck_point[2] - head_point[2], neck_point[1] - head_point[1]) / math.pi * 180

    mask = Image.open(mask_path)
    mask = mask.getchannel(0)
    head_mask = mask.point(lambda p: ((p == 1) and 255) or 0)   
    
    head_bbox = head_mask.getbbox()
    cropped_head = head_mask.crop(head_bbox)
    
    square_size = max(cropped_head.width, cropped_head.height)
    
    if (data_type == "train" and square_size < 32):
        continue
    
    square_head_mask = Image.new("L", (square_size, square_size), 0)
    head_offset_x = round((square_size - cropped_head.width) / 2)
    head_offset_y = round((square_size - cropped_head.height) / 2)
    
    square_head_mask.paste(cropped_head, (head_offset_x, head_offset_y))
    square_head_mask_normalized = square_head_mask.rotate(angle, fillcolor=0)
    square_head_mask_normalized = square_head_mask_normalized.resize((image_size, image_size), Image.NEAREST)
    square_head_mask_normalized.save(os.path.join(r"E:\CNN\masks\data\figures_pictorial\%s\heads_masks" % data_type, image_name.replace(".jpg", ".png")))
    
    for [image_path, output_folder] in [[input_image_path, r"E:\CNN\masks\data\figures_pictorial\%s\heads_input" % data_type], 
                                      [output_image_path, r"E:\CNN\masks\data\figures_pictorial\%s\heads_output" % data_type]]:
        head_image = Image.open(image_path)
        head_image = head_image.resize(head_mask.size, Image.CUBIC)
        masked_head_image = Image.new("RGB", head_mask.size, (255, 255, 255))
        masked_head_image.paste(head_image, (0, 0), head_mask)
        masked_head_image = masked_head_image.crop(head_bbox)
        square_head_image = Image.new("RGB", (square_size, square_size), (255, 255, 255))
        square_head_image.paste(masked_head_image, (round((square_size - cropped_head.width) / 2), round((square_size - cropped_head.height) / 2)))
        square_head_image = square_head_image.rotate(angle, fillcolor=(255, 255, 255))
        square_head_image = square_head_image.resize((image_size, image_size), Image.NEAREST)
        square_head_image.save(os.path.join(output_folder, image_name))
    
    """
    # paste resynthesized texture back to whole figure
    resynthesized_head_image = Image.open(os.path.join(r"E:\CNN\masks\data\figures_pictorial\%s\results" % data_type, image_name.replace(".jpg", ".png")))
    resynthesized_head_image = resynthesized_head_image.rotate(-angle, Image.NEAREST, fillcolor=(255, 255, 255))
    resynthesized_head_image = resynthesized_head_image.resize((square_size, square_size), Image.NEAREST)
    
    square_head_mask_normalized = square_head_mask_normalized.getchannel(0)
    square_head_mask_normalized = square_head_mask_normalized.rotate(-angle, Image.NEAREST, fillcolor=0)
    square_head_mask_normalized = square_head_mask_normalized.resize((square_size, square_size), Image.NEAREST)
                                                               
    head_image.paste(resynthesized_head_image, (head_bbox[0] - head_offset_x, head_bbox[1] - head_offset_y), square_head_mask_normalized)
    head_image.show()

    head_image_np = np.array(head_image)
    mask_np = np.array(mask)
    
    binary_mask = 1 - (mask_np == 255).astype(np.uint8)
    binary_mask_rgb = np.stack([binary_mask, binary_mask, binary_mask], -1)
    
    masked_image_np = binary_mask_rgb * head_image_np + (1 - binary_mask_rgb) * 255
    masked_image_np_rgba = np.concatenate([masked_image_np, binary_mask_rgb[..., 0:1] * 255], -1)
    
    Image.fromarray(masked_image_np_rgba.astype(np.uint8)).save(os.path.join(r"E:\CNN\masks\data\figures_pictorial\%s\results" % data_type, image_name.replace(".jpg", ".png")))
    """
    
    uv_map = np.load(uv_path)
    # only needed for testing data
    uv_map = np.flip(np.swapaxes(uv_map, 0, 1), 0)
    
    u = uv_map[..., 0] * 255
    v = uv_map[..., 1] * 255
    
    uv_map_cropped = []
    
    for uv_coord in [u, v]:
        uv_coord_image = Image.fromarray(uv_coord.astype(np.uint8), "L")
        uv_coord_image = uv_coord_image.resize(head_mask.size, Image.NEAREST)
        masked_head_image = Image.new("L", head_mask.size, 0)
        masked_head_image.paste(uv_coord_image, (0, 0), head_mask)
        masked_head_image = masked_head_image.crop(head_bbox)
        square_head_image = Image.new("L", (square_size, square_size), 0)
        square_head_image.paste(masked_head_image, (round((square_size - cropped_head.width) / 2), round((square_size - cropped_head.height) / 2)))
        square_head_image = square_head_image.rotate(angle, fillcolor= 0)
        square_head_image = square_head_image.resize((image_size, image_size), Image.NEAREST)
        uv_map_cropped.append(np.array(square_head_image) / 255)
        
    uv_map_cropped = np.stack(uv_map_cropped, -1)

    np.save(os.path.join(r"E:\CNN\masks\data\figures_pictorial\%s\heads_uv" % data_type, image_name.replace(".jpg", ".npy")), uv_map_cropped)
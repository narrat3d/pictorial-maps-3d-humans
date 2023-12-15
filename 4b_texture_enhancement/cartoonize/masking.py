from PIL import Image
import numpy as np
import os

image_size = 256

image_folder = r"E:\Repositories\cartoonize\test_code\cartoonized_images"
output_folder = r"E:\Repositories\cartoonize\test_code\cartoonized_figures"
mask_file = r"E:\CNN\implicit_functions\characters\output\%s\body_parts_%s_mask.png"
output_texture_file_name = "body_parts_%s_cartoon_texture.png"

for image_name in os.listdir(image_folder):
    print(image_name)
    
    current_view = None
    image_name_without_view = None
    
    for view in ["front_cropped", "front", "left", "right", "back"]:
        ending = "_%s.jpg" % view
        
        if (image_name.find(ending) != -1):
            image_name_without_view = image_name.replace(ending, "")
            current_view = view
            break
    
    mask = Image.open(mask_file % (image_name_without_view, current_view))
    mask = mask.getchannel(0)
    mask = mask.resize((image_size, image_size))
    mask_np = np.array(mask)
    binary_mask = 1 - (mask_np == 255).astype(np.uint8)
    binary_mask_rgb = np.stack([binary_mask, binary_mask, binary_mask], -1)
    
    image = Image.open(os.path.join(image_folder, image_name))
    image_np = np.array(image)
    
    masked_image_np = binary_mask_rgb * image_np + (1 - binary_mask_rgb) * 255
    masked_image_np_rgba = np.concatenate([masked_image_np, binary_mask_rgb[..., 0:1] * 255], -1)
    
    figure_folder = os.path.join(output_folder, image_name_without_view)
    os.makedirs(figure_folder, exist_ok=True)
    
    output_texture_file_path = os.path.join(figure_folder, output_texture_file_name % current_view)
    Image.fromarray(masked_image_np_rgba.astype(np.uint8)).save(output_texture_file_path)
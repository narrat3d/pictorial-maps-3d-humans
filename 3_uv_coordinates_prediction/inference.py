from model import create_model
from config import IMAGE_SIZE, NUM_FILTERS, NUM_BODY_PARTS
import argparse
import os
import numpy as np
from data_loader import load_texture, load_input_data
from visualize import calculate_uv_map

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default="output")
parser.add_argument("--out_folder", default=r"E:\CNN\implicit_functions\characters\output")
parser.add_argument("--sub_folders", default="")
args = parser.parse_args()

root_folder = args.out_folder
model_folder = args.model_folder  
subfolder_names = os.listdir(root_folder) if args.sub_folders == "" else args.sub_folders.split("|")

model = create_model(IMAGE_SIZE, NUM_BODY_PARTS, NUM_FILTERS)
model.load_weights(os.path.join(model_folder, "weights_best.hdf5"))

for subfolder in subfolder_names:
	subfolder_path = os.path.join(root_folder, subfolder)
	
	print(subfolder_path)
	
	for view in ["front_cropped", "front", "left", "back", "right"]:
		depth_map, normalized_depth_map = load_input_data(subfolder_path, view.replace("_cropped", ""))

		mask = np.ones_like(normalized_depth_map)
		mask[np.isnan(normalized_depth_map)] = np.nan
		
		# make sure that uv-coordinate has a matching texture coordinate
		if (view in ["front_cropped", "back_cropped"]):
			front_texture_path = os.path.join(subfolder_path, "body_parts_front_cropped_texture.png")
			texture_arr = load_texture(front_texture_path, IMAGE_SIZE)
			alpha_channel = texture_arr[..., 3]
			
			if (view == "back_cropped"):
				alpha_channel = np.flip(alpha_channel, axis=0)
			
			mask[alpha_channel == 0] = np.nan
		
		mask = np.expand_dims(mask, axis=2)
		stacked_mask = np.concatenate([mask, mask], axis=2)
		
		uv_map = model(depth_map)
		masked_uv_map = stacked_mask * uv_map[0, ..., 1:].numpy()
		uv_map_image = calculate_uv_map(masked_uv_map)
		uv_map_image.save(os.path.join(subfolder_path, "uv_%s.png" % view))
		
		uv_map_path = os.path.join(subfolder_path, "uv_%s.npy" % view)
		np.save(uv_map_path, masked_uv_map)
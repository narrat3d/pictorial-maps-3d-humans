'''
rasterizes the polygon annotations and processes the skeleton points from supervise.ly,
stores them together with the images in a temporary folder
'''
import json
import os
import shutil
import tarfile
from PIL import Image, ImageDraw
import numpy as np
import argparse

KEYPOINT_MAPPING = {
    "RA": 0,
    "RK": 1,
    "RP": 2,
    "LP": 3,
    "LK": 4,
    "LA": 5,
    "P": 6,
    "T": 7,
    "N": 8,
    "H": 9,
    "RW": 10,
    "RE": 11,
    "RS": 12,
    "LS": 13,
    "LE": 14,
    "LW": 15,
    "RF": 16,
    "LF": 17, 
    "RH": 18,
    "LH": 19,
    "E": 20
}

BODY_PART_MAPPING = {
    "torso": 0,
    "head": 1,
    "right_upper_arm": 2,
    "right_lower_arm": 3,
    "right_hand": 4,
    "right_upper_leg": 5,
    "right_lower_leg": 6,
    "right_foot": 7,
    "left_upper_leg": 8,
    "left_lower_leg": 9,
    "left_foot": 10,
    "left_upper_arm": 11,
    "left_lower_arm": 12,
    "left_hand": 13 
}

body_part_names = list(BODY_PART_MAPPING.keys())

SKELETON_CLASS_NAME = "Skeleton"

keypoint_uuid_mapping = {}

parser = argparse.ArgumentParser()
parser.add_argument("--supervisely_tar_file", default=r"C:\Users\raimund\Downloads\Illustrative Characters.tar")
parser.add_argument("--supervisely_dataset_name", default="Single Characters")
parser.add_argument("--tmp_folder", default=r"E:\CNN\implicit_functions\characters\tmp")
parser.add_argument("--sub_folders", default="")
args = parser.parse_args()

supervisely_tar_file = args.supervisely_tar_file
input_folder = os.path.dirname(supervisely_tar_file)
supervisely_project_name = os.path.basename(supervisely_tar_file).replace(".tar", "")

supervisely_dataset_name = args.supervisely_dataset_name

output_folder = args.tmp_folder

supervisely_meta_json_file_path = os.path.join(input_folder, supervisely_project_name, "meta.json")
supervisely_images_folder = os.path.join(input_folder, supervisely_project_name, supervisely_dataset_name, "img")
supervisely_annotations_folder = os.path.join(input_folder, supervisely_project_name, supervisely_dataset_name, "ann")

subfolder_names = None if args.sub_folders == "" else args.sub_folders.split("|")

def main():
    tar_file = tarfile.open(supervisely_tar_file)
    tar_file.extractall(input_folder)
    tar_file.close()
    
    images_output_folder = os.path.join(output_folder, "images")
    masks_tmp_output_folder = os.path.join(output_folder, "masks")
    keypoints_tmp_output_folder = os.path.join(output_folder, "keypoints")
    
    for sub_folder in [output_folder, images_output_folder, masks_tmp_output_folder, keypoints_tmp_output_folder]:
        if (os.path.exists(sub_folder)):
            shutil.rmtree(sub_folder)
    
        os.mkdir(sub_folder)
    
    
    supervisely_meta_json = json.load(open(supervisely_meta_json_file_path))
    
    for annotation_class in supervisely_meta_json["classes"]:
        if (annotation_class["shape"] == "graph"):
            nodes = annotation_class["geometry_config"]["nodes"]
            
            for node_uuid, node_data in nodes.items():
                keypoint_index = KEYPOINT_MAPPING[node_data["label"]]
                keypoint_uuid_mapping[node_uuid] = keypoint_index
    
    
    for supervisely_image_name in os.listdir(supervisely_images_folder):
        image_name_without_ext, _ = os.path.splitext(supervisely_image_name)
		
        if subfolder_names != None and not image_name_without_ext in subfolder_names:
            continue
			
        print(supervisely_image_name)
     
        supervisely_image_path = os.path.join(supervisely_images_folder, supervisely_image_name)
        supervisely_annotations_json_file_path = os.path.join(supervisely_annotations_folder, supervisely_image_name + ".json")
        
        keypoints_file_path = os.path.join(keypoints_tmp_output_folder, image_name_without_ext + ".json")
        mask_file_path = os.path.join(masks_tmp_output_folder, image_name_without_ext + ".png")
    
        supervisely_annotations_json = json.load(open(supervisely_annotations_json_file_path))
        supervisely_objects = supervisely_annotations_json["objects"]
        
        if (len(supervisely_objects) == 0):
            continue
        
        width = supervisely_annotations_json["size"]["width"]
        height = supervisely_annotations_json["size"]["height"]
        
        
        mask = Image.new('RGB', (width, height), (255, 255, 255))
        
        keypoints = {}

        # keypoints
        for supervisely_object in supervisely_objects:
            if (supervisely_object["classTitle"] == SKELETON_CLASS_NAME):
                
                for node_uuid, node_data in supervisely_object["nodes"].items():
                    keypoint_index = keypoint_uuid_mapping[node_uuid]
                    point = node_data["loc"]
                    keypoints[keypoint_index] = point

                        
        for supervisely_object in supervisely_objects:
            body_part_name = supervisely_object["classTitle"]
            
            if (body_part_name in body_part_names):
                tmp_mask = Image.new('L', (width, height), 0)
                
                exterior = supervisely_object["points"]["exterior"]
                ImageDraw.Draw(tmp_mask).polygon(list(map(tuple, exterior)), 255)
                
                for interior in supervisely_object["points"]["interior"]:
                    ImageDraw.Draw(tmp_mask).polygon(list(map(tuple, interior)), 0)
                
                body_part_index = BODY_PART_MAPPING[body_part_name]
                color = (0, body_part_index, 0)

                body_part_mask = Image.new('RGB', (width, height), color)
                mask.paste(body_part_mask, mask=tmp_mask)
        
        shutil.copy(supervisely_image_path, images_output_folder)
        
        # remove pixel if lower or right pixel is empty
        mask_arr = np.array(mask)
        new_mask_arr = np.copy(mask_arr)
        
        for y in range(height):
            for x in range(width):
                if (np.logical_and.reduce(mask_arr[y][x] != (255, 255, 255)) and 
                    (x+1 == width or y+1 == height or np.logical_and.reduce(mask_arr[y+1][x] == (255, 255, 255)) 
                     or np.logical_and.reduce(mask_arr[y][x+1] == (255, 255, 255)))):
                    new_mask_arr[y][x] = (255, 255, 255)
        
        mask = Image.fromarray(new_mask_arr)
        
        mask.save(mask_file_path)
        
        json.dump([keypoints], open(keypoints_file_path, "w"))



if __name__ == '__main__':
    main()
'''
crops binary square masks for individual body parts. body parts are scaled and 
translated to the center of the corresponding pose points (except head).
the scale is estimated for the 3D body parts from the xy bounding box 
of a body part and the corresponding 3D pose points (see also estimate_body_part_bbox.py).
for symmetric body parts, the average scale of both is calculated later on.
'''
import numpy as np
import json
from PIL import Image
import os
import joblib
import argparse


BODY_PARTS = {
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

body_part_names = list(BODY_PARTS.keys())


body_part_side_names = ["upper_arm", "lower_arm", "hand", "upper_leg", "lower_leg", "foot"]

POSE_POINTS = {
    'right_ankle': 0, 
    'right_knee': 1,
    'right_hip': 2,
    'left_hip': 3,
    'left_knee': 4,
    'left_ankle': 5,
    'pelvis': 6, # hip
    'root': 7, # thorax
    'neck': 8,
    'head': 9,
    'right_wrist': 10,   
    'right_elbow': 11,
    'right_shoulder': 12,
    'left_shoulder': 13,
    'left_elbow': 14,
    'left_wrist': 15,
    'right_foot': 16,
    'left_foot': 17,
    'right_hand': 18,
    'left_hand': 19,
    'eyes': 20
}

BONES = {
    "torso": ["right_shoulder", "left_shoulder", "left_hip", "right_hip"],
    "head": ["head", "eyes"],
    "right_upper_arm": ["right_shoulder", "right_elbow"],
    "right_lower_arm": ["right_elbow", "right_wrist"],
    "right_hand": ["right_wrist", "right_hand"],
    "right_upper_leg": ["right_hip", "right_knee"],
    "right_lower_leg": ["right_knee", "right_ankle"],
    "right_foot": ["right_ankle", "right_foot"],
    "left_upper_leg": ["left_hip", "left_knee"],
    "left_lower_leg": ["left_knee", "left_ankle"],
    "left_foot": ["left_ankle", "left_foot"],
    "left_upper_arm": ["left_shoulder", "left_elbow"],
    "left_lower_arm": ["left_elbow", "left_wrist"],
    "left_hand": ["left_wrist", "left_hand"]  
}


size = 64
size_half = size / 2
offset_factor = 1

def main(input_folder, body_part_name, view, scale_predictor, needs_mirroring=False):
    body_image = Image.open(os.path.join(input_folder, "body_parts_%s_cropped_mask.png" % view))
    # all channels are the same
    body_image = body_image.getchannel(0)
    
    with open(os.path.join(input_folder, "skeleton_%s.json" % view)) as file:
        skeleton = json.load(file)
        
        for key, coords in skeleton.items():
            skeleton[key] = coords
    
    # mid point coordinates are already rotated (in contrast to mesh bounds)
    mid_point = [0, 0, 0]
    joints = list(map(lambda joint_name: skeleton[str(POSE_POINTS[joint_name])], BONES[body_part_name]))
    
    for joint in joints:
        mid_point[0] += joint[0]
        mid_point[1] += joint[1]
        mid_point[2] += joint[2]
            
    num_bones = len(BONES[body_part_name])
    mid_point[0] /= num_bones
    mid_point[1] /= num_bones
    mid_point[2] /= num_bones
    
    # special case for head
    if (body_part_name == "head"):
        mid_point[0] = joints[0][0]
        mid_point[1] = joints[0][1]
        mid_point[2] = joints[0][2]
    
    body_texture = Image.open(os.path.join(input_folder, "body_parts_%s_cropped_texture.png" % view))
    body_texture_np = np.asarray(body_texture)
    
    pose_index = BODY_PARTS[body_part_name]
    body_image_np = np.asarray(body_image)
    body_part_mask_np = (body_image_np == pose_index).astype(np.uint8)      

    body_part_mask = Image.fromarray(body_part_mask_np, "L")
    body_part_mask_bbox = body_part_mask.getbbox()
    
    joints_normalized = list(map(lambda point: (np.array(point) - np.array(mid_point)) / 600, joints)) 
    
    if needs_mirroring:
        joints_normalized = list(map(lambda point: [-point[0], point[1], point[2]], joints_normalized)) 
    
    joints_normalized = np.array(joints_normalized).flatten()
            
    d_minimum = scale_predictor.predict([joints_normalized])[0] / 2
    
    if (body_part_mask_bbox == None):
        white = np.ones((size, size)) * 255
        white_rgb = np.stack([white, white, white], axis=2)
        body_part_texture = Image.fromarray(np.uint8(white_rgb), "RGB")
        
        zeros = np.zeros((size, size))
        body_part_texture_mask = Image.fromarray(zeros, "L")
        
        body_part_joints = {"scale": None, "joints": joints, "mid_point": mid_point}
        return body_part_joints
        
    else :
        body_part_mask_np = np.expand_dims(body_part_mask, axis=2)
        body_part_mask_rgba_np = np.repeat(body_part_mask_np, 4, axis=2)
        
        # mask out other body parts and clip according to the bounding box
        body_part_texture_masked_np = body_part_mask_rgba_np * body_texture_np    
        body_part_texture_masked = Image.fromarray(np.uint8(body_part_texture_masked_np))
        body_part_texture_masked = body_part_texture_masked.crop(body_part_mask_bbox)
    
        [min_x, min_y, max_x, max_y] = body_part_mask_bbox
        
        # find out maximal distance between mid point and bounding box coordinate
        d_mid_min_x = abs(mid_point[0] - min_x)
        d_mid_max_x = abs(max_x - mid_point[0])
        d_mid_min_y = abs(mid_point[1] - min_y)
        d_mid_max_y = abs(max_y - mid_point[1])
                
        max_d_mid = max(d_mid_min_x, d_mid_max_x, d_mid_min_y, d_mid_max_y, d_minimum)
        max_d_mid_with_offset = max_d_mid * offset_factor
        
        min_x_img = round(mid_point[0] - max_d_mid_with_offset)
        max_x_img = round(mid_point[0] + max_d_mid_with_offset)
        min_y_img = round(mid_point[1] - max_d_mid_with_offset)
        max_y_img = round(mid_point[1] + max_d_mid_with_offset)
        
        dx_img = max_x_img - min_x_img
        dy_img = max_y_img - min_y_img
        
        offset_x = body_part_mask_bbox[0] - min_x_img
        offset_y = body_part_mask_bbox[1] - min_y_img
        
        body_part_texture = Image.new("RGBA", (dx_img, dy_img), (255, 255, 255, 0))
        body_part_texture.paste(body_part_texture_masked, (offset_x, offset_y))
        
        body_part_texture_mask = body_part_texture.getchannel(3)
        body_part_texture_mask = body_part_texture_mask.resize((size, size), resample=Image.NEAREST)
        
        body_part_texture = body_part_texture.resize((size, size))
        
    body_part_texture.save(os.path.join(input_folder, "%s_%s_texture.png" % (body_part_name, view)))
    body_part_texture_mask.save(os.path.join(input_folder, "%s_%s_mask.png" % (body_part_name, view)))

    body_part_joints = {}
    body_part_joints["scale"] = max_d_mid_with_offset * 2
    
    body_part_joints[body_part_name] = []

    for joint in joints:
        joint_x = size_half + (joint[0] - mid_point[0]) / max_d_mid_with_offset * size_half
        joint_y = size_half + (joint[1] - mid_point[1]) / max_d_mid_with_offset * size_half
        joint_z = size_half + (joint[2] - mid_point[2]) / max_d_mid_with_offset * size_half
        
        body_part_joints[body_part_name].append([joint_x, joint_y, joint_z])
    
    return body_part_joints


def symmetrize(body_parts):
    for body_part_side_name in body_part_side_names:
        left_scale = body_parts["left_" + body_part_side_name]["scale"]
        right_scale = body_parts["right_" + body_part_side_name]["scale"]
        
        if (left_scale == None):
            left_scale = right_scale
        elif (right_scale == None):
            right_scale = left_scale
        else :
            average_scale = (left_scale + right_scale) / 2
            left_scale = average_scale
            right_scale = average_scale
            
        body_parts["left_" + body_part_side_name]["scale"] = left_scale
        body_parts["right_" + body_part_side_name]["scale"] = right_scale
 

if __name__ == '__main__':
    current_folder = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_folder", default=r"E:\CNN\implicit_functions\characters\output")
    parser.add_argument("--model_folder", default=os.path.join(current_folder, "scale_predictors"))
    parser.add_argument("--sub_folders", default="")
    args = parser.parse_args()
    
    root_folder = args.out_folder
    model_folder = args.model_folder
    subfolder_names = os.listdir(root_folder) if args.sub_folders == "" else args.sub_folders.split("|")
    
    for subfolder_name in subfolder_names:
        input_folder = os.path.join(root_folder, subfolder_name)
        print("Processing %s..." % input_folder)
        
        body_parts = {}
        
        for body_part_name in body_part_names:
            scale_predictor_path = os.path.join(model_folder, body_part_name.replace("right", "left") + ".pkl")
            scale_predictor = joblib.load(scale_predictor_path)
            
            needs_mirroring = body_part_name.find("right") != -1
            body_part_joints = main(input_folder, body_part_name, "front", scale_predictor, needs_mirroring=needs_mirroring)
                        
            body_parts[body_part_name] = body_part_joints
            
        symmetrize(body_parts)
        
        for body_part_name, body_part_joints in body_parts.items():
            # calculate joint positions for hidden body parts
            if "joints" in body_part_joints:
                joints = body_part_joints.pop("joints")
                mid_point = body_part_joints.pop("mid_point")
                scale_half = body_part_joints["scale"] / 2
            
                body_part_joints[body_part_name] = []
            
                for joint in joints:
                    joint_x = size_half + (joint[0] - mid_point[0]) / scale_half * size_half
                    joint_y = size_half + (joint[1] - mid_point[1]) / scale_half * size_half
                    joint_z = size_half + (joint[2] - mid_point[2]) / scale_half * size_half
                    
                    body_part_joints[body_part_name].append([joint_x, joint_y, joint_z])
            
            with open(os.path.join(input_folder, "%s.json" % body_part_name), "w") as file:
                json.dump(body_part_joints, file)
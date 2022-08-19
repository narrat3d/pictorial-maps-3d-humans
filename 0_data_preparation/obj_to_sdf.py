'''
converts split 3D body parts in *.obj to SDF cubic grid.
the body parts are scaled and translated to the center of the corresponding pose points 
(except head). the original scale is preserved in an additional file.
additionally, points near the surface are sampled (but those are not used currently)
'''
from mesh_to_sdf import mesh_to_sdf, sample_sdf_near_surface

import trimesh
import numpy as np
import json
from PIL import Image
import os
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', dest='input_folder')
parser.add_argument('--output_file', dest='output_file')
parser.add_argument('--output_file2', dest='output_file2')
parser.add_argument('--body_part_name', dest='body_part_name')
parser.add_argument('--view', dest='view')

args = parser.parse_args()
input_folder = args.input_folder
body_part_name = args.body_part_name
output_np_file_path = args.output_file
output_np_file_path_surface = args.output_file2
view = args.view


if (input_folder == None):
    input_folder = r"E:\CNN\implicit_functions\smpl-x\debug"
    body_part_name = "left_foot"
    view = "left"
    output_np_file_path = os.path.join(input_folder, "%s.npy" % body_part_name)

if (os.path.exists(output_np_file_path)):
    sys.exit(0)


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

POSE_POINTS = {
    "root": 0,
    "pelvis": 1,
    "left_hip": 2,
    "left_knee": 3,
    "left_ankle": 4,
    "left_foot": 5,
    "right_hip": 6,
    "right_knee": 7,
    "right_ankle": 8,
    "right_foot": 9,
    "spine1": 10,
    "spine2": 11,
    "spine3": 12,
    "neck": 13,
    "head": 14,
    "jaw": 15,
    "left_eye_smplhf": 16,
    "right_eye_smplhf": 17,
    "left_collar": 18,
    "left_shoulder": 19,
    "left_elbow": 20,
    "left_wrist": 21,
    "left_index1": 22,
    "left_index2": 23,
    "left_index3": 24,
    "left_middle1": 25,
    "left_middle2": 26,
    "left_middle3": 27,
    "left_pinky1": 28,
    "left_pinky2": 29,
    "left_pinky3": 30,
    "left_ring1": 31,
    "left_ring2": 32,
    "left_ring3": 33,
    "left_thumb1": 34,
    "left_thumb2": 35,
    "left_thumb3": 36,
    "right_collar": 37,
    "right_shoulder": 38,
    "right_elbow": 39,
    "right_wrist": 40,
    "right_index1": 41,
    "right_index2": 42,
    "right_index3": 43,
    "right_middle1": 44,
    "right_middle2": 45,
    "right_middle3": 46,
    "right_pinky1": 47,
    "right_pinky2": 48,
    "right_pinky3": 49,
    "right_ring1": 50,
    "right_ring2": 51,
    "right_ring3": 52,
    "right_thumb1": 53,
    "right_thumb2": 54,
    "right_thumb3": 55
}

BONES = {
    "torso": ["right_shoulder", "left_shoulder", "left_hip", "right_hip"],
    "head": ["head", "right_eye_smplhf", "left_eye_smplhf"],
    "right_upper_arm": ["right_shoulder", "right_elbow"],
    "right_lower_arm": ["right_elbow", "right_wrist"],
    "right_hand": ["right_wrist", "right_middle1"],
    "right_upper_leg": ["right_hip", "right_knee"],
    "right_lower_leg": ["right_knee", "right_ankle"],
    "right_foot": ["right_ankle", "right_foot"],
    "left_upper_leg": ["left_hip", "left_knee"],
    "left_lower_leg": ["left_knee", "left_ankle"],
    "left_foot": ["left_ankle", "left_foot"],
    "left_upper_arm": ["left_shoulder", "left_elbow"],
    "left_lower_arm": ["left_elbow", "left_wrist"],
    "left_hand": ["left_wrist", "left_middle1"]
}

CAMERA_ANGLES = {
    "front": 0, 
    "left": math.pi / 2, 
    "back": math.pi,
    "right": 3 * math.pi / 2
}

def rotate_2d(x, y, angle):
    rot_x = math.cos(angle) * x - math.sin(angle) * y
    rot_y = math.sin(angle) * x + math.cos(angle) * y
    
    return [rot_x, rot_y]


def main(input_folder, body_part_name, view, output_np_file_path, output_np_file_path_surface):
    body_parts_mask = Image.open(os.path.join(input_folder, "body_parts_%s_mask.png" % view))
    body_parts_mask = body_parts_mask.getchannel(0)
    
    image_size = body_parts_mask.width
    image_size_half = image_size / 2
    
    with open(os.path.join(input_folder, "skeleton_%s.json" % view)) as file:
        skeleton = json.load(file)
        
    with open(os.path.join(input_folder, "body_dimensions.json")) as file:
        body_dimensions = json.load(file)
    
    mesh = trimesh.load(os.path.join(input_folder, "%s.obj" % body_part_name))
    [[min_x, min_y, min_z], [max_x, max_y, max_z]] = mesh.bounds
    
    angle = CAMERA_ANGLES[view]
    
    [min_x, min_z] = rotate_2d(min_x, min_z, angle)
    [max_x, max_z] = rotate_2d(max_x, max_z, angle)
    
    size = 64
    size_half = size / 2
    offset_factor = 1
    
    body_scaling_factor = max(body_dimensions["x"], body_dimensions["y"]) / 2
    
    def coord_transform(x):
        return (x - image_size_half) / image_size_half * body_scaling_factor
    
    def img_transform(x):
        return round((x / body_scaling_factor * image_size_half) + image_size_half)
    
    def body_part_img_transform(x, mid_point_x, dx):
        return ((x - mid_point_x) / dx * size_half) + size_half
    
    # mid point coordinates are already rotated (in contrast to mesh bounds)
    mid_point = [0, 0, 0]
    joints = list(map(lambda joint_name: skeleton[str(POSE_POINTS[joint_name])], BONES[body_part_name]))
    joints_norm = []
    
    for joint in joints:
        joint_norm = list(map(coord_transform, joint))
        # mirror y-axis
        joint_norm[1] *= -1
        
        mid_point[0] += joint_norm[0]
        mid_point[1] += joint_norm[1]
        mid_point[2] += joint_norm[2]
        
        joints_norm.append(joint_norm)
    
    num_bones = len(BONES[body_part_name])
    mid_point[0] /= num_bones
    mid_point[1] /= num_bones
    mid_point[2] /= num_bones
    
    # special case for head
    if (body_part_name == "head"):
        mid_point[0] = joints_norm[0][0]
        mid_point[1] = joints_norm[0][1]
        mid_point[2] = joints_norm[0][2]
        
    # find out maximal distance between mid point and bounding box coordinate
    d_mid_min_x = abs(mid_point[0] - min_x)
    d_mid_max_x = abs(max_x - mid_point[0])
    d_mid_min_y = abs(mid_point[1] - min_y)
    d_mid_max_y = abs(max_y - mid_point[1])  
    d_mid_min_z = abs(mid_point[2] - min_z)
    d_mid_max_z = abs(max_z - mid_point[2])      
    
    max_d_mid = max(d_mid_min_x, d_mid_max_x, d_mid_min_y, 
                    d_mid_max_y, d_mid_min_z, d_mid_max_z)
    
    max_d_mid_with_offset = max_d_mid * offset_factor
    
    new_min_x = mid_point[0] - max_d_mid_with_offset
    new_max_x = mid_point[0] + max_d_mid_with_offset
    new_min_y = mid_point[1] - max_d_mid_with_offset
    new_max_y = mid_point[1] + max_d_mid_with_offset
    
    
    body_texture = Image.open(os.path.join(input_folder, "body_parts_%s_texture.png" % view))
    body_texture_np = np.asarray(body_texture)
    
    pose_index = BODY_PARTS[body_part_name]
    body_part_mask_np = (np.asarray(body_parts_mask) == pose_index).astype(np.uint8)      

    body_part_mask = Image.fromarray(body_part_mask_np, "L")
    body_part_mask_bbox = body_part_mask.getbbox()
    
    if (body_part_mask_bbox == None):
        return
    
    body_part_mask_np = np.expand_dims(body_part_mask, axis=2)
    body_part_mask_rgba_np = np.repeat(body_part_mask_np, 4, axis=2)
    
    # mask out other body parts and clip according to the bounding box
    body_part_texture_masked_np = body_part_mask_rgba_np * body_texture_np    
    body_part_texture_masked = Image.fromarray(np.uint8(body_part_texture_masked_np))
    body_part_texture_masked = body_part_texture_masked.crop(body_part_mask_bbox)

    # clip image according to the 3D bounding box centred at the midpoint
    min_x_img = img_transform(new_min_x)
    max_x_img = img_transform(new_max_x)
    max_y_img = image_size - img_transform(new_min_y)
    min_y_img = image_size - img_transform(new_max_y)
    
    dx_img = max_x_img - min_x_img
    dy_img = max_y_img - min_y_img
    
    offset_x = body_part_mask_bbox[0] - min_x_img
    offset_y = body_part_mask_bbox[1] - min_y_img
    
    body_part_texture = Image.new("RGBA", (dx_img, dy_img), (255, 255, 255, 0))
    body_part_texture.paste(body_part_texture_masked, (offset_x, offset_y))
    
    body_part_texture_mask = body_part_texture.getchannel(3)
    body_part_texture_mask = body_part_texture_mask.resize((size, size), resample=Image.NEAREST)
    body_part_texture_mask.save(os.path.join(input_folder, "%s_%s_mask.png" % (body_part_name, view)))
    
    body_part_texture = body_part_texture.resize((size, size))
    body_part_texture.save(os.path.join(input_folder, "%s_%s_texture.png" % (body_part_name, view)))
    
    if (not os.path.exists(output_np_file_path_surface)):
        body_part_joints = {}
        body_part_joints[body_part_name] = []
        
        for joint_norm in joints_norm:
            joint_img_x = body_part_img_transform(joint_norm[0], mid_point[0], max_d_mid_with_offset)
            joint_img_y = size - body_part_img_transform(joint_norm[1], mid_point[1], max_d_mid_with_offset)
            joint_img_z = body_part_img_transform(joint_norm[2], mid_point[2], max_d_mid_with_offset)
            
            # rotate to front view
            [tmp_z, tmp_x] = rotate_2d(joint_img_z - size_half, joint_img_x - size_half, angle) 
            [joint_img_z, joint_img_x] = [tmp_z + size_half, tmp_x + size_half]
            
            body_part_joints[body_part_name].append([joint_img_x, joint_img_y, joint_img_z])
        
        body_part_joints["scale"] = max_d_mid_with_offset / body_scaling_factor * image_size
            
        with open(os.path.join(input_folder, "%s.json" % body_part_name), "w") as file:
            json.dump(body_part_joints, file)
        
        # in case the body part is not viewed from the front, then the mid point coordinates 
        # (derived from the skeleton) need to be rotated to the front for the mesh
        [mid_point[2], mid_point[0]] = rotate_2d(mid_point[2], mid_point[0], angle)
        
        # normalize vertices to -1 ... 1 for better SDF
        for vertex in mesh.vertices:
            vertex[0] -= mid_point[0]
            vertex[1] -= mid_point[1]
            vertex[2] -= mid_point[2]
            
            vertex[0] = vertex[0] / max_d_mid_with_offset
            vertex[1] = vertex[1] / max_d_mid_with_offset
            vertex[2] = vertex[2] / max_d_mid_with_offset
        
        xs = np.linspace(-1, 1, size)
        ys = np.linspace(-1, 1, size)
        zs = np.linspace(-1, 1, size)
    
        num_points = size*size*size
    
        grid = np.meshgrid(xs, ys, zs, indexing='ij')
        grid = np.moveaxis(grid, 0, -1)
        grid = np.reshape(grid, [num_points, 3])
        
        # regular grid
        sdf = mesh_to_sdf(mesh, grid, sign_method="depth")
        
        cube = np.reshape(sdf, [size, size, size])
        scaled_cube = cube * size_half
        np.save(output_np_file_path, scaled_cube)
        
        # source: https://github.com/marian42/mesh_to_sdf/issues/23
        def compute_unit_sphere_transform(mesh):
            translation = -mesh.bounding_box.centroid
            scale = 1 / np.max(np.linalg.norm(mesh.vertices + translation, axis=1))
            
            return translation, scale
        
        points, sdf = sample_sdf_near_surface(mesh, number_of_points=num_points)
        
        translation, scale = compute_unit_sphere_transform(mesh)
        points = (points / scale) - translation
        sdf /= scale
        
        points = points * size_half + size_half
        sdf = np.reshape(sdf * size_half, (num_points, 1))
        surface_points = np.concatenate([points, sdf], axis=1)
        np.save(output_np_file_path_surface, surface_points.astype(np.float32))
    
        """
		# uncomment to visualize the results
        indices = np.indices((size, size, size))
        indices = np.moveaxis(indices, 0, -1)
        indices = np.reshape(indices, [size*size*size, 3])
        
        xs = indices[:, 0]
        ys = indices[:, 1]
        zs = indices[:, 2]
        
        from mayavi import mlab 
        
        mlab.points3d(xs.flatten()[sdf < 0], ys.flatten()[sdf < 0], zs.flatten()[sdf < 0], sdf[sdf < 0], color=(1., 1., 1.), scale_mode="none")
        mlab.show()  
        
        
        mlab.points3d(xs.flatten(), ys.flatten(), zs.flatten(), sdf, colormap='Spectral', scale_factor=.05)
        mlab.show()
        
           
        points_2d = np.min(cube, axis=2)
        
        points_2d[points_2d > 0] = 0
        points_2d[points_2d < 0] = 255
        
        im = Image.fromarray(np.uint8(np.swapaxes(np.flip(points_2d, 1), 0, 1)), "L")
        im.show()
        """
        
main(input_folder, body_part_name, view, output_np_file_path, output_np_file_path_surface)
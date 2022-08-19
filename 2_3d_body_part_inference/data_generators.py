import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa 
import random
import random as random2
import os
import json
from config import CAMERA_ANGLES, IMAGE_SIZE, SCALING_FACTOR, IMAGE_SIZE_HALF,\
    NUM_SAMPLES_HALF, EPOCHS, VIEWS, BATCH_SIZE, DEBUG
import math
from PIL import Image


def train_data_generator(input_folder, train_subfolder_names, file_checker, body_part_name, use_pose_points):
    batch = []
    
    for _ in range(EPOCHS):
        random.shuffle(train_subfolder_names)
        
        for step, subfolder_name in enumerate(train_subfolder_names): 
            view = np.random.choice(VIEWS)
            
            points_3d_sdf = load_output_data(input_folder, subfolder_name, body_part_name, view)
            mask_points_2d, pose_points = load_input_data(input_folder, subfolder_name, file_checker, body_part_name, view, points_3d_sdf)  
            
            coords = create_coords()
            points_3d = np.reshape(points_3d_sdf, [IMAGE_SIZE*IMAGE_SIZE*IMAGE_SIZE, 1])
            coords_points_3d = np.concatenate((coords[0], points_3d), axis=1)
            
            inside_points = coords_points_3d[:, 3] <= 0.0
            outside_points = coords_points_3d[:, 3] > 0.0
            
            coords_points_3d_inside = coords_points_3d[inside_points]
            coords_points_3d_outside = coords_points_3d[outside_points]
            
            num_points_inside = coords_points_3d_inside.shape[0]
            num_points_outside = coords_points_3d_outside.shape[0]
            
            selected_inside_indices = np.random.choice(np.arange(num_points_inside), NUM_SAMPLES_HALF)
            selected_outside_indices = np.random.choice(np.arange(num_points_outside), NUM_SAMPLES_HALF)
            
            coords_points_3d_inside = np.take(coords_points_3d_inside, selected_inside_indices, 0)
            coords_points_3d_outside = np.take(coords_points_3d_outside, selected_outside_indices, 0)
            
            sampled_coords_points_3d = np.concatenate((coords_points_3d_inside, coords_points_3d_outside), axis=0) 
            
            sampled_coords = sampled_coords_points_3d[:, 0:3]
            sampled_points_3d = sampled_coords_points_3d[:, 3]

            coords = np.expand_dims(sampled_coords, axis=0)
            points_3d = np.expand_dims(sampled_points_3d, axis=0)
            points_3d = np.expand_dims(points_3d, axis=-1)

            inputs = [mask_points_2d, coords]
            outputs = [points_3d]

            if (use_pose_points):
                inputs.append(pose_points)

            batch.append([inputs, outputs])
        
            if ((step + 1) % BATCH_SIZE == 0 or step == len(train_subfolder_names) - 1):
                [input_batch, output_batch] = map(create_batch, prepare_batch(batch))
                yield((input_batch, output_batch)) 
                batch = []


def eval_data_generator(input_folder, test_subfolder_names, file_checker, body_part_name, use_pose_points, eval_all_coords=False):
    for _ in range(EPOCHS):
        for subfolder_name in test_subfolder_names:     
            for view in VIEWS:
                points_3d_sdf = load_output_data(input_folder, subfolder_name, body_part_name, view)
                mask_points_2d, pose_points = load_input_data(input_folder, subfolder_name, file_checker, body_part_name, view, points_3d_sdf)
                
                coords = create_coords()
                if DEBUG or not eval_all_coords:
                    selected_coords = coords[:, ::16]
                else :
                    selected_coords = coords[:, :]
                    
                points_3d = tf.gather_nd(points_3d_sdf[0], selected_coords)
                
                inputs = [mask_points_2d, selected_coords.astype(np.float32)]
                
                if use_pose_points:
                    inputs.append(pose_points)
                
                outputs = [points_3d]
                
                # batch size is always 1 for eval
                batch = [[inputs, outputs]]
                
                [input_batch, output_batch] = map(create_batch, prepare_batch(batch))

                yield((input_batch, output_batch)) 


def prepare_batch(batch):
    input_batch = []
    output_batch = []
    
    for inputs, outputs in batch:
        input_batch.append(inputs)
        output_batch.append(outputs)
    
    return input_batch, output_batch


def create_batch(data_batch):
    num_data = len(data_batch[0])
    joined_data = [[]] * num_data
    
    for data in data_batch:
        for i, data_sample in enumerate(data):
            joined_data[i] = joined_data[i] + [data_sample]
    
    for i in range(num_data):
        joined_data[i] = np.concatenate(joined_data[i], axis=0)
        
    return joined_data


def create_coords():
    grid = np.indices((IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE))
    coords = np.moveaxis(grid, 0, -1)
    coords = np.reshape(coords, (-1, 3))
    coords = np.expand_dims(coords, axis=0)
    
    return coords


def rotate_2d_xz(p, angle):
    rot_x = math.cos(angle) * p[0] - math.sin(angle) * p[2]
    rot_y = math.sin(angle) * p[0] + math.cos(angle) * p[2]
    
    return [rot_x, p[1], rot_y]



def load_input_data(input_folder, subfolder_name, file_checker, body_part_name, view, sdf_field):
    mask_file_name = "%s_%s_mask.png" % (body_part_name, view)
    pose_points_file_name = "%s.json" % body_part_name
    
    # mask points: 0 = inside, 1 = outside
    if (file_checker.exists_file(subfolder_name, mask_file_name)):
        mask_path = os.path.join(input_folder, subfolder_name, mask_file_name)
        body_part_mask = Image.open(mask_path)
        mask_points = 1 - np.asarray(body_part_mask) / 255
        mask_points = np.flip(np.swapaxes(mask_points, 0, 1), 1)

    else :
        # decide whether to generate an empty or complete 2D mask
        # probability is around 0.5 for either case
        random2.seed(subfolder_name)
        generate_empty_mask = random2.choice([True, False])
        
        if generate_empty_mask or sdf_field is None: # blank image
            mask_points = np.ones((IMAGE_SIZE, IMAGE_SIZE))
        else : # create complete mask based on SDF 3D
            body_part_mask = np.min(sdf_field[0], axis=2)
            mask_points = np.heaviside(body_part_mask, 0)
    
    points_2d = np.reshape(mask_points, (1, IMAGE_SIZE, IMAGE_SIZE, 1)) 
    # zero-center points
    points_2d -= 0.5
    
    pose_points_path = os.path.join(input_folder, subfolder_name, pose_points_file_name)
    pose_points = json.load(open(pose_points_path))
    pose_points = np.array(pose_points[body_part_name])
    
    # in training data, head has three pose points (incl. right/left eye)
    # in test data, head has two pose points (incl. eye center)
    if body_part_name == "head" and pose_points.shape[0] == 3:
        point_between_eyes = np.average([pose_points[1], pose_points[2]], axis=0)
        pose_points = np.array([pose_points[0], point_between_eyes])
    
    pose_points = (pose_points - IMAGE_SIZE_HALF) / IMAGE_SIZE_HALF
    
    rotated_pose_points = []
    camera_angle = CAMERA_ANGLES[view]
    
    for i in range(pose_points.shape[0]):
        pose_point = pose_points[i]
        rotated_pose_point = rotate_2d_xz(pose_point, camera_angle)
        rotated_pose_points.append(rotated_pose_point)
    
    pose_points = np.expand_dims(rotated_pose_points, axis=0)
    
    return points_2d, pose_points


def load_output_data(input_folder, subfolder_name, body_part_name, view):
    sdf_object_path = os.path.join(input_folder, subfolder_name, "%s.npy" % body_part_name)
    sdf_object = np.load(sdf_object_path)
    points_3d = np.reshape(sdf_object, (IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE))
    
    points_3d_rotated = rotate_distance_field[view](points_3d)
    points_3d = np.expand_dims(points_3d_rotated, axis=0) / SCALING_FACTOR

    return points_3d


def rotate_distance_field_front(points_3d):
    return points_3d
    
def rotate_distance_field_left(points_3d):
    return np.flip(np.swapaxes(points_3d, 0, 2), 0)

def rotate_distance_field_back(points_3d):
    return np.flip(points_3d, 0)

def rotate_distance_field_right(points_3d):
    return np.flip(np.swapaxes(points_3d, 0, 2), 2)


rotate_distance_field = {
    "front": rotate_distance_field_front,
    "left": rotate_distance_field_left,
    "back": rotate_distance_field_back,
    "right": rotate_distance_field_right
}
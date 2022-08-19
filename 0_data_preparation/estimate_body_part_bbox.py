'''
helps to predict the size of a body part from the corresponding pose points.
this is needed since a 2D body part mask does not contain any depth information.
one could have predicted the depth only, but as 2D body part masks are incomplete
sometimes (e.g. due to occlusions), it makes sense to predict the uniform scale.

the trained models are used in supervisely_extract_body_parts.py
'''
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.neural_network._multilayer_perceptron import MLPRegressor
from sklearn.linear_model import LinearRegression
import math

POSE_POINTS_SMPLX = {
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
    "head": ["head", "left_eye_smplhf", "right_eye_smplhf"],
    "left_upper_leg": ["left_hip", "left_knee"],
    "left_lower_leg": ["left_knee", "left_ankle"],
    "left_foot": ["left_ankle", "left_foot"],
    "left_upper_arm": ["left_shoulder", "left_elbow"],
    "left_lower_arm": ["left_elbow", "left_wrist"],
    "left_hand": ["left_wrist", "left_middle1"]  
}

root_folder = r"C:\Users\raimund\Downloads\tmp2"

all_subfolders = os.listdir(root_folder)

split_size = math.floor(len(all_subfolders) * 0.9)
train_subfolders = all_subfolders[:split_size]
test_subfolders = all_subfolders[split_size:]

def get_data(subfolders, body_part_name):
    X = []
    Y = []
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(root_folder, subfolder)
        
        if os.path.isfile(subfolder_path):
            continue
        
        with open(os.path.join(subfolder_path, "skeleton_front.json")) as skeleton_file:
            skeleton = json.load(skeleton_file)
    
        body_part_path = os.path.join(subfolder_path, body_part_name + ".json")
        
        # left_hand is hidden sometimes
        if not os.path.exists(body_part_path):
            continue
    
        with open(body_part_path) as body_part_file:
            body_part_metadata = json.load(body_part_file)
            
            scale = body_part_metadata["scale"]
        
        bones = BONES[body_part_name]
        
        pose_points = []
        
        for pose_point_name in bones:
            pose_index = POSE_POINTS_SMPLX[pose_point_name]
            pose_point = skeleton[str(pose_index)]
            pose_points.append(np.array(pose_point))
        
        mid_point = np.average(pose_points, axis=0)
        
		# special case
        if body_part_name == "head":
            mid_point = np.array(pose_points[0])
            pose_points = [mid_point, np.average([pose_points[1], pose_points[2]], axis=0)]
        
        pose_points_normalized = list(map(lambda point: (point - mid_point) / 600, pose_points)) 
        
        pose_points_normalized = np.array(pose_points_normalized).flatten()
        
        X.append(pose_points_normalized)
        Y.append(scale)
        
    return X, Y

for body_part_name in BONES.keys():
    print(body_part_name)
    X_train, Y_train = get_data(train_subfolders, body_part_name)
    
    model = MLPRegressor(
      hidden_layer_sizes=(20, 40, 20),
      verbose=True,
      max_iter=2000,
      n_iter_no_change=1000
    )
    
    # model = LinearRegression()
    regr = model.fit(X_train, Y_train)
    print(model.loss_)
    
    X_test, Y_test = get_data(test_subfolders, body_part_name)
    Y_pred = regr.predict(X_test)
    
    plt.scatter(Y_test, Y_pred)
    plt.show()
    
    joblib.dump(model, "scale_predictors/" + body_part_name + ".pkl")
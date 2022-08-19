'''
iterates through the training data for body_part_splitter.py 
'''
import subprocess
import os
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--gender', dest='gender')
parser.add_argument('--size', dest='size')

args = parser.parse_args()
gender = args.gender
size = args.size


genders = {
    "female": {
        "index": 0,
        "texture_paths": []
    },
    "male": {
        "index": 1,
        "texture_paths": []
    }
}

proportions = {
    "small": {
        "weight": 60,
        "height": 1.40,
        "index": 1
    }, 
    "medium": {
        "weight": 75,
        "height": 1.80,
        "index": 0
    }, 
    "large": {
        "weight": 90,
        "height": 2.20,
        "index": 2
    }    
}


def get_files_in_folder(folder_path):
    file_names = os.listdir(folder_path)
    
    file_paths = list(map(lambda file_name: os.path.join(folder_path, file_name), file_names))
    
    return file_paths

def get_file_by_name_in_subfolders(folder_path, file_name):
    subfolder_names = os.listdir(folder_path)
    
    # filter files like Thumbs.db
    subfolder_names = filter(lambda subfolder_name: os.path.isdir(os.path.join(folder_path, subfolder_name)), subfolder_names)
        
    file_paths = list(map(lambda subfolder_name: os.path.join(folder_path, subfolder_name, file_name), 
                          subfolder_names))
    
    return file_paths


def get_files_by_extension_in_subfolder(folder_path, file_extension):
    file_paths = []
    
    subfolder_names = os.listdir(folder_path)
    
    for subfolder_name in subfolder_names:   
        subfolder_path = os.path.join(folder_path, subfolder_name)
        file_names = os.listdir(subfolder_path)
        
        for file_name in file_names:
            file_path = os.path.join(subfolder_path, file_name)
        
            [_, ext] = os.path.splitext(file_path)
            
            if (ext != file_extension):
                continue
            
            file_paths.append(file_path)
    
    return file_paths


data_folder = r"N:\Pictorial-Maps-DISN\data"
blender_python_folder = r"C:\Program Files\Blender Foundation\Blender 2.93\2.93\python"
root_output_folder = r"N:\Pictorial-Maps-DISN\data\narrat3d"

pose_folder = os.path.join(data_folder, r"SMPL\smplx_gt")
male_texture_folder1 = os.path.join(data_folder, r"SMPL\SURREAL\textures\male")
female_texture_folder1 = os.path.join(data_folder, r"SMPL\SURREAL\textures\female")
male_texture_folder2 = os.path.join(data_folder, r"SMPL\Multi-Garment_dataset")

pose_file_paths = get_files_by_extension_in_subfolder(pose_folder, ".pkl")

male_texture_pack1 = get_files_in_folder(male_texture_folder1)
female_texture_pack1 = get_files_in_folder(female_texture_folder1)
male_texture_pack2 = get_file_by_name_in_subfolders(male_texture_folder2, "registered_tex.jpg")

genders["male"]["texture_paths"].extend(male_texture_pack1)
genders["female"]["texture_paths"].extend(female_texture_pack1)
genders["male"]["texture_paths"].extend(male_texture_pack2)

blender_python_paths = map(lambda subfolder: os.path.join(blender_python_folder, subfolder), 
                           ["bin", "DLLs", "lib", r"lib\site-packages"])
    
os.environ["PYTHONPATH"] = ";".join(blender_python_paths)


height = proportions[size]["height"]
weight = proportions[size]["weight"]

poses_per_configuration = math.floor(len(pose_file_paths) / 6)
pose_start_index = (genders[gender]["index"] * 3 + proportions[size]["index"]) * poses_per_configuration

texture_paths = genders[gender]["texture_paths"]
num_textures = len(texture_paths)

for pose_index in range(pose_start_index, pose_start_index + poses_per_configuration):
    pose_file_path = pose_file_paths[pose_index]
    pose_file_name = os.path.basename(pose_file_path)
    [pose_name, _] = os.path.splitext(pose_file_name)

    output_folder = os.path.join(root_output_folder, "%s_%s_%s" % (pose_name, gender, size))
    lock_file_path = os.path.join(output_folder, "lock.lock")
    
    texture_path = texture_paths[pose_index % num_textures]
    
    if (not os.path.exists(output_folder)):
        os.mkdir(output_folder)
    
    if (os.path.exists(lock_file_path)):
        print("%s is currently processed. Skipping." % output_folder)
    
    elif (not os.path.exists(os.path.join(output_folder, "skeleton_left.json"))):
        print("Processing %s..." % output_folder)
        
        with open(lock_file_path, "w") as lock_file:
            lock_file.write("")
            
        subprocess.run(["blender", "body_part_splitter_skeleton_only.blend", "-P", "body_part_splitter_skeleton_only.py", "-b", "--", 
                        output_folder, gender, str(height), str(weight), pose_file_path, texture_path])
                        
        if (os.path.exists(lock_file_path)):
            os.remove(lock_file_path)

    else :
        print("%s already processed" % output_folder)
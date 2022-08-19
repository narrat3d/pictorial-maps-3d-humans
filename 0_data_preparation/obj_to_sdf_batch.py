'''
iterates through the training data for obj_to_sdf.py 
'''
import subprocess
import os
import argparse
import time
import random

root_folder = r"N:\Pictorial-Maps-DISN\data\narrat3d"

parser = argparse.ArgumentParser()
parser.add_argument('--part', dest='body_part_name')

args = parser.parse_args()
body_part_name = args.body_part_name

processed_folder_names = []

while (True):
    folder_names = os.listdir(root_folder)
    
    # all folders processed
    if (len(processed_folder_names) == len(folder_names)):
        break

    for folder_name in folder_names:
        if (folder_name in processed_folder_names):
            continue

        input_folder = os.path.join(root_folder, folder_name)

        lock_file_path = os.path.join(input_folder, "%s.lock" % body_part_name)
        output_file_path = os.path.join(input_folder, "%s.npy" % body_part_name)
        output_file_path2 = os.path.join(input_folder, "%s_surface.npy" % body_part_name)
        
        if (os.path.exists(lock_file_path)):
            print("Folder %s currently processed." % input_folder)
            
        elif (os.path.exists(output_file_path2)):
            print("Folder %s already processed." % input_folder)
            
        else :
            with open(lock_file_path, "w") as lock_file:
                lock_file.write("")
            
            print("Processing %s..." % input_folder)
            for view in ["front", "left", "back", "right"]:
                subprocess.run(["python", "obj_to_sdf_new_patched.py", "--input_folder", input_folder, 
                                "--output_file", output_file_path, "--output_file2", output_file_path2,
                                "--body_part_name", body_part_name, "--view", view])
            
            # sometimes duplicate processing can occur
            if (os.path.exists(lock_file_path)):
                os.remove(lock_file_path)
                
            time.sleep(random.random() * 10)
        
        processed_folder_names.append(folder_name)
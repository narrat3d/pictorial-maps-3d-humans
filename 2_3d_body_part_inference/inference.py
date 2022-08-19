import os
from train_and_eval import create_model, get_best_weights_file_path
from visualize import show_results
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default="output")
parser.add_argument("--model_name", default="disn_one_stream_pose")
parser.add_argument("--run_nr", default=1)
parser.add_argument("--out_folder", default=r"E:\CNN\implicit_functions\characters\output")
parser.add_argument("--sub_folders", default="")
args = parser.parse_args()

root_folder = args.out_folder
model_root_folder = args.model_folder
model_name = args.model_name
run_nr = args.run_nr
subfolder_names = os.listdir(root_folder) if args.sub_folders == "" else args.sub_folders.split("|")

class File_Checker():

    root_folder = None

    def __init__(self, root_folder):
        self.root_folder = root_folder
    
    def exists_file(self, subfolder_name, file_name):
        file_path = os.path.join(self.root_folder, subfolder_name, file_name)
        return os.path.exists(file_path)

file_checker = File_Checker(root_folder)

for body_part_name in ["torso", "head",
    "left_lower_arm", "left_upper_arm", "left_hand", 
    "left_upper_leg", "left_lower_leg", "left_foot",
    "right_lower_arm", "right_upper_arm", "right_hand", 
    "right_upper_leg", "right_lower_leg", "right_foot"]:
    needs_mirrowing = body_part_name.find("right") != -1
    
    model = create_model(model_name, body_part_name)
    
    model_folder = body_part_name.replace("right_", "left_") + "_" + model_name + "_run%s" % run_nr
    model_path = os.path.join(model_root_folder, model_folder)
    weights_file_path = get_best_weights_file_path(model_path)
    model.load_weights(weights_file_path)
    
    show_results(root_folder, subfolder_names, None, file_checker, 
                 model, body_part_name, ["front"], 
                 ground_truth_available=False, mirror=needs_mirrowing)
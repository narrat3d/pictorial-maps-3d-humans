import subprocess
import gc
import time
import os

input_folder = os.path.join(os.path.expanduser("~"), "Downloads", "tmp2")
output_folder = os.path.join(os.path.dirname(__file__), "experiments")

num_runs = 5

for body_part_name in ["torso", "head", "left_upper_arm", "left_lower_arm", "left_hand", "left_upper_leg", "left_lower_leg", "left_foot"]:
    for run_nr in range(num_runs):
        for model_name in ["disn_two_stream_pose"]: # "disn_two_stream", , "disn_one_stream", "disn_one_stream_pose"
            subprocess.run(["python", "train_and_eval.py", 
                            "--body_part_name", body_part_name, 
                            "--runs_from_generator", "True",
                            "--input_folder", input_folder,
                            "--output_folder", output_folder,
                            "--model_name", model_name,
                            "--run_nr", str(run_nr + 1),
                            "--eval_only", "True"])
            gc.collect()
            time.sleep(10)
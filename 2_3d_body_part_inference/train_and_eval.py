from config import EPOCHS, LEARNING_RATE, DEBUG, VIEWS, BATCH_SIZE
from models import BUILDS

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

import math
import os
import json
import argparse
from data_generators import train_data_generator, eval_data_generator
from visualize import show_results

current_folder = os.path.dirname(__file__)


def create_output_folder():
    os.makedirs(root_output_folder, exist_ok=True)
    
    run_name = body_part_name + "_" + model_name
    run_nr = 1
    
    output_folder = os.path.join(root_output_folder, run_name + "_run%s")
    
    while(os.path.exists(output_folder % run_nr)):
        run_nr += 1
    
    output_folder = output_folder % run_nr
    os.mkdir(output_folder)
    
    return output_folder

    
# alternative to os.path.exists, which may be slow via network drives
class File_Checker():

    training_data_files = None

    def __init__(self, file_list_path):
        with open(file_list_path) as file_list:
            self.training_data_files = json.load(file_list)
    
    def exists_file(self, subfolder_name, file_name):
        return file_name in self.training_data_files[subfolder_name]


def initialise_folders():
    global train_subfolder_names
    global test_subfolder_names
    global num_test_subfolders
    global file_checker
    
    file_list_path = os.path.join(input_folder, "index.json")
    file_checker = File_Checker(file_list_path)
    
    sdf_file_name = "%s.npy" % body_part_name
    subfolder_names = list(filter(lambda subfolder_name: file_checker.exists_file(subfolder_name, sdf_file_name), 
                                  file_checker.training_data_files.keys()))
    
    split_index = round(len(subfolder_names) * 0.9)
    train_subfolder_names = subfolder_names[:split_index] 
    test_subfolder_names = subfolder_names[split_index:] 
    
    
    if DEBUG:
        train_subfolder_names = train_subfolder_names[0:1] * 1000
        # the generator is called twice in each epoch...
        test_subfolder_names = train_subfolder_names[0:1] * 2
        num_test_subfolders = 1
    else :
        num_test_subfolders = len(test_subfolder_names) * len(VIEWS)


def get_weights_file_path(output_folder):
    weights_file_path = os.path.join(current_folder, output_folder, "weights.hdf5")
    return weights_file_path


def get_best_weights_file_path(output_folder):
    weights_file_path = os.path.join(current_folder, output_folder, "weights_best.hdf5")
    return weights_file_path


def create_model(model_name, body_part_name):
    if body_part_name == "torso":
        num_pose_points = 4
    else :
        num_pose_points = 2
    
    model_builder = BUILDS[model_name]
    model_inputs, model_outputs = model_builder(num_pose_points)
    
    model = Model(inputs=model_inputs, outputs=model_outputs)  
    model.summary()
    
    return model


def train_and_eval(body_part_name):
    model = create_model(model_name, body_part_name)
    use_pose_points = len(model.inputs) == 3
    
    train_gen = train_data_generator(input_folder, train_subfolder_names, file_checker, body_part_name, use_pose_points)
    eval_gen = eval_data_generator(input_folder, test_subfolder_names, file_checker, body_part_name, use_pose_points, eval_all_coords=False)
    
    """
    from tensorflow.keras.utils import plot_model
    plot_file_path = os.path.join(temp_folder, "model.png")
    plot_model(model, to_file=plot_file_path, show_shapes=True)
    """
   
    model.compile(loss=["mse"] * len(model.outputs), optimizer=Adam(learning_rate=LEARNING_RATE)) # , run_eagerly=True
    weights_file_path = get_weights_file_path(output_folder)
    
    if (os.path.exists(weights_file_path)):
        try :
            model.load_weights(weights_file_path)
        except:
            print("Weights could not be loaded.")
    
    checkpoint = ModelCheckpoint(weights_file_path, save_weights_only=True, save_best_only=False)
    
    best_weights_file_path = get_best_weights_file_path(output_folder)
    
    checkpoint_best = ModelCheckpoint(best_weights_file_path, save_weights_only=True, monitor="val_loss",
                                      mode="min", verbose=2, save_best_only=True)  

    # to prevent that two Excel files cannot be opened with the same name
    output_folder_name = os.path.basename(output_folder)
    csv_log_file_path = os.path.join(output_folder, "%s.csv" % output_folder_name)
    csv_logger = CSVLogger(csv_log_file_path, ";")

    steps = math.ceil(len(train_subfolder_names) / BATCH_SIZE)
    
    verbosity = 2 if runs_from_generator else 1
    
    model.fit(train_gen, validation_data=eval_gen, epochs=EPOCHS, steps_per_epoch=steps, verbose=verbosity,
              validation_steps=num_test_subfolders, callbacks=[checkpoint, checkpoint_best, csv_logger])

    return model


# reason: https://stackoverflow.com/questions/43055289/kerass-model-fit-is-not-consistent-with-model-evaluate
def evaluate(body_part_name):
    model = create_model(model_name, body_part_name)
    use_pose_points = len(model.inputs) == 3
    
    eval_gen = eval_data_generator(input_folder, test_subfolder_names, file_checker, body_part_name, use_pose_points, eval_all_coords=True)
       
    model.compile(loss=["mse"] * len(model.outputs), optimizer=Adam(learning_rate=LEARNING_RATE))
    
    best_weights_file_path = get_best_weights_file_path(output_folder)   
    model.load_weights(best_weights_file_path)
    
    verbosity = 2 if runs_from_generator else 1
    
    output_folder_name = os.path.basename(output_folder)
    csv_log_file_path = os.path.join(output_folder, "%s_eval.csv" % output_folder_name)

    result = model.evaluate_generator(eval_gen, steps=num_test_subfolders, 
                                      verbose=verbosity)

    # CSV logger does not seem to work: https://github.com/keras-team/keras/issues/9484
    with open(csv_log_file_path, "w") as csv_file:
        csv_file.write("val_loss\n%s" % result)

    return result
    
            
if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('--body_part_name', dest='body_part_name', default="left_hand")
    parser.add_argument('--runs_from_generator', dest='runs_from_generator', default=False)
    parser.add_argument('--model_name', dest='model_name', default="disn_one_stream")
    parser.add_argument('--input_folder', dest='input_folder', default=os.path.join(os.path.expanduser("~"), "Downloads", "tmp2"))
    parser.add_argument('--output_folder', dest='output_folder', default="output")
    
    args = parser.parse_args()
    body_part_name = args.body_part_name
    runs_from_generator = bool(args.runs_from_generator)
    model_name = args.model_name
    input_folder = args.input_folder
    root_output_folder = args.output_folder
    
    initialise_folders()
    
    output_folder = create_output_folder()
    model = train_and_eval(body_part_name)
    weights_file_path = get_best_weights_file_path(output_folder)
    model.load_weights(weights_file_path)
    show_results(input_folder, test_subfolder_names, output_folder, file_checker,
                 model, body_part_name, VIEWS, runs_from_generator=runs_from_generator)
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import time
from visualize import calculate_uv_map
from data_loader import load_input_data, load_output_data
from config import IMAGE_SIZE, NUM_FILTERS, BATCH_SIZE, NUM_BODY_PARTS, TRAINING_DATA_FOLDER, DEBUG
from model import create_model

current_folder = os.path.dirname(__file__)
output_folder = os.path.join(current_folder, "output")

if (not os.path.exists(output_folder)):
    os.mkdir(output_folder)

log_file = None

best_weights_file_path = os.path.join(output_folder, "weights_best.hdf5")
weights_file_path = os.path.join(output_folder, "weights.hdf5")
log_file_path = os.path.join(output_folder, "log.txt")
loss_log_file_path = os.path.join(output_folder, "loss.txt")
predicted_image_path = os.path.join(output_folder, "prediction.png")

data_folder = TRAINING_DATA_FOLDER

optimizer = Adam(lr=0.001)

subfolders = os.listdir(data_folder)

subfolders = list(map(lambda subfolder: os.path.join(data_folder, subfolder), subfolders))
subfolders = list(filter(os.path.isdir, subfolders))

split_index = round(len(subfolders) * 0.9) 
train_subfolders = subfolders[:split_index]
test_subfolders = subfolders[split_index:]

cache = {}

if DEBUG:
    train_subfolders = train_subfolders[:1]
    test_subfolders = train_subfolders

@tf.function
def calc_mean_error(masked_error, masked_uv):
    squared_error_sum = tf.reduce_sum(masked_error, axis=[1, 2])
    covered_pixels = tf.reduce_sum(masked_uv, axis=[1, 2])
    mean_error = tf.reduce_mean(squared_error_sum / covered_pixels)
    
    return mean_error

@tf.function
def mae_loss(ytrue, ypred): 
    # only consider available uv coordinates
    masked_uv = tf.stack([ytrue[..., 0], ytrue[..., 0]], axis=-1)
    masked_error = masked_uv * tf.abs(ytrue[..., 1:] - ypred[..., 1:])

    mean_uv_error = calc_mean_error(masked_error, masked_uv)
    return mean_uv_error

@tf.function
def rmse_loss(ytrue, ypred): 
    # only consider available uv coordinates
    masked_uv = tf.stack([ytrue[..., 0], ytrue[..., 0]], axis=-1)
    masked_squared_error = masked_uv * tf.square(ytrue[..., 1:] - ypred[..., 1:])

    mean_uv_error = calc_mean_error(masked_squared_error, masked_uv)
    root_mean_uv_error = tf.sqrt(mean_uv_error)
    
    return root_mean_uv_error

# @tf.function
def train_step(model, x, y_true):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = rmse_loss(y_true, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return tf.reduce_mean(loss)

@tf.function
def eval_step(model, depth_maps, uv_maps):
    predicted_uv_maps = model(depth_maps)
            
    loss = mae_loss(uv_maps, predicted_uv_maps)
    
    return loss


def log(*args, end="\n"):
    print(*args, end=end)
    
    text = " ".join(map(str, args))
    log_file.write(text + end)
    log_file.flush()


def loop(subfolders, model, stepping_func):
    losses = []
    depth_maps = []
    uv_maps = []
    
    for i, subfolder in enumerate(subfolders):        
        for view in ["front", "left", "back", "right"]:
            depth_map, _ = load_input_data(subfolder, view)
            uv_map = load_output_data(subfolder, view)
            
            depth_maps.append(depth_map)
            uv_maps.append(uv_map)
        
        if (i % BATCH_SIZE == BATCH_SIZE - 1):
            log(".", end ="")
            depth_maps = np.concatenate(depth_maps, axis=0) 
            uv_maps = np.concatenate(uv_maps, axis=0) 
            
            loss = stepping_func(model, depth_maps, uv_maps)
            losses.append(loss)
            
            depth_maps = []
            uv_maps = []
    
    avg_loss = tf.reduce_mean(losses)
    return avg_loss.numpy()

if (os.path.exists(loss_log_file_path)):
    with open(loss_log_file_path) as loss_log_file:
        lowest_eval_loss = float(loss_log_file.read())
else :
    lowest_eval_loss = 1


def train_and_eval():
    global log_file
    start = time.time()

    log_file = open(log_file_path, "w")

    model = create_model(IMAGE_SIZE, NUM_BODY_PARTS, NUM_FILTERS)
    
    if (os.path.exists(best_weights_file_path)):
        model.load_weights(best_weights_file_path)
    
    # loss = loop(train_subfolders, model, train_step)
    # log("train loss", loss)
    
    loss = loop(test_subfolders, model, eval_step)
    log("eval loss", loss)
    
    # model.save_weights(weights_file_path)
    
    if (loss < lowest_eval_loss):
        log("New record!")
        # with open(loss_log_file_path, "w") as loss_log_file:
        #     loss_log_file.write(str(loss.item()))
            
        # model.save_weights(best_weights_file_path)
        
        depth_map, _ = load_input_data(test_subfolders[0], "front")
        uv_map = load_output_data(test_subfolders[0], "front")[0]
        
        mask = uv_map[..., :1]
        uv_map = uv_map[..., 1:]
        
        stacked_mask = np.concatenate([mask, mask], axis=2)
        
        uv_map = model(depth_map)
        uv_map_image = calculate_uv_map(stacked_mask * uv_map[0, ..., 1:])
        uv_map_image.save(predicted_image_path)
    
    log("time", round(time.time() - start, 2))
    
    log_file.flush()
    log_file.close()
    
    
if __name__ == '__main__':
    train_and_eval()
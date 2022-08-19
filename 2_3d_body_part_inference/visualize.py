import numpy as np
from PIL import Image
from data_generators import create_coords, load_input_data, load_output_data
from config import IMAGE_SIZE, SCALING_FACTOR
import os
from numba import cuda
import matplotlib.pyplot as plt
import math
from keras.losses import mean_squared_error
import tensorflow as tf


# make figure larger
plt.rcParams["figure.figsize"] = (6,9)


def occupancy_field(distance_field, file_path=None):
    grid_size = IMAGE_SIZE
    
    grid = np.indices((grid_size, grid_size, grid_size))
    indices = np.moveaxis(grid, 0, -1)
    
    coords = -1 + indices / (grid_size/2)
    xs = coords[:, :, :, 0]
    ys = coords[:, :, :, 1]
    zs = coords[:, :, :, 2]
    
    distances = distance_field.flatten()
    condition = distances < 0
    
    if (np.sum(condition) == 0):
        return
    
    from mayavi import mlab
    
    fig = mlab.figure(size=(256,256), bgcolor=(0., 0., 0.))    
    mlab.points3d(xs.flatten()[condition], ys.flatten()[condition], zs.flatten()[condition], distances[condition], color=(1., 1., 1.), scale_mode="none", figure=fig,
                  reset_zoom=False)
    # source: https://stackoverflow.com/questions/32514744/setting-parallel-prospective-in-mlab-mayavi-python
    fig.scene.parallel_projection = True
    fig.scene.camera.parallel_scale = 1
    
    fig.scene._tool_bar.setVisible(False)
    mlab.view(azimuth=0, elevation=0, focalpoint=(0,0,0), figure=fig)
    
    if file_path is None:
        mlab.show()
    else :
        mlab.savefig(file_path, figure=fig)


def show_results(input_folder, test_subfolder_names, output_folder, file_checker,
                 model, body_part_name, views, 
                 ground_truth_available=True, mirror=False, runs_from_generator=False):
    use_pose_points = len(model.inputs) == 3
    
    # test data
    for subfolder_nr, subfolder_name in enumerate(test_subfolder_names):
        print(subfolder_name)
        
        for view in views: 
            if ground_truth_available:
                points_3d_ground_truth = load_output_data(input_folder, subfolder_name, body_part_name, view)
            else :
                points_3d_ground_truth = None
                
            mask_points_2d, pose_points_test = load_input_data(input_folder, subfolder_name, file_checker, body_part_name, view, points_3d_ground_truth)
     
            if (mirror):
                mask_points_2d = np.flip(mask_points_2d, axis=1)       
                pose_points_test = np.apply_along_axis(lambda coords: [-coords[0], coords[1], coords[2]], 2, pose_points_test)
                    
            coords = create_coords()
            
            inputs = [mask_points_2d, coords]
            
            if (use_pose_points):
                inputs.append(pose_points_test)
            
            result = model(inputs)
            distance_field = (result[0]).numpy()

            distance_field = np.reshape(distance_field, (IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE))
        
            points_3d_predicted = distance_field * SCALING_FACTOR   
            
            if (mirror):
                points_3d_predicted = np.flip(points_3d_predicted, axis=0) 
            
            if (not ground_truth_available):
                np.save(os.path.join(input_folder, subfolder_name, "%s.npy" % body_part_name), points_3d_predicted)
                
                points_2d = np.min(points_3d_predicted, axis=2)
                binary_image = 1 - np.heaviside(points_2d, 0)
                binary_image = pil_image(binary_image)
                binary_image.save(os.path.join(input_folder, subfolder_name, "%s_%s_pred.png" % (body_part_name, view)))
            
            else :
                loss = tf.reduce_mean(mean_squared_error(distance_field, points_3d_ground_truth[0]))
                print(loss.numpy().item())
                
                _, axs = plt.subplots(3, 2)
                axs[0][1].imshow(pil_image(mask_points_2d[0, :, :, 0] + 0.5), cmap='gray', vmin=0, vmax=255)
                axs[0][1].set_title("Test image")
                hide_ticks(axs[0][1])
            
                raymarched_image = np.ones((IMAGE_SIZE, IMAGE_SIZE)) 
                raymarch_distance_field_front[IMAGE_SIZE, IMAGE_SIZE](raymarched_image, points_3d_ground_truth[0])
                
                axs[1][1].imshow(pil_image(raymarched_image), cmap='gray', vmin=0, vmax=255)
                axs[1][1].set_title("Ground truth")
                hide_ticks(axs[1][1])

                # when result[1] ==> axis = 2, result[2] ==> axis = 1, result[3] => axis = 0
                points_2d = np.min(distance_field, axis=2)
                axs[1][0].imshow(grad_image(points_2d), cmap='seismic', vmin=0, vmax=1)
                axs[1][0].set_title("Predicted field")
                hide_ticks(axs[1][0])
                
                points_2d_gt = np.min(points_3d_ground_truth[0], axis=2)
                axs[0][0].imshow(grad_image(points_2d_gt), cmap='seismic', vmin=0, vmax=1)
                axs[0][0].set_title("Ground truth field")
                hide_ticks(axs[0][0])
                
                points_2d_diff = points_2d - points_2d_gt
                axs[2][0].imshow(grad_image(points_2d_diff / 2), cmap='seismic', vmin=0, vmax=1)
                axs[2][0].set_title("Field difference")
                hide_ticks(axs[2][0])
        
                binary_image = np.heaviside(points_2d, 0)
                axs[2][1].imshow(pil_image(binary_image), cmap='gray', vmin=0, vmax=255)
                axs[2][1].set_title("Predicted image")
                hide_ticks(axs[2][1])
                
                if (runs_from_generator):
                    plt.savefig(os.path.join(output_folder, 'test_image_%s_%s.png' % (subfolder_nr, view)))
                    continue
                else:
                    plt.show()

                    
                occupancy_field(distance_field)
        
        if (runs_from_generator and subfolder_nr > 9):
            break
        
        
def hide_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    
def pil_image(points_2d):
    mirrored_image = np.flip(points_2d, axis=1)
    yx_image = np.swapaxes(mirrored_image, 1, 0)
    int_image = np.uint8(yx_image * 255)
    image = Image.fromarray(int_image, "L")
    
    return image


def grad_image(points_2d):
    points_2d[points_2d < -1] = -1
    points_2d[points_2d > 1] = 1
    
    points_2d = points_2d *0.5 + 0.5
    
    mirrored_image = np.flip(points_2d, axis=1)
    yx_image = np.swapaxes(mirrored_image, 1, 0)
    
    return yx_image


@cuda.jit
def raymarch_distance_field_front(result, distance_field):
    pos = cuda.grid(1)

    if (pos < result.size):
        i = int(math.floor(pos / IMAGE_SIZE))
        j = pos - i * IMAGE_SIZE
        
        y = j
        x = i
        z = 0
        
        while(True):
            dist = distance_field[x][y][z]
            
            if (dist < 0):
                result[i][j] = 0
                break
            
            z += max(int(dist), 1)
            
            if (z > IMAGE_SIZE - 1):
                break
            
            
if __name__ == '__main__':
    from config import VIEWS
    from train_and_eval import input_folder, root_output_folder, get_best_weights_file_path,\
        create_model, test_subfolder_names, file_checker
    
    body_part_name = "left_hand"
    model_name = "disn_one_stream_pose"
    run_nr = 1
    
    output_folder = os.path.join(root_output_folder, body_part_name + "_" + model_name + "_run%s" % run_nr)
    
    model = create_model(model_name, body_part_name)
    weights_file_path = get_best_weights_file_path(output_folder)
    model.load_weights(weights_file_path)
    show_results(input_folder, test_subfolder_names, output_folder, # ["rp_steve_posed_001_0_0_male_small"]
                 file_checker, model, body_part_name, VIEWS)
    """
    
    folder = r"E:\CNN\implicit_functions\characters\output\Patagonischer_Riese_links"
    
    for body_part_name in ["left_hand", "left_upper_arm", "left_lower_arm", "left_lower_leg", "left_upper_leg", "left_foot", "torso",
                           "right_hand", "right_upper_arm", "right_lower_arm", "right_lower_leg", "right_upper_leg", "right_foot", "head"]:
        distance_field = np.load(os.path.join(folder, "%s.npy" % body_part_name))
        occupancy_field(distance_field, file_path=os.path.join(folder, "%s_front_pred_3d.png" % body_part_name))
	"""
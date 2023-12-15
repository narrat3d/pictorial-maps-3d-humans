from numba import cuda
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import math
import sys
from optimization.scene.formulas_numba import cuboid, normalize2, add2, sub2, mult2,\
    dot2, sphere, translate, union, union_smooth, rotate_y, rotate_x, rotate_z,\
    negate, tx
from scipy.ndimage import gaussian_filter
import json
import os
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.use('TkAgg')

body_size = 300
body_size_half = body_size / 2
image_size = 256 + 1
body_part_size = 64.
body_part_size_half = body_part_size / 2

POSE_POINTS = {
    'right_ankle': 0, 
    'right_knee': 1,
    'right_hip': 2,
    'left_hip': 3,
    'left_knee': 4,
    'left_ankle': 5,
    'pelvis': 6, # hip
    'root': 7, # thorax
    'neck': 8,
    'head': 9,
    'right_wrist': 10,   
    'right_elbow': 11,
    'right_shoulder': 12,
    'left_shoulder': 13,
    'left_elbow': 14,
    'left_wrist': 15,
    'right_foot': 16,
    'left_foot': 17,
    'right_middle1': 18,
    'left_middle1': 19,
    'eyes': 20,
}

BONES = {
    "torso": ['right_shoulder', 'left_shoulder', 'left_hip', 'right_hip'],
    "head": ["head"],
    "right_upper_arm": ['right_shoulder', 'right_elbow'],
    "right_lower_arm": ['right_elbow', 'right_wrist'],
    "right_hand": ['right_wrist'],
    "right_upper_leg": ['right_hip', 'right_knee'],
    "right_lower_leg": ['right_knee', 'right_ankle'],
    "right_foot": ['right_ankle'],
    "left_upper_leg": ['left_hip', 'left_knee'],
    "left_lower_leg": ['left_knee', 'left_ankle'],
    "left_foot": ['left_ankle'],
    "left_upper_arm": ['left_shoulder', 'left_elbow'],
    "left_lower_arm": ['left_elbow', 'left_wrist'],
    "left_hand": ['left_wrist']     
}

RGB_COLORS = (
    (0.25, 0.25, 0.25),
    (0.75, 0.75, 0.75),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (0.5, 0.5, 0.0),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.0, 0.5),
)


def calculate_mid_point(pose_points):
    mid_point = np.average(pose_points, axis=0)
    mid_point[1] = 600 - mid_point[1]
    centred_mid_point = mid_point - 300
    return centred_mid_point / (600 / body_size)


def raymarch(root_folder):
    skeleton_original = json.load(open(os.path.join(root_folder, "skeleton_front.json")))
    skeleton_animation = np.load(os.path.join(root_folder, "skeleton_animation.npy"))
    body_part_rotations = np.load(os.path.join(root_folder, "rotation_animation.npy"))
    num_frames = skeleton_animation.shape[0]

    object_translations = []
    body_part_translations = np.zeros((num_frames, 14, 3))
    sdf_objects = []
    scales = []
    textures = []
    
    for bone_index, bone_name in enumerate(BONES.keys()):
        keypoints = BONES[bone_name]
        pose_points = []
        
        for keypoint in keypoints:
            keypoint_index = POSE_POINTS[keypoint]
            pose_point = skeleton_original[str(keypoint_index)]
            pose_points.append(pose_point)
        
        mid_point = calculate_mid_point(pose_points)
        object_translations.append(mid_point)
        
        for frame_nr in range(num_frames):
            skeleton = skeleton_animation[frame_nr]
            pose_points = []
            
            for keypoint in keypoints:
                keypoint_index = POSE_POINTS[keypoint]
                pose_point = skeleton[keypoint_index]
                pose_points.append(pose_point)
        
            mid_point_animation = calculate_mid_point(pose_points)
            body_part_translation = mid_point_animation - mid_point
            body_part_translations[frame_nr, bone_index] = body_part_translation
                
        object_path = os.path.join(root_folder, "%s.npy" % bone_name)
        
        if (not os.path.exists(object_path)):
            continue
        
        sdf_object = np.load(object_path)
        sdf_objects.append(sdf_object)
        
        body_part_metadata = json.load(open(os.path.join(root_folder, "%s.json" % bone_name)))
        scale = body_part_metadata["scale"]
        scales.append(scale) 
    
    print(body_part_translations[:, 2, :])
    
    """
    rotation = body_part_rotations[2]
    [start_point_name, end_point_name] = ['right_shoulder', 'right_elbow']
    keypoint_index = POSE_POINTS[start_point_name]
    start_point = skeleton_original[str(keypoint_index)]
    keypoint_index = POSE_POINTS[end_point_name]
    end_point = skeleton_original[str(keypoint_index)]
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    
    mid_point = np.average([start_point, end_point], 0)
    mid_point[1] = 600 - mid_point[1]
    mid_point = (np.array(mid_point) - 300) / 2.
    
    start_point[1] = 600 - start_point[1]
    end_point[1] = 600 - end_point[1]
    
    start_point = (start_point - 300) / 2.
    end_point = (end_point - 300) / 2.
    
    bone_vector = end_point - start_point
    bone_vector = bone_vector / np.linalg.norm(bone_vector)

    rot = Rotation.from_rotvec(-rotation[0] * np.array([0, 1, 0]))
    bone_vector = rot.apply(bone_vector)
    
    rot = Rotation.from_rotvec(rotation[1] * np.array([0, 0, 1]))
    bone_vector = rot.apply(bone_vector)
    print(bone_vector)
    """
    
    for view in ["front_cropped", "left", "back", "right", "front"]:
        texture = Image.open(os.path.join(root_folder, "body_parts_%s_texture.png" % view)) # body_parts_%s_texture
        texture = texture.convert("RGB")
        texture = texture.resize((body_size, body_size))
        texture = np.asarray(texture) / 255.
        texture = np.swapaxes(texture, 0, 1)
        texture = np.flip(texture, axis=1)
        textures.append(texture)
    
    mask = Image.open(os.path.join(root_folder, "body_parts_front_mask.png"))
    mask = mask.resize((body_size, body_size), resample=Image.NEAREST)
    
    
    sdf_objects = np.array(sdf_objects)
    object_translations = np.array(object_translations)
    body_part_translations = np.array(body_part_translations)
    body_part_rotations = np.array(body_part_rotations)
    # object_translations = np.zeros((14, 3))
    scales = np.array(scales) * 0.5
    textures = np.array(textures)

    global camera_position
    global camera_forward
    global camera_right
    global camera_up
    global camera_distance
    global camera_coords
    global target_point
    global mouse_xy
    global render_loop
    
    render_loop = True
    
    mouse_xy = None
    # longitude and latitude in radians
    camera_coords = np.array([0.0, 0.0])
    # center of the coordinate system
    target_point = np.array([0.0, 0.0, 0.0])
    camera_distance = body_size
    camera_position = None
    camera_forward = None
    camera_right = None
    camera_up = None

    pan()
    
    plt.ion()
    ax = plt.gca()

    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    canvas = ax.figure.canvas
    canvas.manager.window.wm_geometry("+%d+%d" % (0, 0))
    
    canvas.mpl_connect("button_press_event", on_press)
    canvas.mpl_connect("button_release_event", on_release)
    canvas.mpl_connect("motion_notify_event", on_move)
    canvas.mpl_connect("scroll_event", on_scroll)
    canvas.mpl_connect("close_event", on_close)
    
    result = np.zeros((image_size, image_size, 3), dtype=np.uint16)
    im = plt.imshow(result)
    
    time = 0

    # rendering loop
    while(render_loop):
        body_part_translation = body_part_translations[time % num_frames]
        body_part_rotation = body_part_rotations[time % num_frames]
        result = np.ones((image_size, image_size, 3), dtype=np.uint16) * 255 # white background
        # result = np.zeros((image_size, image_size, 3), dtype=np.uint16)
        my_kernel[image_size, image_size](result, camera_position, camera_forward, 
                                          camera_up, camera_right, camera_distance, 
                                          sdf_objects, body_part_rotation, object_translations, 
                                          body_part_translation, scales, textures, time)
        
        im.set_data(result)
           
        canvas.draw()
        canvas.flush_events()
        
        time += 1 
        
        # rotating model
        # camera_coords[0] += -0.05
        # pan()



def smooth(arr, filter_size):
    padding_size = int((filter_size - 1) / 2)
    padded_arr = np.pad(arr, ((padding_size, padding_size), (padding_size, 2)), 'constant', constant_values=np.nan)
    
    new_arr = np.full_like(arr, np.nan)
    
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            if (np.isnan(arr[x][y])):
                continue
            
            surrounding_values = []
            
            for i in range(filter_size):
                for j in range(filter_size):
                    surrounding_values.append(padded_arr[x + i][y + j])
            
            new_arr[x][y] = np.nanmean(np.array(surrounding_values))
    
    return new_arr


@cuda.jit(device=True)
def shapes(point, sdf_objects, body_part_rotations, object_translations, body_part_translations, scales, time):
    """
    position = translate(point, (50, 25, 0))
    distance = sphere(position, 25)
    return (distance, 1000.)
    """
    
    [x, y, z] = point
    distance = 10000.
    min_distance = 10000.
    min_index = 0
    
    for i in range(14): # :
        sdf_object = sdf_objects[i]
        [xt, yt, zt] = object_translations[i]

        scale = scales[i]

        point = translate((x, y, z), (xt, yt, zt))
        point = translate(point, body_part_translations[i])
        
        point = rotate_y(point, body_part_rotations[i][3])
        point = rotate_z(point, -body_part_rotations[i][2])
        point = rotate_z(point, body_part_rotations[i][1])
        point = rotate_y(point, -body_part_rotations[i][0])
        
        # point = tx(point, body_part_rotations[i])
               
        [xp, yp, zp] = point
        
        scaling_factor = scale / body_part_size
        scaled_size = body_part_size_half * scaling_factor
        bounding_cube_size = (body_part_size_half - 6) * scaling_factor / 2
        
        distance2 = cuboid(point, (bounding_cube_size, bounding_cube_size, bounding_cube_size))
        # distance2 = sphere(point, bounding_cube_size)

        gx = (xp + scaled_size) / scaling_factor
        gy = (yp + scaled_size) / scaling_factor
        gz = (zp + scaled_size) / scaling_factor
        x1 = int(gx)
        y1 = int(gy)
        z1 = int(gz)
        
        if (x1 >= 0 and x1 < body_part_size - 1 and y1 >= 0 and y1 < body_part_size - 1 and z1 >= 0 and z1 < body_part_size - 1):
            dx = gx - x1
            dy = gy - y1
            dz = gz - z1
            
            c00 = sdf_object[x1][y1][z1] * (1 - dx) + sdf_object[x1+1][y1][z1] * dx
            c01 = sdf_object[x1][y1][z1+1] * (1 - dx) + sdf_object[x1+1][y1][z1+1] * dx
            c10 = sdf_object[x1][y1+1][z1] * (1 - dx) + sdf_object[x1+1][y1+1][z1] * dx
            c11 = sdf_object[x1][y1+1][z1+1] * (1 - dx) + sdf_object[x1+1][y1+1][z1+1] * dx

            c0 = c00 * (1 - dy) + c10 * dy
            c1 = c01 * (1 - dy) + c11 * dy
            
            distance2 = c0 * (1 - dz) + c1 * dz
            distance2 *= scaling_factor

        distance = union(distance, distance2)
        min_distance = union_smooth(min_distance, distance2, 2)
        
        if (distance == distance2):
            min_index = i
        
    return min_distance, min_index

# adapted from http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/#surface-normals-and-lighting
@cuda.jit(device=True)
def estimate_normal(ray_point, sdf_objects, body_part_rotations, object_translations, body_part_translations, scales, time):
    EPSILON = 1
    [x, y, z] = ray_point
    
    vector = (
        shapes((x + EPSILON, y, z), sdf_objects, body_part_rotations, object_translations, body_part_translations, scales, time)[0] - 
        shapes((x - EPSILON, y, z), sdf_objects, body_part_rotations, object_translations, body_part_translations, scales, time)[0],
        shapes((x, y + EPSILON, z), sdf_objects, body_part_rotations, object_translations, body_part_translations, scales, time)[0] - 
        shapes((x, y - EPSILON, z), sdf_objects, body_part_rotations, object_translations, body_part_translations, scales, time)[0],
        shapes((x, y, z + EPSILON), sdf_objects, body_part_rotations, object_translations, body_part_translations, scales, time)[0] - 
        shapes((x, y, z - EPSILON), sdf_objects, body_part_rotations, object_translations, body_part_translations, scales, time)[0]
    )
    normal = normalize2(vector)
    
    return normal

@cuda.jit(device=True)
def rotate_2d(x, y, angle):
    rot_x = math.cos(angle) * x - math.sin(angle) * y
    rot_y = math.sin(angle) * x + math.cos(angle) * y
    
    return (rot_x, rot_y)


@cuda.jit()
def my_kernel(result, camera_position, camera_forward, camera_up, camera_right, camera_distance, 
              sdf_objects, body_part_rotations, object_translations, body_part_translations, scales, textures, time):
    # time = 0
    pos = cuda.grid(1)
    
    if (pos < result.size):
        i = int(math.floor(pos / image_size))
        j = pos - i * image_size
        
        # normalize pixel coordinates
        half_block_width = int((image_size - 1) / 2) 
        y = (j - half_block_width) / (image_size - 1)
        x = (i - half_block_width) / (image_size - 1)
        
        # ray_direction = camera_forward + camera_right * x + camera_up * y
        ray_direction = add2(camera_forward, add2(mult2(camera_right, x), mult2(camera_up, y)))
        ray_direction = normalize2(ray_direction)
        
        stepped_distance = 0.0
        
        while(True):
            # ray_point = camera_position + ray_direction * stepped_distance
            ray_point = add2(camera_position, mult2(ray_direction, stepped_distance))
            
            distance, min_index = shapes(ray_point, sdf_objects, body_part_rotations, object_translations, body_part_translations, scales, time)
            
            if (distance < 1):
                    
                translated_ray_point = translate(ray_point, object_translations[min_index])
                translated_ray_point = translate(translated_ray_point, body_part_translations[min_index])
                
                rotated_ray_point = rotate_y(translated_ray_point, body_part_rotations[min_index][3])
                rotated_ray_point = rotate_z(rotated_ray_point, -body_part_rotations[min_index][2])
                rotated_ray_point = rotate_z(rotated_ray_point, body_part_rotations[min_index][1])
                rotated_ray_point = rotate_y(rotated_ray_point, -body_part_rotations[min_index][0])
                
                rotated_ray_point = translate(rotated_ray_point, negate(object_translations[min_index]))

                surface_normal = estimate_normal(ray_point, sdf_objects, body_part_rotations, object_translations, body_part_translations, scales, time)
                light_intensity = abs(dot2(surface_normal, ray_direction))
        
                rotated_surface_normal = rotate_y(surface_normal, body_part_rotations[min_index][3])
                rotated_surface_normal = rotate_z(rotated_surface_normal, -body_part_rotations[min_index][2])
                rotated_surface_normal = rotate_z(rotated_surface_normal, body_part_rotations[min_index][1])
                rotated_surface_normal = rotate_y(rotated_surface_normal, -body_part_rotations[min_index][0])

                [x_n, _, z_n] = rotated_surface_normal # 
                [x_hit, y_hit, z_hit] = rotated_ray_point # 
                
                # needed for depth calculation
                # [x_n, _, z_n] = camera_position
                
                # needed for four-variate mapping and depth calculation
                angle = math.atan2(x_n, z_n)
                angle = angle % (2*math.pi)
                texture_index = angle / (2*math.pi) * 4
                texture_index = round(texture_index % 3.5)
                angle = (texture_index / 4) * 2*math.pi

                """
                # bi-variate mapping
                if (z_n > 0):
                    texture_index = 0
                    angle = 0
                else :
                    texture_index = 2
                    angle = math.pi
                """

                #  texture mapping
                u = x_hit * math.cos(angle) - math.sin(angle) * z_hit
                [r, g, b] = textures[texture_index][int(u + body_size_half)][int(y_hit + body_size_half)]
                
                """mask_value = mask[int(x_hit + body_size_half)][int(y_hit + body_size_half)]
                
                if (mask_value == 0):
                    [r, g, b] = (1., 1., 1.)"""
                
                # body part colors
                # [r, g, b] = RGB_COLORS[min_index]
                # (r, g, b) = (0.5, 0.5, 0.5)
                """
                # depth
                u = z_hit * math.cos(angle) + math.sin(angle) * x_hit
                depth = ((u / body_size * 0.5) + 0.5) 
                [r, g, b] = (1 - depth, 1 - depth, 1 - depth)
                """
                
                result[image_size - 1 - j][i][0] = r * 255
                result[image_size - 1 - j][i][1] = g * 255
                result[image_size - 1 - j][i][2] = b * 255
                break

            stepped_distance += distance
            
            if (stepped_distance > camera_distance*2):
                break
        



def normalize(vector):
    return vector / np.linalg.norm(vector)


def on_press(event):    
    global mouse_xy
    mouse_xy = np.array([event.xdata, -event.ydata])


def on_move(event):
    global mouse_xy
    global camera_coords
    
    if (mouse_xy is None): 
        return; 
    
    current_xy = np.array([event.xdata, -event.ydata])
    
    offset = ((mouse_xy - current_xy) / image_size) * 2 * math.pi
    camera_coords += offset
    
    # prevent overflow of latitude
    camera_coords[1] = max(min(camera_coords[1], math.pi / 2 - 0.01), -math.pi / 2 + 0.01)
    
    mouse_xy = current_xy
    pan()


def on_release(_):
    global mouse_xy
    mouse_xy = None
    

def on_scroll(event):
    global camera_distance
    
    if (event.button == "up"):
        camera_distance *= 0.9
    elif (event.button == "down"):
        camera_distance /= 0.9
        
    pan()

def on_close(_):
    global render_loop
    render_loop = False

# right-handed coordinate system (x = thumb, y = pointer, z = middle finger)
def pan():
    global camera_position
    global camera_forward
    global camera_right
    global camera_up
    global target_point
    
    lon = camera_coords[0]
    lat = camera_coords[1]

    # rotation according to right-hand rule
    camera_position = [0, 0, camera_distance]
    # first rotate around the x-axis
    rotation_x = Rotation.from_rotvec([-lat, 0, 0])
    camera_position = rotation_x.apply(camera_position)
    # then rotate around the y-axis
    rotation_y = Rotation.from_rotvec([0, lon, 0])
    camera_position = rotation_y.apply(camera_position)
    
    # adapted from https://stackoverflow.com/questions/3427379/effective-way-to-calculate-the-up-vector-for-glulookat-which-points-up-the-y-axi
    camera_forward = normalize(target_point - camera_position)
    camera_right = normalize(np.cross(camera_forward, np.array([0, 1, 0])))
    camera_up = np.cross(camera_right, camera_forward)


if __name__ == '__main__':
    root_folder = r"E:\CNN\implicit_functions\characters\output"
    # root_folder = r"D:\Repositories\pictorial-maps-3d-humans\data\out"

    subfolders = os.listdir(root_folder)
        
    for subfolder in ["Antoine_Corbineau_Metropolitan_Eurostar_Bruxelles_Brugman_map"]: # subfolders: #
        subfolder_path = os.path.join(root_folder, subfolder)
        print(subfolder_path)
        raymarch(subfolder_path)
from numba import cuda
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import math
from formulas_numba import cuboid, normalize2, add2, mult2, dot2, sphere, translate, union, union_smooth, rotate_y, difference, min2
import json
import open3d as o3d
import pymeshlab
import os
from PIL import Image
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="render")
parser.add_argument("--out_folder", default=r"E:\CNN\implicit_functions\characters\output")
parser.add_argument("--sub_folders", default="")
args = parser.parse_args()

mode = args.mode
root_folder = args.out_folder
subfolder_names = os.listdir(root_folder) if args.sub_folders == "" else args.sub_folders.split("|") 

image_size = 512
image_size_half = int(image_size / 2)
body_size = 300
body_size_half = body_size / 2
body_part_size = 64.
body_part_size_half = body_part_size / 2
scaling_factor = 0.54 # default: 0.5, but usually a bit larger to close gaps between body parts

lighting = False
batch_processing = False
depth_mapping = False
export_point_cloud = False
render_video = False

if mode == "parts":
    perspective_projection = False
    lighting = True
    texture_mapping = None
elif mode == "depth":
    perspective_projection = False
    depth_mapping = True
    batch_processing = True
    texture_mapping = None
elif mode == "render":
    perspective_projection = True
    texture_mapping = 4
elif mode == "points":
    perspective_projection = False
    texture_mapping = 4
    export_point_cloud = True
elif mode == "video":
    perspective_projection = True
    texture_mapping = 4
    render_video = True


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
    'eyes': 20
}

# source: https://stackoverflow.com/questions/66180706/ply-file-format-what-is-the-correct-header-for-the-point-cloud-with-color-rgb
ply_header = """ply
format ascii 1.0
element vertex %s
property uint16 x
property uint16 y
property uint16 z
property uint16 red
property uint16 green
property uint16 blue
end_header
"""


BONES = {
    "torso": ["right_shoulder", "left_shoulder", "left_hip", "right_hip"],
    "head": ["head", "eyes"],
    "right_upper_arm": ["right_shoulder", "right_elbow"],
    "right_lower_arm": ["right_elbow", "right_wrist"],
    "right_hand": ["right_wrist", "right_middle1"],
    "right_upper_leg": ["right_hip", "right_knee"],
    "right_lower_leg": ["right_knee", "right_ankle"],
    "right_foot": ["right_ankle", "right_foot"],
    "left_upper_leg": ["left_hip", "left_knee"],
    "left_lower_leg": ["left_knee", "left_ankle"],
    "left_foot": ["left_ankle", "left_foot"],
    "left_upper_arm": ["left_shoulder", "left_elbow"],
    "left_lower_arm": ["left_elbow", "left_wrist"],
    "left_hand": ["left_wrist", "left_middle1"]
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



@cuda.jit(device=True)
def shapes(point, sdf_objects, object_translations, scales, skeleton_points):
    [x, y, z] = point
    distance = 10000.
    min_distance = 10000.
    min_index = None
    
    for i in range(14):        
        sdf_object = sdf_objects[i]
        [xt, yt, zt] = object_translations[i]
        scale = scales[i]
        
        point = translate((x, y, z), (xt, yt, zt))

        [xp, yp, zp] = point
        
        scale_half = 0.5 * scale
        bounding_cube_size = (body_part_size_half - 2) * scale / body_part_size

        gx = xp / scale_half * body_part_size_half + body_part_size_half
        gy = yp / scale_half * body_part_size_half + body_part_size_half
        gz = zp / scale_half * body_part_size_half + body_part_size_half
        x1 = int(gx)
        y1 = int(gy)
        z1 = int(gz)
        
        # tri-linear interpolation of SDF grid values
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
            distance2 = distance2 / body_size_half * scale * 2
        else :
            distance2 = cuboid(point, (bounding_cube_size, bounding_cube_size, bounding_cube_size))
          
        distance = union(distance, distance2)
        min_distance = union(min_distance, distance2) 
        # min_distance = union_smooth(min_distance, distance2, 2.)
        
        if (distance == distance2):
            min_index = i
    
    """
    # uncomment to show skeleton points
    for i in range(14):
        [xt, yt, zt] = object_translations[i]
        # [xt, yt, zt] = skeleton_points[i]
        point = translate((x, y, z), (xt, yt, zt)) 
        
        distance2 = sphere(point, 0.1)
        min_distance = union(min_distance, distance2)
        
        if (distance2 == min_distance):
            min_index = 1
    """
    return min_distance, min_index

# adapted from http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/#surface-normals-and-lighting
@cuda.jit(device=True)
def estimate_normal(ray_point, sdf_objects, object_translations, scales, skeleton_points):
    EPSILON = 1
    [x, y, z] = ray_point
    
    vector = (
        shapes((x + EPSILON, y, z), sdf_objects, object_translations, scales, skeleton_points)[0] - 
        shapes((x - EPSILON, y, z), sdf_objects, object_translations, scales, skeleton_points)[0],
        shapes((x, y + EPSILON, z), sdf_objects, object_translations, scales, skeleton_points)[0] - 
        shapes((x, y - EPSILON, z), sdf_objects, object_translations, scales, skeleton_points)[0],
        shapes((x, y, z + EPSILON), sdf_objects, object_translations, scales, skeleton_points)[0] - 
        shapes((x, y, z - EPSILON), sdf_objects, object_translations, scales, skeleton_points)[0]
    )
    normal = normalize2(vector)
    
    return normal

@cuda.jit(device=True)
def rotate_2d(x, y, angle):
    rot_x = math.cos(angle) * x - math.sin(angle) * y
    rot_y = math.sin(angle) * x + math.cos(angle) * y
    
    return (rot_x, rot_y)


@cuda.jit()
def my_kernel(result, result_3d, camera_position, camera_forward, camera_up, camera_right, camera_distance, 
              sdf_objects, object_translations, scales, skeleton_points, textures, adjusted_body_size):
    pos = cuda.grid(1)
    
    if (pos < result.size):
        i = int(math.floor(pos / image_size))
        j = pos - i * image_size
        
        # normalize pixel coordinates
        y = (j - image_size_half) / (image_size - 1)
        x = (i - image_size_half) / (image_size - 1)
        
        if perspective_projection: 
            # ray_direction = camera_forward + camera_right * x + camera_up * y
            ray_direction = add2(camera_forward, add2(mult2(camera_right, x), mult2(camera_up, y)))
            ray_direction = normalize2(ray_direction)
        else :
            pixel_position = add2(camera_position, add2(mult2(camera_right, x * adjusted_body_size), mult2(camera_up, y * adjusted_body_size)))
            ray_direction = camera_forward
        
        stepped_distance = 0.0
        
        while(True):
            if perspective_projection:
                # ray_point = camera_position + ray_direction * stepped_distance
                ray_point = add2(camera_position, mult2(ray_direction, stepped_distance))
            else :  
                ray_point = add2(pixel_position, mult2(ray_direction, stepped_distance))
                
            distance, min_index = shapes(ray_point, sdf_objects, object_translations, scales, skeleton_points)
            
            if (distance < 0.5):
                [x_hit, y_hit, z_hit] = ray_point
                
                surface_normal = estimate_normal(ray_point, sdf_objects, object_translations, scales, skeleton_points)
                
                if lighting:
                    light_intensity = abs(dot2(surface_normal, ray_direction)) 
                else :
                    light_intensity = 1
                
                [x_n, _, z_n] = surface_normal
                
                # needed for depth calculation
                if depth_mapping:
                    [x_n, _, z_n] = camera_position
                    
                # calculate angle of the normal in xz-plane
                if (texture_mapping == 4 or depth_mapping):
                    angle = math.atan2(x_n, z_n)
                    angle = angle % (2*math.pi)
     
                elif (texture_mapping == 2):
                    # bi-variate mapping
                    if (z_n > 0):
                        texture_index = 0
                        angle = 0
                    else :
                        texture_index = 2
                        angle = math.pi
                
                if (texture_mapping != None):
                    rgb = (0, 0, 0)
                    
                    distinct_angles = (0., math.pi/2, math.pi, 3*math.pi/2)
                    factors = (
                        2 * abs(angle/math.pi - 1) - 1, 
                        -2 * abs(angle/math.pi - 0.5) + 1, 
                        -2 * abs(angle/math.pi - 1.) + 1, 
                        -2 * abs(angle/math.pi - 1.5) + 1
                    )
                    
                    for texture_index in range(4):
                        texture_factor = factors[texture_index]
                        
                        if (texture_factor < 0):
                            continue
                        
                        distinct_angle = distinct_angles[texture_index]
                        
                        # texture blending
                        u = x_hit * math.cos(distinct_angle) - math.sin(distinct_angle) * z_hit
                        v = y_hit
                        [rt, gt, bt, a] = textures[texture_index][int(u + body_size_half)][int(v + body_size_half)]
        
                        # use generated front texture if original one does not provide a value
                        if (a == 0):
                            [rt, gt, bt, a] = textures[4][int(u + body_size_half)][int(v + body_size_half)]
                        
                        rgb_texture = (rt, gt, bt)
                         
                        rgb = add2(rgb, mult2(rgb_texture, texture_factor))
                    
                else :
                    # body part colors
                    rgb = RGB_COLORS[min_index]

                if depth_mapping:
                    u = z_hit * math.cos(angle) + math.sin(angle) * x_hit
                    # scale to 0...1
                    depth = ((u / body_size * 0.5) + 0.5) 
                    rgb = (depth, (min_index + 1) / 255., 0)
                    # [r, g, b] = (depth, depth, depth) # for debugging

                result[image_size - 1 - j][i][0] = rgb[0] * 255 * light_intensity 
                result[image_size - 1 - j][i][1] = rgb[1] * 255 * light_intensity
                result[image_size - 1 - j][i][2] = rgb[2] * 255 * light_intensity
                
                adjusted_body_size_half = adjusted_body_size / 2
                x_hit_int = int((x_hit + adjusted_body_size_half)) 
                y_hit_int = int((y_hit + adjusted_body_size_half)) 
                z_hit_int = int((z_hit + adjusted_body_size_half)) 
                
                if (x_hit_int > 0 and y_hit_int > 0 and z_hit_int > 0 and 
                    x_hit_int < adjusted_body_size and y_hit_int < adjusted_body_size and z_hit_int < adjusted_body_size):                  
                    result_3d[x_hit_int][y_hit_int][z_hit_int][0] = rgb[0] * 255 * light_intensity 
                    result_3d[x_hit_int][y_hit_int][z_hit_int][1] = rgb[1] * 255 * light_intensity
                    result_3d[x_hit_int][y_hit_int][z_hit_int][2] = rgb[2] * 255 * light_intensity
                
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
    camera_position = [0, 0, camera_distance + 100]
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


# calculate an average value based on the surrounding values
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


# source: https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def get_bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, cmin, rmax, cmax


def write_xyzrgb(file_path, a):
    # source: https://stackoverflow.com/questions/9360103/how-to-print-a-numpy-array-without-brackets
    np.savetxt(file_path, a, fmt="%i")


def xyzrgb_to_x3d(xyzrgb_file_path, x3d_file_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(xyzrgb_file_path, strformat="X Y Z R G B", separator="SPACE", rgbmode="[0-255]")
    ms.compute_normal_for_point_clouds(k=20)
    ms.generate_surface_reconstruction_ball_pivoting()
    ms.meshing_remove_unreferenced_vertices()
    ms.save_current_mesh(x3d_file_path)


def write_ply(file_path, a):
    vertex_count = a.shape[0]
    write_xyzrgb(file_path, a)

    with open(file_path, "r+") as ply_file:
        content = ply_file.read()
        ply_file.seek(0, 0)
        
        ply_file.write(ply_header % vertex_count)
        ply_file.write(content)

def show_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([pcd])


def raymarch(root_folder):    
    skeleton = json.load(open(os.path.join(root_folder, "skeleton_front.json")))

    object_translations = []
    sdf_objects = []
    scales = []
    textures = []
    skeleton_points = []
    bbox = [0, 0, 0, 0] # min_x, min_y, max_x, max_y
    
    for point in skeleton.values():
        point = np.array(point)
        point[1] = 600 - point[1]
        point = point - 300
        skeleton_points.append(point / (600 / body_size))
    
    for bone_name, keypoints in BONES.items():
        pose_points = []
        
        for keypoint in keypoints:
            keypoint_index = POSE_POINTS[keypoint]
            pose_point = skeleton[str(keypoint_index)]
            pose_points.append(pose_point)
        
        mid_point = np.average(pose_points, axis=0)
        
        # special case for head
        if (bone_name == "head"):
            mid_point[0] = pose_points[0][0]
            mid_point[1] = pose_points[0][1]
            mid_point[2] = pose_points[0][2]        
        
        # invert y-axis    
        mid_point[1] = 600 - mid_point[1]
        # set origin from (300, 300, 300) to (0, 0, 0)
        centred_mid_point = mid_point - 300
        object_translations.append(centred_mid_point / (600 / body_size))
        
        object_path = os.path.join(root_folder, "%s.npy" % bone_name)
        
        if (not os.path.exists(object_path)):
            continue
        
        sdf_object = np.load(object_path)
        sdf_objects.append(sdf_object)
        
        body_part_metadata = json.load(open(os.path.join(root_folder, "%s.json" % bone_name)))
        scale = body_part_metadata["scale"] * scaling_factor
        scales.append(scale) 
        
        # calculate bounding box to fit the generated figure into the image
        sdf_object_2d = np.min(sdf_object, axis=2)
        sdf_object_2d_binary = 1 - np.heaviside(sdf_object_2d, 0)
        sdf_object_2d_bbox = get_bbox(sdf_object_2d_binary)
        sdf_object_2d_bbox = [sdf_object_2d_bbox[0], body_part_size - sdf_object_2d_bbox[3] - 1, 
                              sdf_object_2d_bbox[2], body_part_size - sdf_object_2d_bbox[1] - 1]
        
        sdf_object_2d_bbox_scaled = (np.array(sdf_object_2d_bbox) - body_part_size_half) / body_part_size_half * scale
        
        bbox = [
            min(mid_point[0] - body_size + sdf_object_2d_bbox_scaled[0], bbox[0]),
            min(mid_point[1] - body_size - sdf_object_2d_bbox_scaled[3], bbox[1]),
            max(mid_point[0] - body_size + sdf_object_2d_bbox_scaled[2], bbox[2]),
            max(mid_point[1] - body_size - sdf_object_2d_bbox_scaled[1], bbox[3])
        ]    

    # 2px padding to not cut off soles of foot and parts of the hair 
    # when exporting point cloud
    adjusted_body_size = max(body_size, -bbox[0], -bbox[1], bbox[2], bbox[3]) + 2
    adjusted_body_size = math.ceil(adjusted_body_size)
    
    for view in ["front_cropped", "left", "back", "right", "front"]:
        texture_path = os.path.join(root_folder, "body_parts_%s_texture.png" % view)
        
        if (not os.path.exists(texture_path)):
            continue
        
        texture = Image.open(texture_path)
        texture = texture.resize((body_size, body_size), Image.NEAREST)
        texture = np.asarray(texture) / 255.
        texture = np.swapaxes(texture, 0, 1)
        texture = np.flip(texture, axis=1)
        textures.append(texture)    
    
    sdf_objects = np.array(sdf_objects)
    object_translations = np.array(object_translations)
    skeleton_points = np.array(skeleton_points)
    scales = np.array(scales)
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
    px = 1/plt.rcParams['figure.dpi']
    ax.figure.set_size_inches(image_size*px, image_size*px)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    canvas.mpl_connect("button_press_event", on_press)
    canvas.mpl_connect("button_release_event", on_release)
    canvas.mpl_connect("motion_notify_event", on_move)
    canvas.mpl_connect("scroll_event", on_scroll)
    canvas.mpl_connect("close_event", on_close)
    
    result = np.zeros((image_size, image_size, 3), dtype=np.uint16)
    result_3d = np.full((adjusted_body_size, adjusted_body_size, adjusted_body_size, 3), 256, dtype=np.uint16)
    indices_3d = np.indices((adjusted_body_size, adjusted_body_size, adjusted_body_size), dtype=np.uint16).transpose(1, 2, 3, 0)
    
    im = plt.imshow(result)
    
    rotation_angle = 0
    rotation_angle2 = -45
    
    if (render_video):
        video_folder = os.path.join(root_folder, "video")
        
        if not os.path.exists(video_folder):
            os.mkdir(video_folder)
    
    if not batch_processing:
        # rendering loop
        while(render_loop):
            # start_time = time.time()
            result = np.ones((image_size, image_size, 3), dtype=np.uint16) * 255 # white background
            # result = np.zeros((image_size, image_size, 3), dtype=np.uint16) # black background
            my_kernel[image_size, image_size](result, result_3d, camera_position, camera_forward, 
                                              camera_up, camera_right, camera_distance, 
                                              sdf_objects, object_translations, scales, 
                                              skeleton_points, textures, adjusted_body_size)
            
            im.set_data(result)
               
            canvas.draw()
            canvas.flush_events()
            
            # print(time.time() - start_time)

            if export_point_cloud:
                camera_coords[0] = rotation_angle*math.pi/180
                camera_coords[1] = rotation_angle2*math.pi/180
                pan()
     
                
                if (rotation_angle == 355):
                    rotation_angle = 0
                    rotation_angle2 += 45
                    
                    if (rotation_angle2 == 90):
                        break
                else :  
                    rotation_angle += 5
                
            
            if render_video:
                # render images for video
                keyframe_img = Image.fromarray(np.uint8(result))
                keyframe_img.save(os.path.join(video_folder, "img%03d.png" % rotation_angle))
                
                rotation_angle += 1
                camera_coords[0] = rotation_angle*math.pi/180
                pan()
                
                if rotation_angle == 360:
                    break

        
        if export_point_cloud:
            hits_3d = np.logical_not(np.all(result_3d == 256, axis = 3))
            result_3d_filtered = result_3d[hits_3d]
            indices_3d_filtered = indices_3d[hits_3d]
            
            result_with_indices = np.concatenate([indices_3d_filtered, result_3d_filtered], axis=1)
            
            xyzrgb_file_path = os.path.join(root_folder, "point_cloud.txt")
            x3d_file_path = os.path.join(root_folder, "mesh.x3d")
            
            write_xyzrgb(xyzrgb_file_path, result_with_indices)
            xyzrgb_to_x3d(xyzrgb_file_path, x3d_file_path)
            
            # uncomment to visualize the point cloud immediately
            # ply_file_path = os.path.join(root_folder, "point_cloud.ply")
            # write_ply(ply_file_path, result_with_indices)
            # show_ply(ply_file_path)
            
        if render_video:
            subprocess.run([r"C:\Program Files\FFmpeg\bin\ffmpeg", "-i", "img%03d.png", "-framerate", 
                            "24", "-y", "-c:v" , "libx264", "-pix_fmt", "yuv420p", video_folder + ".mp4"], cwd=video_folder)

        
    else :
        CAMERA_POSITIONS = {
            "front": 0,
            "left": math.pi/2,
            "back": math.pi,
            "right": 3 * math.pi / 2
        }
        
        import tensorflow as tf
        
        def resize_array(arr, new_size):
            num_dims = len(arr.shape)
            
            if (num_dims == 2): # width x height
                arr = arr[tf.newaxis, ..., tf.newaxis]
            elif (num_dims == 3): # width x height x channels
                arr = arr[tf.newaxis, ...]
            # else batch x width x height x channels    
            
            tensor = tf.constant(arr)
            resized_tensor = tf.image.resize(tensor, new_size)
            resized_arr = resized_tensor.numpy()
            
            if (num_dims == 2):
                result = resized_arr[0, ..., 0]
            elif (num_dims == 3):
                result = resized_arr[0, ...]
                         
            return result
        
        for view in ["front", "left", "back", "right"]:
            camera_coords[0] = CAMERA_POSITIONS[view]
            pan()
        
            result = np.zeros((image_size, image_size, 3), dtype=np.uint16)
            my_kernel[image_size, image_size](result, result_3d, camera_position, camera_forward, 
                                              camera_up, camera_right, camera_distance, 
                                              sdf_objects, object_translations, scales, 
                                              skeleton_points, textures, adjusted_body_size)
            
            depth_channel = result[:, :, 0]
            is_zero = depth_channel == 0
            
            depth_map = (1 - depth_channel / 255.) - 0.5
            
            depth_map[is_zero] = np.nan
        
            resized_depth_map = resize_array(depth_map, (256, 256))
            
            padded_depth_map = np.pad(resized_depth_map, ((1, 1), (1, 1)), 'constant', constant_values=np.nan)
            
            filled_depth_map = np.full_like(resized_depth_map, np.nan)
            
            # pads the silhouette by 2px
            for x in range(resized_depth_map.shape[0]):
                for y in range(resized_depth_map.shape[1]):
                    if (np.isnan(resized_depth_map[x][y])):
                        surrounding_values = np.array([
                            padded_depth_map[x][y],
                            padded_depth_map[x + 1][y],
                            padded_depth_map[x + 2][y],
                            padded_depth_map[x][y + 1],
                            padded_depth_map[x + 2][y + 1],
                            padded_depth_map[x][y + 2],
                            padded_depth_map[x + 1][y + 2],
                            padded_depth_map[x + 2][y + 2]
                        ])
                        
                        filled_depth_map[x][y] = np.nanmean(surrounding_values)
                    else :
                        filled_depth_map[x][y] = resized_depth_map[x][y]
                        
            filled_depth_map = smooth(filled_depth_map, 5)
    
            filled_depth_map = np.flip(filled_depth_map, axis=0)
            filled_depth_map = np.swapaxes(filled_depth_map, 0, 1)
            np.savez_compressed(os.path.join(root_folder, "depth_%s" % view), filled_depth_map)
    
            body_part_channel = result[:, :, 1] - 1
            body_part_channel[body_part_channel == -1] = 255.
            
            body_part_mask = np.stack([body_part_channel, body_part_channel, body_part_channel], axis=-1)
            
            body_part_mask_image = Image.fromarray(np.uint8(body_part_mask))
            body_part_mask_image.save(os.path.join(root_folder, "body_parts_%s_mask.png" % view))
      

if __name__ == '__main__':        
    for subfolder_name in subfolder_names:
        subfolder_path = os.path.join(root_folder, subfolder_name)
        print(subfolder_path)
        
        raymarch(subfolder_path)
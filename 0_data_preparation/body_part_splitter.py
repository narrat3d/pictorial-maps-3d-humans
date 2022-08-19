'''
splits 3D mesh from SMPL-X humans into individual 3D body parts (*.obj), 
smoothes the boundary edges, derives body part mask images, UV images, depth images
for 4 perspectives (front, right, back, left). additionally, textures are mapped 
(but not used later-on). extracts 3D skeleton points from the bones.
'''

"""
# uncomment for setting breakpoints in PyDev for Eclipse
import sys
PYDEV_SOURCE_DIR = 'C:/Users/raimund/.p2/pool/plugins/org.python.pydev_6.2.0.201711281614/pysrc'

if PYDEV_SOURCE_DIR not in sys.path:
    sys.path.append(PYDEV_SOURCE_DIR)
    
import pydevd
pydevd.settrace()
"""

import sys
import os
import math
import json
from collections import Counter
import numpy as np

import bpy, bmesh
from mathutils import Color, Vector

argv = sys.argv


# remove previous objects (except Camera)
objs = bpy.data.objects

for obj in objs:
    obj_name = obj.name
    
    if (obj_name == "Camera"):
        continue
    
    objs.remove(objs[obj_name])
    

try :
    # get all args after "--"
    additional_argument_index = argv.index("--")
    arg = argv[additional_argument_index + 1:]  
    
    output_folder = arg[0]
    gender = arg[1]
    height = float(arg[2])
    weight = float(arg[3])
    pose_path = arg[4]
    texture_path = arg[5]
    
except ValueError:
    output_folder = r"E:\CNN\implicit_functions\smpl-x\10031_m_John_0_0_female_medium"
    gender = "female"
    height = 1.80
    weight = 75
    pose_path = r"E:\Data\SMPL\smplx_gt\trainset_3dpeople_adults_bfh\10031_m_John_0_0.pkl"
    texture_path = r"E:\Data\SMPL\SURREAL\textures\male\nongrey_male_0925.jpg"


BODY_PARTS = {
    "torso": 0,
    "head": 1,
    "right_upper_arm": 2,
    "right_lower_arm": 3,
    "right_hand": 4,
    "right_upper_leg": 5,
    "right_lower_leg": 6,
    "right_foot": 7,
    "left_upper_leg": 8,
    "left_lower_leg": 9,
    "left_foot": 10,
    "left_upper_arm": 11,
    "left_lower_arm": 12,
    "left_hand": 13 
}

BODY_PART_NRS = list(BODY_PARTS.values())

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
    'right_middle1': 16,
    'right_foot': 17,
    'left_foot': 18,
    'left_middle1': 19
}
    
RGB_COLORS = [
    [0.25, 0.25, 0.25],
    [0.75, 0.75, 0.75],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.5, 0.5, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.5],
]

SMPLX_BODY_PART_MAPPING = {
    0: BODY_PARTS["torso"], # pelvis
    1: BODY_PARTS["left_upper_leg"], # left_hip
    2: BODY_PARTS["left_lower_leg"], # left_knee
    3: BODY_PARTS["left_foot"], # left_ankle
    4: BODY_PARTS["left_foot"], # left_foot
    5: BODY_PARTS["right_upper_leg"], # right_hip
    6: BODY_PARTS["right_lower_leg"], # right_knee
    7: BODY_PARTS["right_foot"], # right_ankle
    8: BODY_PARTS["right_foot"], # right_foot
    9: BODY_PARTS["torso"], # spine1
    10: BODY_PARTS["torso"], # spine2
    11: BODY_PARTS["torso"], # spine3
    12: BODY_PARTS["torso"], # neck
    13: BODY_PARTS["head"], # head
    14: BODY_PARTS["head"], # jaw
    15: BODY_PARTS["head"], # left_eye_smplhf
    16: BODY_PARTS["head"], # right_eye_smplhf
    17: BODY_PARTS["torso"], # left_collar
    18: BODY_PARTS["left_upper_arm"], # left_shoulder
    19: BODY_PARTS["left_lower_arm"], # left_elbow
    20: BODY_PARTS["left_hand"], # left_wrist
    21: BODY_PARTS["left_hand"], # left_index1
    22: BODY_PARTS["left_hand"], # left_index2
    23: BODY_PARTS["left_hand"], # left_index3
    24: BODY_PARTS["left_hand"], # left_middle1
    25: BODY_PARTS["left_hand"], # left_middle2
    26: BODY_PARTS["left_hand"], # left_middle3
    27: BODY_PARTS["left_hand"], # left_pinky1
    28: BODY_PARTS["left_hand"], # left_pinky2
    29: BODY_PARTS["left_hand"], # left_pinky3
    30: BODY_PARTS["left_hand"], # left_ring1
    31: BODY_PARTS["left_hand"], # left_ring2
    32: BODY_PARTS["left_hand"], # left_ring3
    33: BODY_PARTS["left_hand"], # left_thumb1
    34: BODY_PARTS["left_hand"], # left_thumb2
    35: BODY_PARTS["left_hand"], # left_thumb3
    36: BODY_PARTS["torso"], # right_collar
    37: BODY_PARTS["right_upper_arm"], # right_shoulder
    38: BODY_PARTS["right_lower_arm"], # right_elbow
    39: BODY_PARTS["right_hand"], # right_wrist
    40: BODY_PARTS["right_hand"], # right_index1
    41: BODY_PARTS["right_hand"], # right_index2
    42: BODY_PARTS["right_hand"], # right_index3
    43: BODY_PARTS["right_hand"], # right_middle1
    44: BODY_PARTS["right_hand"], # right_middle2
    45: BODY_PARTS["right_hand"], # right_middle3
    46: BODY_PARTS["right_hand"], # right_pinky1
    47: BODY_PARTS["right_hand"], # right_pinky2
    48: BODY_PARTS["right_hand"], # right_pinky3
    49: BODY_PARTS["right_hand"], # right_ring1
    50: BODY_PARTS["right_hand"], # right_ring2
    51: BODY_PARTS["right_hand"], # right_ring3
    52: BODY_PARTS["right_hand"], # right_thumb1
    53: BODY_PARTS["right_hand"], #  right_thumb2
    54: BODY_PARTS["right_hand"], # right_thumb3
}

# adapted from the SMPL-X plugin
def set_texture(texture_path, obj):
    mat = obj.data.materials[0]
    links = mat.node_tree.links
    nodes = mat.node_tree.nodes

    # Find texture node
    node_texture = None
    for node in nodes:
        if node.type == 'TEX_IMAGE':
            node_texture = node
            break

    # Find shader node
    node_shader = None
    for node in nodes:
        if node.type.startswith('BSDF'):
            node_shader = node
            break

    if node_texture is None:
        node_texture = nodes.new(type="ShaderNodeTexImage")

    image = bpy.data.images.load(texture_path)
    node_texture.image = image

    # Link texture node to shader node if not already linked
    if len(node_texture.outputs[0].links) == 0:
        links.new(node_texture.outputs[0], node_shader.inputs[0])

    # Switch viewport shading to Material Preview to show texture
    if bpy.context.space_data:
        if bpy.context.space_data.type == 'VIEW_3D':
            bpy.context.space_data.shading.type = 'MATERIAL'
            
    return node_texture


# set cursor to origin
# source: https://blender.stackexchange.com/questions/5359/how-to-set-cursor-location-pivot-point-in-script
bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)

smplx_tool = bpy.data.window_managers["WinMan"].smplx_tool

smplx_tool.smplx_gender = gender
bpy.ops.scene.smplx_add_gender()

if (pose_path != None):
    bpy.ops.object.smplx_load_pose(filepath = pose_path)

if (gender != "neutral"):
    smplx_tool.smplx_height = height
    smplx_tool.smplx_weight = weight
    bpy.ops.object.smplx_measurements_to_shape()


root_obj = bpy.context.scene.objects.get('SMPLX-%s' % gender)
obj = bpy.context.scene.objects.get('SMPLX-mesh-%s' % gender)

texture_node = set_texture(texture_path, obj)

# recenter object
bound_box = obj.bound_box
[x_min, y_min, z_min] = bound_box[0]
[x_max, y_max, z_max] = bound_box[6]

x_offset = x_max + x_min
y_offset = y_max + y_min
z_offset = z_max + z_min

root_obj.location.x = -x_offset/2
root_obj.location.y = -y_offset/2
root_obj.location.z = -z_offset/2

bpy.context.view_layer.objects.active = root_obj
root_obj.select_set(True)
bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

# https://blender.stackexchange.com/questions/6101/poll-failed-context-incorrect-example-bpy-ops-view3d-background-image-add?lq=1
"""
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        override = bpy.context.copy()
        override['area'] = area
        bpy.ops.view3d.snap_selected_to_cursor(override, use_offset=False)
        break
"""

body_width = obj.dimensions.x
body_depth = obj.dimensions.y
body_height = obj.dimensions.z

body_size = max(body_width, body_height)
body_size_half = body_size / 2

# create material shader
# https://www.youtube.com/watch?v=eo7UjKFiagk
# https://blender.stackexchange.com/questions/8108/node-for-setting-render-colour-regardless-of-lighting
# https://docs.blender.org/api/current/bpy.types.html
material = obj.active_material

material_nodes = material.node_tree.nodes

light_path_node = material_nodes.new("ShaderNodeLightPath")
vertex_color_node = material_nodes.new("ShaderNodeVertexColor")
emission_node = material_nodes.new("ShaderNodeEmission")
mix_shader_node = material_nodes.new("ShaderNodeMixShader")
material_output_node = material_nodes.get("Material Output")
bsdf_node = material_nodes.get("Principled BSDF")


light_path_node.location = (-1000, 0)
vertex_color_node.location = (-800, 0)
emission_node.location = (-600, 0)
mix_shader_node.location = (-400, 0)
material_output_node.location = (-200, 0)

material.node_tree.links.new(vertex_color_node.outputs[0], emission_node.inputs[0])
material.node_tree.links.new(light_path_node.outputs[0], mix_shader_node.inputs[0])
material.node_tree.links.new(emission_node.outputs[0], mix_shader_node.inputs[2])


# otherwise colors look grayish
bpy.context.scene.view_settings.view_transform = 'Standard'

# sources: https://blender.stackexchange.com/questions/5364/how-to-apply-shape-keys +
# https://blender.stackexchange.com/questions/49768/can-i-set-my-pose-position-like-the-new-rest-position
combined_key = obj.shape_key_add(from_mix=True)

for shape_key in obj.data.shape_keys.key_blocks:
    obj.shape_key_remove(shape_key)

bpy.ops.object.modifier_apply(modifier="Armature")

mesh = obj.data

vertex_group_names = {}
vertex_groups = {}  
edge_groups = {}
    
for vertex_group in obj.vertex_groups:
    vertex_group_names[vertex_group.index] = vertex_group.name


# assign body part groups to vertices
for index, vertex in enumerate(mesh.vertices):
    max_weight = -1
    max_group = -1
    
    for vertex_group in vertex.groups:
        if (vertex_group.weight > max_weight):
            max_weight = vertex_group.weight
            max_group = vertex_group.group
    
    mapped_group = SMPLX_BODY_PART_MAPPING[max_group]
    vertex_groups[index] = mapped_group


# assign edges of polygons to different body part groups
# need string comparison since meshes will be duplicated later on
for polygon in mesh.polygons:
    groups = []
    
    for vertex_index in polygon.vertices:  
        vertex_group = vertex_groups[vertex_index]
        groups.append(vertex_group)

    c = Counter(groups) 

    [most_common_group, count] = c.most_common(1)[0]

    if (count >= 2):
        vertex0 = mesh.vertices[polygon.vertices[0]]
        vertex1 = mesh.vertices[polygon.vertices[1]]
        vertex2 = mesh.vertices[polygon.vertices[2]]
        
        vertex0_coords = str(vertex0.co)[9:-2].replace(" ", "")
        vertex1_coords = str(vertex1.co)[9:-2].replace(" ", "")
        vertex2_coords = str(vertex2.co)[9:-2].replace(" ", "")
        
        edges = edge_groups.setdefault(most_common_group, [])
        edges.append(",".join(sorted([vertex0_coords, vertex1_coords])))
        edges.append(",".join(sorted([vertex0_coords, vertex2_coords])))
        edges.append(",".join(sorted([vertex1_coords, vertex2_coords])))

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_mode(type="FACE")
bpy.ops.object.mode_set(mode='OBJECT')


spikes_to_add_for_group = {}

for group in BODY_PART_NRS:
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')    

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # select polygons of a body parts based on the vertex groups
    for polygon in mesh.polygons:
        groups = []
        
        for vertex_index in polygon.vertices:  
            vertex_group = vertex_groups[vertex_index]
            groups.append(vertex_group)
    
        c = Counter(groups) 
       
        [most_common_group, count] = c.most_common(1)[0]
        
        if (most_common_group == group):
            polygon.select = True 
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.duplicate()
    bpy.ops.mesh.separate(type='SELECTED')
    
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')

    body_part_obj = bpy.context.scene.objects.get('SMPLX-mesh-%s.0%02d' % (gender, group + 1))
        
    bpy.context.view_layer.objects.active = body_part_obj
    body_part_obj.select_set(True)
    
    bpy.ops.object.mode_set(mode='EDIT')
    body_part_mesh = bmesh.from_edit_mesh(body_part_obj.data)
    
    # find the boundary edges to other body parts by checking if the 
    # edges are included in another body part
    for edge_group in BODY_PART_NRS:
        if (edge_group == group):
            continue
        
        edges = edge_groups[edge_group]
        
        boundary_edges_by_vertex = {}   
        
        # store the two boundary edges per boundary vertex
        # 
        for face in body_part_mesh.faces:
            for edge in face.edges:
                vertex0 = edge.verts[0]
                vertex0_index = vertex0.index
                vertex1 = edge.verts[1]
                vertex1_index = vertex1.index
            
                vertex0_coords = str(vertex0.co)[9:-2].replace(" ", "")
                vertex1_coords = str(vertex1.co)[9:-2].replace(" ", "")
            
                edge_str = ",".join(sorted([vertex0_coords, vertex1_coords]))

                if (edge_str in edges):   
                    edges_by_vertex = boundary_edges_by_vertex.setdefault(vertex0_index, [])
                    edges_by_vertex.append(edge)
                    
                    edges_by_vertex = boundary_edges_by_vertex.setdefault(vertex1_index, [])
                    edges_by_vertex.append(edge)
        
        boundary_vertices_num = len(boundary_edges_by_vertex.keys())
    
        if (boundary_vertices_num > 0):
            # iterate through the boundary loop by following the end vertex of each edge
            # which is connected to the next edge
            
            # choose an edge (and its vertices) to begin with
            # preferably take a loose edge (this is the case for 
            # torso - left/right upper leg)
            for vertex_index, edges in boundary_edges_by_vertex.items():
                start_vertex_index = vertex_index
                
                if (len(edges) == 1):        
                    break
            
            current_vertex_index = start_vertex_index
            current_edge = boundary_edges_by_vertex[start_vertex_index][0]
            current_vertices = [current_edge.verts[0], current_edge.verts[1]]
            
            # ensure correct vertex order for edge
            if (current_vertices[0].index != current_vertex_index):
                current_vertices = [current_vertices[1], current_vertices[0]]
            
            while (True):                                      
                next_vertex_index = current_vertices[1].index
                possible_next_edges = boundary_edges_by_vertex[next_vertex_index]
                next_edge = possible_next_edges[0]
                
                current_edge.select = True
                
                if (current_edge == next_edge):
                    if (len(possible_next_edges) == 1):
                        # loose end reached 
                        break
                    
                    next_edge = possible_next_edges[1]
                                    
                next_vertices = [next_edge.verts[0], next_edge.verts[1]]
            
                # ensure correct vertex order for edge
                if (next_vertices[0].index != next_vertex_index):
                    next_vertices = [next_vertices[1], next_vertices[0]]
                                                
                adjacent_faces = current_vertices[1].link_faces
                
                # suitable edges for subdivision:
                # a\ triangles where one vertex is not connected to any other edges
                if (len(adjacent_faces) == 1):
                    current_edge.select = True
                    next_edge.select = True
                
                # b\ two triangles which share an edge
                elif (len(adjacent_faces) == 2):
                    adjecent_edges1 = adjacent_faces[0].edges
                    adjecent_edges2 = adjacent_faces[1].edges
                    adjacent_edges_indices1 = set(map(lambda e: e.index, adjecent_edges1))
                    adjacent_edges_indices2 = set(map(lambda e: e.index, adjecent_edges2))
                    
                    adjacent_edges_indices = adjacent_edges_indices1.intersection(adjacent_edges_indices2)
                    
                    if (len(adjacent_edges_indices) == 1):
                        shared_edge_index = adjacent_edges_indices.pop()
                        shared_edge = list(filter(lambda e: e.index == shared_edge_index, adjecent_edges1))[0]
                        
                        current_edge.select = True
                        shared_edge.select = True
                        next_edge.select = True
                
                current_vertex_index = next_vertex_index
                current_edge = next_edge
                current_vertices = next_vertices
                
                if (current_vertex_index == start_vertex_index):
                    break
            
            # subdivide faces by the selected edges
            bpy.ops.mesh.subdivide()

            # select resulting triangles from the subdivision
            subdivided_faces = [f for f in body_part_mesh.faces if f.select]
            for subdivided_face in subdivided_faces:
                if (len(subdivided_face.verts) == 4):
                    subdivided_face.select = False
                elif (len(subdivided_face.verts) == 3):
                    boundary_face = False
                    
                    # rare case for right upper leg after subdivision
                    for edge in subdivided_face.edges:
                        if (len(edge.link_faces) == 1):
                            boundary_face = True
                            break
                
                    subdivided_face.select = boundary_face
            
            # remove spikes from the body part mesh
            bpy.ops.mesh.separate(type='SELECTED')  
            separated_spikes_obj = set(bpy.context.selected_objects).difference([bpy.context.view_layer.objects.active]).pop()
            separated_spikes_obj.name = "separated_spikes_for_%s" % edge_group
            
            spikes_to_add = spikes_to_add_for_group.setdefault(edge_group, [])
            spikes_to_add.append(separated_spikes_obj)
            separated_spikes_obj.select_set(False)
            
            # convert quads from the subdivision to triangles
            for face in body_part_mesh.faces:
                if (len(face.verts) == 4):
                    face.select = True                
            
            bpy.ops.mesh.quads_convert_to_tris()
            bpy.ops.mesh.select_all(action='DESELECT')


# merge the separated spikes with the corresponding other body part
for group, spike_objects in spikes_to_add_for_group.items():
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    
    body_part_obj = bpy.context.scene.objects.get('SMPLX-mesh-%s.0%02d' % (gender,  group + 1))
    body_part_obj.select_set(True)
    bpy.context.view_layer.objects.active = body_part_obj
    
    for spike_object in spike_objects:
        spike_object.select_set(True)
            
    bpy.ops.object.join()
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    # remove duplicate edges and vertices
    bpy.ops.mesh.remove_doubles()
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.normals_make_consistent(inside=True)
    bpy.ops.mesh.select_all(action='DESELECT')


vertex_loops_for_group = {}

# detect vertices of boundary loops 
# compared to the upper procedure (which is similar), 
# the connecting body part does not matter anymore
for group in BODY_PART_NRS:
    body_part_obj = bpy.context.scene.objects.get('SMPLX-mesh-%s.0%02d' % (gender, group + 1))
    
    bpy.context.view_layer.objects.active = body_part_obj
    body_part_obj.select_set(True)
    
    bpy.ops.object.mode_set(mode='EDIT')
    
    body_part_mesh = bmesh.from_edit_mesh(body_part_obj.data)
    boundary_edges_for_vertex = {}
    
    # store which two* boundary edges are connected to each boundary vertex 
    # * special case: four boundary edges 
    for edge in body_part_mesh.edges:
        if (len(edge.link_faces) == 1):
            vertex0 = edge.verts[0]
            vertex1 = edge.verts[1]
            
            vertex0_index = vertex0.index
            vertex1_index = vertex1.index
            
            boundary_edges = boundary_edges_for_vertex.setdefault(vertex0_index, [])
            boundary_edges.append(edge)
            boundary_edges = boundary_edges_for_vertex.setdefault(vertex1_index, [])
            boundary_edges.append(edge)
    
    vertex_loops = []
    vertex_loops_for_group[group] = vertex_loops
      
    # order the boundary vertices according their position in the loop
    # e.g. torso has four loop, upper arm has two loops, foot has one loop
    while (len(boundary_edges_for_vertex.keys()) > 0):
        vertex_index = list(boundary_edges_for_vertex.keys())[0]
        
        vertices = []
        vertices.append(vertex_index)
                
        first_edge = boundary_edges_for_vertex[vertex_index][0]
        
        current_vertex_index = vertex_index
        current_edge = first_edge
        
        # need to be careful for special case where the attached spike
        # has created a mini-loop consisting of one triangle
        while (True):
            current_edge.select = True
            vertex0 = current_edge.verts[0]
            vertex1 = current_edge.verts[1]
            
            # ensure correct vertex order
            if (current_vertex_index == vertex0.index):
                current_vertex_index = vertex1.index
            else :
                current_vertex_index = vertex0.index
        
            boundary_edges = boundary_edges_for_vertex[current_vertex_index]
            
            # remove current edge
            boundary_edges = list(filter(lambda e: e != current_edge, boundary_edges))
            
            if (len(boundary_edges) == 0):
                # special case: some boundary loops may not be closed
                del boundary_edges_for_vertex[current_vertex_index]
                closed_loop = False
                
                # delete edge of first vertex which led to the second vertex
                boundary_edges = boundary_edges_for_vertex[vertex_index]
                boundary_edges = list(filter(lambda e: e != first_edge, boundary_edges))
                
                # when the first vertex was connected to only one edge, remove this vertex
                if (len(boundary_edges) == 0):
                    del boundary_edges_for_vertex[vertex_index]
                else :
                    boundary_edges_for_vertex[vertex_index] = boundary_edges
                break
            
            current_edge = boundary_edges.pop()
            
            if (len(boundary_edges) == 0):
                del boundary_edges_for_vertex[current_vertex_index]
            else :
                boundary_edges_for_vertex[current_vertex_index] = boundary_edges
            
            # loop closed
            if (current_vertex_index == vertex_index):
                closed_loop = True
                break
            
            vertices.append(current_vertex_index)
        
        if (closed_loop):
            vertex_loops.append(vertices)
    
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.select_all(action='DESELECT')


for group_name, group in BODY_PARTS.items():
    body_part_obj = bpy.context.scene.objects.get('SMPLX-mesh-%s.0%02d' % (gender, group + 1))
    bpy.context.view_layer.objects.active = body_part_obj
    body_part_obj.select_set(True)
    
    bpy.ops.object.mode_set(mode='EDIT')
    body_part_mesh = bmesh.from_edit_mesh(body_part_obj.data)
    body_part_mesh.verts.ensure_lookup_table()
    
    previous_edges = None
    
    # close holes by connecting edges to other body parts with a midpoint vertex
    vertex_loops = vertex_loops_for_group[group]
    xyz_means = []
    mesh_vertices_loops = []
    
    # calculate the mean from all boundary vertices
    for vertices in vertex_loops:
        x_sum = 0
        y_sum = 0
        z_sum = 0
        
        mesh_vertices = []
        
        for vertex_index in vertices:
            vertex = body_part_mesh.verts[vertex_index]
            x_sum += vertex.co.x
            y_sum += vertex.co.y
            z_sum += vertex.co.z
            mesh_vertices.append(vertex)
            
        num_vertices = len(vertices)
        x_mean = x_sum / num_vertices
        y_mean = y_sum / num_vertices
        z_mean = z_sum / num_vertices
        
        xyz_means.append([x_mean, y_mean, z_mean])
        mesh_vertices_loops.append(mesh_vertices)

    # add the mean vertex and connect it with the boundary vertices
    # cannot modify mesh directly, otherwise vertex indices are changed
    for mesh_vertices, xyz_mean in zip(mesh_vertices_loops, xyz_means): 
        [x_mean, y_mean, z_mean] = xyz_mean      
        vertex_mean = body_part_mesh.verts.new((x_mean, y_mean, z_mean))

        for position in range(len(mesh_vertices)):
            vertex0 = mesh_vertices[position]
            vertex1 = mesh_vertices[(position + 1) % len(mesh_vertices)]
            
            try:
                body_part_mesh.faces.new((vertex0, vertex1, vertex_mean))
            except ValueError:
                pass

    bmesh.update_edit_mesh(body_part_obj.data)
                
    # recalculate normals for newly inserted faces
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    
    # remove single edges (= artefacts after subdivision)
    bpy.ops.mesh.delete_loose()
    
    bpy.ops.mesh.select_all(action='DESELECT')
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    
    # source: https://blender.stackexchange.com/questions/909/how-can-i-set-and-get-the-vertex-color-property
    body_part_obj = bpy.context.scene.objects.get('SMPLX-mesh-%s.0%02d' % (gender, group + 1))
    body_part_mesh = body_part_obj.data        
    body_part_mesh.vertex_colors.new()
        
    color_layer = body_part_mesh.vertex_colors["Col"]
    
    [r, g, b] = RGB_COLORS[group]
    
    i = 0
    for poly in body_part_mesh.polygons:
        for idx in poly.loop_indices:
            color_layer.data[i].color = (r, g, b, 1.0)
            i += 1
    
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = body_part_obj
    body_part_obj.select_set(True)
    
    # export .obj files
    body_part_path = os.path.join(output_folder, "%s.obj" % group_name)
    bpy.ops.export_scene.obj(filepath=body_part_path, use_selection=True, axis_forward="-Z", axis_up="Y")



# save rendered file
# https://stackoverflow.com/questions/14982836/rendering-and-saving-images-through-blender-python

image_size = 600
image_size_half = image_size / 2

bpy.context.scene.render.image_settings.file_format='PNG'
bpy.context.scene.render.resolution_x = image_size
bpy.context.scene.render.resolution_y = image_size

camera = bpy.data.objects['Camera']
camera_distance = 10.0
camera_perspectives = [
    ["front", 0, -camera_distance, 0], 
    ["left", camera_distance, 0, math.pi / 2], 
    ["back", 0, camera_distance, math.pi],
    ["right", -camera_distance, 0, 3 * math.pi / 2]
]


camera.data.ortho_scale = body_size

bpy.context.view_layer.objects.active = root_obj
root_obj.select_set(True)
bones = root_obj.pose.bones

bpy.ops.object.mode_set(mode='POSE')

# do not let original mesh interfere with extracted body parts
obj.hide_render = True

for [camera_view, x, y, z_angle] in camera_perspectives:
    camera.location.x = x
    camera.location.y = y    
    camera.rotation_euler[2] = z_angle
    
    for [shading_type, input_node] in [["texture", texture_node], ["colormask", mix_shader_node]]:
        material.node_tree.links.new(input_node.outputs[0], material_output_node.inputs[0])
    
        image_path = os.path.join(output_folder, "body_parts_%s_%s.png" 
                                 % (camera_view, shading_type))
        bpy.context.scene.render.filepath = image_path
        bpy.ops.render.render(use_viewport = True, write_still=True)


    # reclassify mask image
    colormask_path = os.path.join(output_folder, "body_parts_%s_colormask.png" % camera_view)                       
    colormask_image = bpy.data.images.load(colormask_path)

    colormask_image_np = np.array(colormask_image.pixels)
    colormask_image_np = colormask_image_np.reshape(image_size, image_size, 4)
    colormask_image_np = colormask_image_np[:, :, 0:3]
    body_parts_mask_np = np.zeros((image_size, image_size))

    for body_part_index, rgb in enumerate(RGB_COLORS):
        colormask_image_diff_np = colormask_image_np - np.array(rgb)
        colormask_image_diff_np = np.abs(colormask_image_diff_np)
        
        # check if RGB value is within a certain range since
        # there are minor color difference artefacts during rendering
        body_part_mask_np = np.logical_and(np.logical_and(colormask_image_diff_np[:, :, 0] < 3/255, colormask_image_diff_np[:, :, 1] < 3/255), 
                                           colormask_image_diff_np[:, :, 2] < 3/255).astype(np.uint8)
                                
        body_parts_mask_np += body_part_mask_np * (255 - body_part_index)

    # make background white
    alpha = np.ones_like(body_parts_mask_np) * 255   
    body_parts_mask_rgba_np = np.stack([alpha - body_parts_mask_np, alpha - body_parts_mask_np, alpha - body_parts_mask_np, alpha], axis=2)

    # source: https://blender.stackexchange.com/questions/92692/how-to-convert-numpy-array-into-image-and-add-it-to-images-inside-bpy
    image = bpy.data.images.new("body_parts_%s_mask" % camera_view, width=600, height=600)
    image.filepath = os.path.join(output_folder, "body_parts_%s_mask.png" % camera_view)
    image.file_format = 'PNG'
    image.pixels = body_parts_mask_rgba_np.ravel() / 255
    image.save()
    
    os.remove(colormask_path)

    # export joints
    skeleton = {}
    
    for bone_name, keypoint_index in POSE_POINTS.items():
        [x, y, z] = bones[bone_name].head
        
        x_rotated = math.cos(-z_angle) * x - math.sin(-z_angle) * y
        y_rotated = math.sin(-z_angle) * x + math.cos(-z_angle) * y
        
        image_x = x_rotated/(body_size/2) * image_size/2 + image_size/2
        image_y = image_size - (z/(body_size/2) * image_size/2 + image_size/2) # inverted
        image_z = image_size - (y_rotated/(body_size/2) * image_size/2 + image_size/2) # inverted
        
        skeleton[keypoint_index] = [round(image_x, 2), round(image_y, 2), round(image_z, 2)]
    
    skeleton_path = os.path.join(output_folder, "skeleton_%s.json" % camera_view)
    
    with open(skeleton_path, "w") as file:
        json.dump(skeleton, file)


obj.hide_render = False


bpy.ops.object.mode_set(mode='OBJECT')

def get_uv(ray_origin, ray_direction, obj):
    obj_data = obj.data    
    (hit, hit_location, _, polygon_id) = obj.ray_cast(ray_origin, ray_direction)

    if (not hit):
        return [None, None]

    polygon = obj_data.polygons[polygon_id]
    # obj_data.polygons[polygon_id].select = True

    vertices = []
    uvs = []

    for loop_id in polygon.loop_indices:
        loop = obj_data.loops[loop_id]
        vertex_id = loop.vertex_index
        vertex = obj_data.vertices[vertex_id]
        
        uv = obj_data.uv_layers.active.data[loop_id].uv

        vertices.append(vertex.co)
        uvs.append(uv)

    [p1, p2, p3] = vertices
    [uv1, uv2, uv3] = uvs
    f = hit_location

    # source: https://answers.unity.com/questions/383804/calculate-uv-coordinates-of-3d-point-on-plane-of-m.html
    f1 = p1 - f
    f2 = p2 - f
    f3 = p3 - f

    a = (p1-p2).cross(p1-p3).magnitude
    a1 = f2.cross(f3).magnitude / a
    a2 = f3.cross(f1).magnitude / a
    a3 = f1.cross(f2).magnitude / a

    uv = uv1 * a1 + uv2 * a2 + uv3 * a3
    
    if (ray_direction[0] == 0):
        depth_channel = 1
    else :
        depth_channel = 0
    
    depth_vector = hit_location * Vector(ray_direction)
    depth = depth_vector[depth_channel]
    
    return [uv, depth]

for [camera_view, camera_x, camera_y, z_angle] in camera_perspectives:
    uv_array = np.zeros((image_size, image_size, 2))
    uv_image = np.zeros((image_size, image_size, 4))
    depth_map = np.zeros((image_size, image_size))
    
    ray_direction = (-camera_x, -camera_y, 0)
    
    for x in range(image_size):
        for z in range(image_size):    
            x_norm = body_size_half * ((x - image_size_half) / image_size_half)
            z_norm = body_size_half * ((z - image_size_half) / image_size_half)
            
            x_rotated = math.cos(z_angle) * x_norm - math.sin(z_angle) * -camera_distance
            y_rotated = math.sin(z_angle) * x_norm + math.cos(z_angle) * -camera_distance
            ray_origin = (x_rotated, y_rotated, z_norm)
            
            [uv, depth] = get_uv(ray_origin, ray_direction, obj)
            
            uv_array[x, z] = uv
            depth_map[x, z] = depth
            
            if (uv != None):
                u_norm = uv[0] - 0.5
                v_norm = uv[1] - 0.5
                
                angle = math.atan2(v_norm, u_norm) + math.pi
                hue = angle / (2 * math.pi)
    
                # source: https://stackoverflow.com/questions/13211595/how-can-i-convert-coordinates-on-a-circle-to-coordinates-on-a-square
                circle_u = u_norm * math.sqrt(1 - 0.5 * v_norm**2)
                circle_v = v_norm * math.sqrt(1 - 0.5 * u_norm**2)
                
                length = math.sqrt(circle_u**2 + circle_v**2)
                saturation = length * 2
                
                c = Color()
                c.hsv = (hue, saturation, 1.0)
                uv_image[z, x] = [c.r, c.g, c.b, 1.0]
    
    uv_map_path = os.path.join(output_folder, "uv_%s.npy" % camera_view)
    np.save(uv_map_path, uv_array)
    
    depth_map_path = os.path.join(output_folder, "depth_%s.npy" % camera_view)
    np.save(depth_map_path, depth_map)
    
    uv_image_name = "uv_%s.png" % camera_view
    uv_image_path = os.path.join(output_folder, uv_image_name)

    output_image = bpy.data.images.new(uv_image_name, width=image_size, height=image_size, alpha=False)
    output_image.pixels = uv_image.ravel()
    output_image.filepath_raw = bpy.path.abspath(uv_image_path)
    output_image.save()
    

body_dimensions_path = os.path.join(output_folder, "body_dimensions.json")

body_dimensions = {
    "x": body_width,
    "y": body_height,
    "z": body_depth
}

with open(body_dimensions_path, "w") as file:
    json.dump(body_dimensions, file)
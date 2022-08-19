'''
converts remeshed figures from x3d to gltf/glb in Blender
'''
import bpy
import os
import sys

try :
    argv = sys.argv
    # get all args after "--"
    additional_argument_index = argv.index("--")
    arg = argv[additional_argument_index + 1:]  
    
    root_folder = arg[0]
    sub_folders = arg[1]
    subfolder_names = os.listdir(root_folder) if sub_folders == "" else sub_folders.split("|") 
    
except ValueError:
    root_folder = r"E:\CNN\implicit_functions\characters\output"
    subfolder_names = os.listdir(root_folder)

for subfolder in subfolder_names:
    # remove previous objects
    objs = bpy.data.objects

    for obj in objs:
        obj_name = obj.name    
        objs.remove(objs[obj_name])
    
    x3d_path = os.path.join(root_folder, subfolder, "mesh.x3d")
    bpy.ops.import_scene.x3d(filepath=x3d_path)
    obj = bpy.context.scene.objects.get("Shape_IndexedFaceSet")

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mat = bpy.data.materials['Material']
    obj.active_material = mat

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    node_shader = None
    for node in nodes:
        if node.type.startswith('BSDF'):
            node_shader = node
            break

    vertex_color_node = nodes.new(type="ShaderNodeVertexColor")
    links.new(vertex_color_node.outputs[0], node_shader.inputs[0])
    
    glb_path = os.path.join(root_folder, subfolder, "mesh.glb")
    bpy.ops.export_scene.gltf(filepath=glb_path)

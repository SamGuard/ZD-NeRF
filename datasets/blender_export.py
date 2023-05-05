import sys, os
import json
import bpy
import bmesh
import mathutils
import numpy as np

DEBUG = False

VIEWS = 1
RESOLUTION = 800
dataset = "train"
dataset_name = "shadows"
RESULTS_PATH = f"C:/Users/Student/projects/nerf/data/{dataset_name}/{dataset}"
IMAGE_PATH = f"/home/ruilongli/data/dnerf/{dataset_name}"
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = "PNG"
RANDOM_VIEWS = False
UPPER_VIEWS = False
USE_POINTS = False
CIRCLE_FIXED_START = (0.1, 0, np.pi)
LENGTH = 0
TIME_STEPS = list(
    range(0, 59, 10)
)  # [0,9,19,29,39,49] # None if time varies, otherwise set to desired frame
FILE_NAME_OFFSET = 0


fp = bpy.path.abspath(f"{RESULTS_PATH}")


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


if not os.path.exists(fp):
    os.makedirs(fp)

# Data to store in JSON file
out_data = {
    "camera_angle_x": bpy.data.objects["Camera"].data.angle_x,
}

# Render Optimizations
bpy.context.scene.render.use_persistent_data = True


# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes for additionally dumping albedo and normals.
# bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.image_settings.file_format = str(FORMAT)
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

if not DEBUG:
    # Create input render layer node.
    render_layers = tree.nodes.new("CompositorNodeRLayers")

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = "Depth Output"
    if FORMAT == "OPEN_EXR":
        links.new(render_layers.outputs["Depth"], depth_file_output.inputs[0])
    else:
        # Remap as other types can not represent the full range of depth.
        node_map = tree.nodes.new(type="CompositorNodeMapValue")
        # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
        node_map.offset = [-0.7]
        node_map.size = [DEPTH_SCALE]
        node_map.use_min = True
        node_map.min = [0]
        links.new(render_layers.outputs["Depth"], node_map.inputs[0])

        links.new(node_map.outputs[0], depth_file_output.inputs[0])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = "Normal Output"
    links.new(render_layers.outputs["Normal"], normal_file_output.inputs[0])

# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True

# Create collection for objects not to render with background


objs = [
    ob
    for ob in bpy.context.scene.objects
    if ob.type in ("EMPTY") and "Empty" in ob.name
]
bpy.ops.object.delete({"selected_objects": objs})


def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty


def get_vertex_pos_obj(obj, every_n=1):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    verts = obj.data.vertices
    bm = bmesh.new()
    bm.from_object(obj, depsgraph)
    bm.verts.ensure_lookup_table()
    poss = []
    matrix_world = obj.matrix_world
    for i,v in enumerate(bm.verts):
        if(i % every_n == 0):
            p = matrix_world @ v.co
            poss.append({"index": v.index // every_n, "pos": (p[0], p[1], p[2])})
    return poss


def get_vertex_pos():
    objs = [
        bpy.data.objects["Icosphere"],
        bpy.data.objects["Icosphere.001"],
        bpy.data.objects["Icosphere.002"],
    ]
    obj_points = list(map(lambda obj: get_vertex_pos_obj(obj,100), objs))
    points_index = 0
    out = []
    for obj in obj_points:
        for p in obj:
            out.append(p)
            out[-1]["index"] += points_index
        points_index += len(obj)
    
    return out



scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100

cam = scene.objects["Camera"]
cam.location = (0, 10.0, 2.0)
cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

scene.render.image_settings.file_format = "PNG"  # set output format to .png

from math import radians

stepsize = 1 * 360.0 / (VIEWS * len(TIME_STEPS))
rotation_mode = "X"

if not DEBUG:
    for output_node in [depth_file_output, normal_file_output]:
        output_node.base_path = ""

out_data["frames"] = []
out_data["vertices"] = []

if not RANDOM_VIEWS:
    b_empty.rotation_euler = CIRCLE_FIXED_START

for index, time in enumerate(TIME_STEPS):
    bpy.context.scene.frame_set(time)

    if(USE_POINTS):
        out_data["vertices"].append(
            {"time": time / max(TIME_STEPS), "data": get_vertex_pos()}
        )

    for i in range(0, VIEWS):
        image_file_name = "/r_" + str(i + index * VIEWS)
        scene.render.filepath = fp + image_file_name
        # depth_file_output.file_slots[0].path = scene.render.filepath + "_depth_"
        # normal_file_output.file_slots[0].path = scene.render.filepath + "_normal_"

        if DEBUG:
            continue
        else:
            bpy.ops.render.render(write_still=True)  # render still

        frame_data = {
            "file_path": IMAGE_PATH + "/" + dataset + image_file_name,
            "time": time / max(TIME_STEPS),
            "rotation": radians(stepsize),
            "transform_matrix": listify_matrix(cam.matrix_world),
        }
        out_data["frames"].append(frame_data)

        if RANDOM_VIEWS:
            if UPPER_VIEWS:
                rot = np.random.uniform(0, 1, size=3) * (1, 0, 2 * np.pi)
                rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi / 2)
                b_empty.rotation_euler = rot
            else:
                b_empty.rotation_euler[2] = np.random.uniform(0, 2 * np.pi, size=1)
        else:
            b_empty.rotation_euler[2] += radians(stepsize)
            print(b_empty.rotation_euler[2])

if not DEBUG:
    with open(fp + "/transforms_" + dataset + ".json", "w") as out_file:
        json.dump(out_data, out_file, indent=4)

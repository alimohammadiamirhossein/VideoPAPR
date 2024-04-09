"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple

import bpy
from mathutils import Vector
import cv2

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    default="/localhome/aaa324/Project/ButterflyExample/Blender/source/butterfly.glb",
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="/localhome/aaa324/Project/ButterflyExample/Blender")
parser.add_argument(
    "--engine", type=str, default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--num_images", type=int, default=12)
parser.add_argument("--camera_dist", type=int, default=1.5)

# argv = sys.argv[sys.argv.index("--") + 1 :]
# args = parser.parse_args(argv)
args = parser.parse_args()
args.engine = "CYCLES"

context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 32
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def add_lighting() -> None:
    # delete the default light
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 30
    bpy.data.objects["Area"].location[2] = 0.5
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint


def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)
    reset_scene()
    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene()
    add_lighting()
    cam, cam_constraint = setup_camera()
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    azimuth_angles = [0, 30, 150, 270, 90, 210, 330]
    zenith_angles = [0, 30, 30, 30, -20, -20, -20]

    num_frames = args.num_images
    for frame in range(0, 41, 5):
        # Set the frame
        scene.frame_set(frame)

        # Loop through different camera positions
        for i in range(len(azimuth_angles)):
            # set the camera position
            theta = math.radians(azimuth_angles[i]+270)
            phi = math.radians(zenith_angles[i]+270)
            point = (
                args.camera_dist * math.sin(phi) * math.cos(theta),
                args.camera_dist * math.sin(phi) * math.sin(theta),
                args.camera_dist * math.cos(phi),
            )
            cam.location = point

            # render the image
            render_path = os.path.join(args.output_dir, object_uid, f"{frame:02d}_{i:02d}.png")
            scene.render.filepath = render_path
            bpy.ops.render.render(write_still=True)

    return os.path.join(args.output_dir, object_uid)


def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


def make_transparent_white(image):
    # Iterate over each pixel
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Check if alpha channel value is less than 255 (not fully opaque)
            if image[y, x, 3] < 255:
                # Set pixel values to white
                image[y, x] = [255, 255, 255, 255]  # Set RGB values to white, alpha to fully opaque
            # else:
            #     image[y, x, [0, 1, 2]] = image[y, x, [2, 1, 0]]


def read_images(path):
    """Reads images from a specific frame directory."""
    images = {}
    # Iterate over the files in the directory
    for filename in sorted(os.listdir(path)):
        if filename.endswith(".png"):
            filepath = os.path.join(path, filename)
            # Read the image using OpenCV
            image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            make_transparent_white(image)
            if image is not None:
                fname = filename.split(".")[0]
                images[fname] = image
    return images


def concatenate_images(frames_dictionary, frame_number):
    """
    Concatenate all images for a specific frame horizontally.

    Args:
    - frame_directory (str): Directory containing images for the frame.
    - frame_number (str): Frame number in the format '00', '05', '10', etc.

    Returns:
    - concatenated_image (numpy.ndarray): Concatenated image.
    """
    # Read images for the specified frame
    images = []
    for filename in frames_dictionary.keys():
        if filename.startswith(frame_number + "_"):
            image = frames_dictionary[filename]
            if image is not None:
                images.append(image)

    # Concatenate images horizontally
    top_row = cv2.hconcat([images[1], images[4]])
    middle_row = cv2.hconcat([images[2], images[5]])
    bottom_row = cv2.hconcat([images[3], images[6]])
    concatenated_image = cv2.vconcat([top_row, middle_row, bottom_row])
    input_image = images[0]
    gt_image = concatenated_image

    return input_image, gt_image


def delete_files_in_directory(directory):
    # Get the list of files in the directory
    file_list = os.listdir(directory)

    # Iterate through the files and delete each one
    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


if __name__ == "__main__":
    try:
        start_time = time.time()
        # Download the object if it's a URL, otherwise use the provided path
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path

        # Save rendered images
        saved_path = save_images(local_path)
        end_time = time.time()
        print("Finished", local_path, "in", end_time - start_time, "seconds")

        # Delete the object if it was downloaded
        if args.object_path.startswith("http"):
            os.remove(local_path)

        # Read and process rendered images
        frames_dic = read_images(saved_path)
        frames = {}
        for fname in frames_dic.keys():
            fnum = fname.split("_")[0]
            if fnum not in frames:
                input_im, gt = concatenate_images(frames_dic, fnum)
                frames[fnum] = (input_im, gt)

        # Remove images from the saved path
        delete_files_in_directory(saved_path)

        # Save samples
        for fname in frames.keys():
            input_im, gt = frames[fname]
            input_path = os.path.join(saved_path, f"{fname}_input.png")
            gt_path = os.path.join(saved_path, f"{fname}_gt.png")
            cv2.imwrite(input_path, input_im)
            cv2.imwrite(gt_path, gt)

    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)

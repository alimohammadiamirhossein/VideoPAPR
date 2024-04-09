import bpy
import json
import os
from mathutils import Matrix
import math
# Define the main directory where you want to save the images
# Make sure to update this path to a directory on your system
main_directory = "/localhome/aaa324/Project/ButterflyExample/Blend"


# Ensure the directory exists
if not os.path.exists(main_directory):
    os.makedirs(main_directory)

# Set the scene
scene = bpy.context.scene

# Set render settings
scene.render.image_settings.file_format = 'PNG'  # Set image format

# Function to render from each camera
def render_from_each_camera(cameras, frame):
    # Create a subdirectory for the frame
    frame_directory = os.path.join(main_directory, f"frame_{frame:03}")
    if not os.path.exists(frame_directory):
        os.makedirs(frame_directory)

    for cam in cameras:
        # Set active camera
        scene.camera = cam

        # Set the current frame
        scene.frame_set(frame)

        # Set the filepath for the render
        filepath = os.path.join(frame_directory, f"{cam.name}_frame_{frame:03}.png")

        # Render the image
        scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)

# Get a list of all cameras in the scene
cameras = [obj for obj in scene.objects if obj.type == 'CAMERA']

# Loop through the frames and render
for frame in range(0, 41, 5):  # Start from frame 0, end at 40, step by 10 frames
    render_from_each_camera(cameras, frame)

print("Rendering completed.")
import bpy
import json
import os
from mathutils import Matrix

threshold = -1
data_save_path = "test"

# Function to get the camera's horizontal field of view in radians
def get_camera_angle_x(camera):
    if camera.data.type == "PERSP":
        return camera.data.angle_x
    else:
        sensor_width = camera.data.sensor_width
        return 2 * math.atan((sensor_width / 2) / camera.data.ortho_scale)


# Function to construct the transformation matrix for the camera
def get_transform_matrix(obj):
    return obj.matrix_world.inverted()


# Directory where the JSON files will be saved
main_directory = (
    "your\\directory\\here"
)
if not os.path.exists(main_directory):
    os.makedirs(main_directory)

# Set the scene
scene = bpy.context.scene

# Get a list of all cameras in the scene
cameras = [obj for obj in scene.objects if obj.type == "CAMERA"]

# Loop through each frame and gather JSON data for all cameras
for frame in range(scene.frame_start, scene.frame_end + 1, 5):
    # Create a directory for the current frame
    frame_directory = os.path.join(main_directory, f"frame_{frame:03}")
    if not os.path.exists(frame_directory):
        os.makedirs(frame_directory)

    # Initialize the frame data structure
    frame_data = {
        "frame": frame,
        "camera_angle_x": get_camera_angle_x(cameras[0]),
        "camera_data": [],
    }

    # Gather data for each camera
    for cam_index, cam in enumerate(cameras, 1):
        if cam_index > threshold:
            # Set the scene to the current frame
            scene.frame_set(frame)

            # Camera attributes for JSON
            camera_info = {
                "file_path": os.path.join(
                    f"./{data_save_path}/", f"{cam.name:02}_frame_{frame:03}_depth_0000.png"
                ),
                "rotation": cam.rotation_euler.z,
                "transform_matrix": [list(row) for row in get_transform_matrix(cam)],
            }
            frame_data["camera_data"].append(camera_info)

    # Write the JSON data to a file for the frame
    json_path = os.path.join(frame_directory, f"frame_{frame:03}_depth.json")
    with open(json_path, "w") as json_file:
        json.dump(frame_data, json_file, indent=4)

print("JSON generation completed.")

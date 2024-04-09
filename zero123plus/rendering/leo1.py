import bpy
import math
from mathutils import Vector

def create_camera_at_sphere_point(radius, azimuth, zenith, index):
    """
    Create a camera at a specified point on a sphere.

    Parameters:
        radius (float): Radius of the sphere.
        azimuth (float): Azimuth angle in radians.
        zenith (float): Zenith angle in radians.
        index (int): Index of the camera.

    Returns:
        bpy.types.Object: The created camera object.
    """
    # Convert spherical coordinates to Cartesian coordinates for the camera's position
    x = radius * math.sin(zenith) * math.cos(azimuth)
    y = radius * math.sin(zenith) * math.sin(azimuth)
    z = radius * math.cos(zenith)

    # Create the camera at this position
    bpy.ops.object.camera_add(location=(x, y, z))
    camera = bpy.context.object
    camera.name = f"Camera_{index:02}"

    # Point the camera towards the origin by changing its rotation
    direction = Vector((0.0, 0.0, 0.0)) - camera.location
    # Rotate it
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    return camera

# Location from the provided image
camera_location = Vector((-44.15891, 97.26670, 38.43147))

# Remove all cameras
for obj in bpy.context.scene.objects:
    if obj.type == 'CAMERA':
        bpy.data.objects.remove(obj, do_unlink=True)

# Calculate radius from the origin
radius = camera_location.length

# Define azimuth and zenith angles in degrees
azimuth_angles = [0, 30, 150, 270, 90, 210, 330]
zenith_angles = [0, 30, 30, 30, -20, -20, -20]

# Convert azimuth and zenith angles from degrees to radians
azimuth_angles = [math.radians(angle + 270) for angle in azimuth_angles]  # Adding 90 to match the original code
zenith_angles = [math.radians(angle + 270) for angle in zenith_angles]  # Adding 90 to match the original code

# Create cameras around the origin using azimuth and zenith angles
for i in range(len(azimuth_angles)):
    azimuth = azimuth_angles[i]
    zenith = zenith_angles[i]
    create_camera_at_sphere_point(radius, azimuth, zenith, i)

print("Cameras have been created.")
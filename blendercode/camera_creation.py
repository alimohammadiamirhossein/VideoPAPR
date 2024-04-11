import bpy
import math
import random
from mathutils import Vector


# Function to create a camera looking at the origin
def create_camera_at_sphere_point(radius, theta, phi, index):
    # Convert spherical coordinates to Cartesian coordinates for the camera's position
    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * math.cos(theta)

    # Create the camera at this position
    bpy.ops.object.camera_add(location=(x, y, z))
    camera = bpy.context.object
    camera.name = f"Camera_{index:02}"

    # Point the camera towards the origin by changing its rotation
    direction = Vector((0.0, 0.0, 0.0)) - camera.location
    # Rotate it
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()

    return camera


# Location from the provided image
camera_location = Vector((-47.15891, 99.26670, 50.43147))

# Calculate radius from the origin
radius = camera_location.length

# Create 100 cameras randomly positioned around the origin
for i in range(100):
    # Generate random spherical coordinates
    theta = random.uniform(0, math.pi)  # Angle from the top
    phi = random.uniform(0, 2 * math.pi)  # Angle around the sphere
    create_camera_at_sphere_point(radius, theta, phi, i)

print("100 cameras have been created.")

import bpy
import os
import argparse
import sys
import subprocess


def make_sphere(radius: int, path: str):

    check_blender()
    # Add a UV Sphere with a high number of segments and rings for smoothness
    bpy.ops.mesh.primitive_uv_sphere_add(segments=256, ring_count=128, radius=radius, location=(0, 0, 0))

    # Get the created sphere
    obj = bpy.context.object

    # Ensure the object is selected
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Enable the OBJ export add-on
    bpy.ops.preferences.addon_enable(module='io_scene_obj')

    # Export the object as an OBJ file
    bpy.ops.wm.obj_export(filepath=path)

    print(f"Sphere with radius {radius} saved to {path}")


def check_blender():
    # Check if Blender is installed
    try:
        subprocess.run(["blender", "--version"], capture_output=True, check=True)
        print(f"Blender is installed and available at {subprocess.check_output(['which', 'blender']).decode().strip()}")
        print(f"Blender version: {subprocess.check_output(['blender', '--version']).decode().strip()}")
    except FileNotFoundError:
        print("Blender is not installed. Please install Blender to use this script.")
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("Blender is not installed correctly. Please reinstall Blender.")
        sys.exit(1)


if __name__ == "__main__":
    make_sphere(10, "../data/sphere10.obj")

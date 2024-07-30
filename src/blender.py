import bpy
import os


def make_sphere(radius: int, path: str):
    # Add a UV Sphere with a high number of segments and rings for smoothness
    bpy.ops.mesh.primitive_uv_sphere_add(segments=256, ring_count=128, radius=radius, location=(0, 0, 0))

    # Get the created sphere
    obj = bpy.context.object

    # Ensure the object is selected
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Export the object as an OBJ file
    bpy.ops.export_scene.obj(filepath=path, use_selection=True, axis_forward='-Z', axis_up='Y')


if __name__ == "__main__":
    # blender --background --python ./src/blender.py
    r = 16
    make_sphere(16, os.path.join(os.path.dirname(__file__), f"../data/sphere{r}.obj"))

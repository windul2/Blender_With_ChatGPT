"""Create a medium-poly mechanical robot, render it, and export source assets.

Run inside Blender:
    blender --background --factory-startup --python scripts/create_mech.py -- \
      --output-dir output --resolution 1024 --samples 32
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import bpy
from mathutils import Vector


EXPORT_OBJECTS: list[bpy.types.Object] = []


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    script_args = argv[argv.index("--") + 1 :] if "--" in argv else []

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args(script_args)

    args.resolution = max(512, min(args.resolution, 2160))
    args.samples = max(8, min(args.samples, 256))
    return args


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def create_material(
    name: str,
    base_color: tuple[float, float, float, float],
    *,
    metallic: float = 0.0,
    roughness: float = 0.45,
    emission_color: tuple[float, float, float, float] | None = None,
    emission_strength: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        raise RuntimeError(f"Principled BSDF node missing for {name}")

    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness

    if emission_color is not None and emission_strength > 0:
        emission_input = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
        if emission_input is not None:
            emission_input.default_value = emission_color
        strength_input = bsdf.inputs.get("Emission Strength")
        if strength_input is not None:
            strength_input.default_value = emission_strength

    return material


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    if obj.data and hasattr(obj.data, "materials"):
        obj.data.materials.append(material)


def add_bevel(obj: bpy.types.Object, width: float, segments: int = 3) -> None:
    modifier = obj.modifiers.new(name="Edge bevel", type="BEVEL")
    modifier.width = width
    modifier.segments = segments
    modifier.limit_method = "ANGLE"
    modifier.angle_limit = math.radians(25.0)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def bevelled_box(
    name: str,
    location: tuple[float, float, float],
    half_size: tuple[float, float, float],
    material: bpy.types.Material,
    *,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    bevel: float = 0.12,
    export: bool = True,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.scale = half_size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    add_bevel(obj, min(bevel, min(half_size) * 0.45), segments=3)
    assign_material(obj, material)
    if export:
        EXPORT_OBJECTS.append(obj)
    return obj


def cylinder(
    name: str,
    location: tuple[float, float, float],
    radius: float,
    depth: float,
    material: bpy.types.Material,
    *,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 20,
    bevel: float = 0.06,
    export: bool = True,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        end_fill_type="NGON",
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    add_bevel(obj, bevel, segments=2)
    assign_material(obj, material)
    if export:
        EXPORT_OBJECTS.append(obj)
    return obj


def uv_sphere(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    *,
    export: bool = True,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=24,
        ring_count=12,
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    assign_material(obj, material)
    if export:
        EXPORT_OBJECTS.append(obj)
    return obj


def add_mech(materials: dict[str, bpy.types.Material]) -> None:
    dark = materials["dark"]
    armor = materials["armor"]
    accent = materials["accent"]
    rubber = materials["rubber"]
    glow = materials["glow"]

    # Feet and lower legs.
    for side, x in (("L", -1.25), ("R", 1.25)):
        bevelled_box(
            f"{side}_Foot",
            (x, -0.18, 0.48),
            (0.78, 1.15, 0.38),
            dark,
            bevel=0.18,
        )
        bevelled_box(
            f"{side}_ToeArmor",
            (x, -1.05, 0.64),
            (0.68, 0.35, 0.22),
            armor,
            rotation=(math.radians(-8), 0.0, 0.0),
            bevel=0.10,
        )
        cylinder(
            f"{side}_Ankle",
            (x, 0.0, 1.02),
            0.35,
            0.52,
            rubber,
            vertices=20,
        )
        bevelled_box(
            f"{side}_Shin",
            (x, 0.08, 2.05),
            (0.70, 0.72, 1.05),
            armor,
            rotation=(math.radians(-2 if side == "L" else 2), 0.0, 0.0),
            bevel=0.17,
        )
        bevelled_box(
            f"{side}_ShinPlate",
            (x, -0.72, 2.12),
            (0.54, 0.16, 0.76),
            accent,
            rotation=(math.radians(-4), 0.0, 0.0),
            bevel=0.07,
        )
        cylinder(
            f"{side}_KneeJoint",
            (x, 0.0, 3.22),
            0.48,
            0.62,
            rubber,
            rotation=(math.radians(90), 0.0, 0.0),
            vertices=20,
            bevel=0.05,
        )
        bevelled_box(
            f"{side}_KneeCap",
            (x, -0.55, 3.22),
            (0.54, 0.20, 0.46),
            accent,
            rotation=(math.radians(12), 0.0, 0.0),
            bevel=0.09,
        )
        bevelled_box(
            f"{side}_Thigh",
            (x, 0.06, 4.18),
            (0.62, 0.62, 0.85),
            dark,
            rotation=(0.0, math.radians(-3 if side == "L" else 3), 0.0),
            bevel=0.15,
        )
        bevelled_box(
            f"{side}_ThighArmor",
            (x, -0.62, 4.24),
            (0.48, 0.13, 0.56),
            armor,
            bevel=0.06,
        )

    # Pelvis and torso core.
    bevelled_box("Pelvis", (0.0, 0.05, 5.15), (1.85, 0.98, 0.58), dark, bevel=0.22)
    bevelled_box("PelvisFront", (0.0, -0.96, 5.18), (1.20, 0.18, 0.42), accent, bevel=0.10)
    for x in (-1.55, 1.55):
        bevelled_box("HipArmor", (x, -0.12, 5.08), (0.45, 0.92, 0.52), armor, bevel=0.14)

    cylinder("WaistJoint", (0.0, 0.0, 5.88), 0.88, 0.50, rubber, vertices=24, bevel=0.07)
    bevelled_box("Abdomen", (0.0, 0.02, 6.35), (1.15, 0.78, 0.55), dark, bevel=0.16)
    bevelled_box("Torso", (0.0, 0.08, 7.45), (2.15, 1.08, 1.20), armor, bevel=0.25)
    bevelled_box(
        "ChestCenter",
        (0.0, -1.08, 7.50),
        (1.15, 0.18, 0.78),
        accent,
        rotation=(math.radians(-4), 0.0, 0.0),
        bevel=0.09,
    )
    bevelled_box("ChestCore", (0.0, -1.30, 7.58), (0.44, 0.09, 0.30), glow, bevel=0.05)

    # Chest vents and small hard-surface details.
    for side in (-1, 1):
        for index in range(3):
            bevelled_box(
                f"ChestVent_{side}_{index}",
                (side * (1.18 + index * 0.20), -1.12, 7.54 - index * 0.12),
                (0.065, 0.055, 0.36),
                dark,
                rotation=(0.0, math.radians(side * 12), math.radians(side * -12)),
                bevel=0.025,
            )

    # Neck and head.
    cylinder("Neck", (0.0, 0.0, 8.77), 0.42, 0.42, rubber, vertices=20, bevel=0.05)
    bevelled_box("Head", (0.0, -0.03, 9.28), (0.75, 0.66, 0.53), armor, bevel=0.16)
    bevelled_box("FacePlate", (0.0, -0.68, 9.23), (0.57, 0.10, 0.34), dark, bevel=0.07)
    bevelled_box("Visor", (0.0, -0.79, 9.38), (0.45, 0.055, 0.095), glow, bevel=0.035)
    cylinder("HeadSideL", (-0.78, -0.02, 9.27), 0.22, 0.18, accent, rotation=(0.0, math.radians(90), 0.0), vertices=16)
    cylinder("HeadSideR", (0.78, -0.02, 9.27), 0.22, 0.18, accent, rotation=(0.0, math.radians(90), 0.0), vertices=16)
    cylinder("Antenna", (0.42, 0.0, 9.98), 0.055, 1.12, dark, rotation=(0.0, math.radians(-13), 0.0), vertices=12, bevel=0.02)
    uv_sphere("AntennaLight", (0.55, 0.0, 10.53), (0.10, 0.10, 0.10), glow)

    # Arms. Right side ends in a cannon, left side in a claw-like hand.
    for side, x in (("L", -3.0), ("R", 3.0)):
        cylinder(
            f"{side}_ShoulderJoint",
            (x, 0.02, 7.76),
            0.64,
            0.72,
            rubber,
            rotation=(0.0, math.radians(90), 0.0),
            vertices=24,
            bevel=0.07,
        )
        bevelled_box(
            f"{side}_ShoulderArmor",
            (x, -0.05, 7.86),
            (0.78, 0.96, 0.66),
            armor,
            rotation=(0.0, 0.0, math.radians(3 if side == "L" else -3)),
            bevel=0.19,
        )
        bevelled_box(
            f"{side}_UpperArm",
            (x, 0.02, 6.64),
            (0.53, 0.58, 0.72),
            dark,
            bevel=0.14,
        )
        cylinder(
            f"{side}_Elbow",
            (x, 0.0, 5.77),
            0.40,
            0.56,
            rubber,
            rotation=(math.radians(90), 0.0, 0.0),
            vertices=20,
        )

    # Left forearm and hand.
    bevelled_box("L_Forearm", (-3.0, -0.02, 4.93), (0.60, 0.68, 0.70), armor, bevel=0.17)
    cylinder("L_Wrist", (-3.0, -0.03, 4.12), 0.30, 0.35, rubber, vertices=16)
    bevelled_box("L_Palm", (-3.0, -0.08, 3.72), (0.48, 0.52, 0.34), dark, bevel=0.11)
    for index, dx in enumerate((-0.34, 0.0, 0.34)):
        bevelled_box(
            f"L_Finger_{index}",
            (-3.0 + dx, -0.42, 3.32),
            (0.11, 0.14, 0.36),
            armor,
            rotation=(math.radians(-8), 0.0, 0.0),
            bevel=0.05,
        )

    # Right forearm cannon.
    bevelled_box("R_CannonBody", (3.0, -0.08, 4.82), (0.76, 0.78, 0.92), dark, bevel=0.19)
    cylinder("R_CannonBarrel", (3.0, -0.20, 3.62), 0.42, 1.75, armor, vertices=24, bevel=0.08)
    cylinder("R_CannonMuzzle", (3.0, -0.20, 2.72), 0.54, 0.34, accent, vertices=24, bevel=0.06)
    cylinder("R_CannonCore", (3.0, -0.20, 2.52), 0.25, 0.12, glow, vertices=20, bevel=0.03)
    for angle in (0, 90, 180, 270):
        rad = math.radians(angle)
        bevelled_box(
            f"R_CannonFin_{angle}",
            (3.0 + math.cos(rad) * 0.56, -0.20 + math.sin(rad) * 0.56, 3.75),
            (0.13, 0.13, 0.75),
            accent,
            bevel=0.05,
        )

    # Backpack and exhausts.
    bevelled_box("Backpack", (0.0, 1.12, 7.32), (1.48, 0.48, 1.00), dark, bevel=0.17)
    for x in (-0.78, 0.78):
        cylinder(
            "BackExhaust",
            (x, 1.58, 7.18),
            0.32,
            0.70,
            accent,
            rotation=(math.radians(90), 0.0, 0.0),
            vertices=20,
            bevel=0.05,
        )


def add_stage(materials: dict[str, bpy.types.Material]) -> None:
    floor = materials["floor"]
    dark = materials["dark"]
    accent = materials["accent"]

    cylinder(
        "DisplayBase",
        (0.0, 0.0, 0.02),
        5.25,
        0.28,
        floor,
        vertices=64,
        bevel=0.10,
        export=False,
    )
    cylinder(
        "DisplayRing",
        (0.0, 0.0, 0.18),
        4.55,
        0.10,
        dark,
        vertices=64,
        bevel=0.04,
        export=False,
    )

    for angle in range(0, 360, 45):
        rad = math.radians(angle)
        bevelled_box(
            f"BaseMarker_{angle}",
            (math.cos(rad) * 4.25, math.sin(rad) * 4.25, 0.27),
            (0.42, 0.10, 0.04),
            accent,
            rotation=(0.0, 0.0, rad + math.radians(90)),
            bevel=0.025,
            export=False,
        )

    bpy.ops.mesh.primitive_plane_add(size=50, location=(0.0, 0.0, -0.16))
    plane = bpy.context.object
    plane.name = "Ground"
    assign_material(plane, floor)


def point_at(obj: bpy.types.Object, target: Iterable[float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_area_light(
    name: str,
    location: tuple[float, float, float],
    energy: float,
    size: float,
    color: tuple[float, float, float],
    target: tuple[float, float, float],
) -> None:
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name=name, object_data=data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    point_at(obj, target)


def configure_camera_and_lights() -> None:
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (11.8, -15.2, 9.4)
    camera_data.lens = 58
    camera_data.sensor_width = 36
    point_at(camera, (0.0, 0.0, 5.0))
    bpy.context.scene.camera = camera

    add_area_light(
        "KeyLight",
        (6.5, -8.0, 13.5),
        1600,
        5.5,
        (1.0, 0.80, 0.62),
        (0.0, 0.0, 5.0),
    )
    add_area_light(
        "FillLight",
        (-8.0, -3.0, 8.0),
        900,
        5.0,
        (0.38, 0.58, 1.0),
        (0.0, 0.0, 5.0),
    )
    add_area_light(
        "RimLight",
        (2.0, 7.5, 12.5),
        1300,
        4.0,
        (0.35, 0.72, 1.0),
        (0.0, 0.0, 6.2),
    )
    add_area_light(
        "FrontSoftbox",
        (0.0, -5.0, 4.0),
        500,
        3.0,
        (1.0, 0.36, 0.16),
        (0.0, 0.0, 5.0),
    )


def configure_render(args: argparse.Namespace, output_dir: Path) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 2
    scene.cycles.transparent_max_bounces = 4
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.filepath = str(output_dir / "mech_render.png")

    scene.render.image_settings.color_depth = "8"
    scene.render.resolution_percentage = 100

    world = scene.world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.008, 0.012, 0.025, 1.0)
        background.inputs["Strength"].default_value = 0.18

    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except (TypeError, ValueError):
        print("AgX look preset is unavailable; using Blender default look.")


def export_assets(output_dir: Path) -> None:
    blend_path = output_dir / "mid_poly_mech.blend"
    glb_path = output_dir / "mid_poly_mech.glb"

    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    bpy.ops.object.select_all(action="DESELECT")
    for obj in EXPORT_OBJECTS:
        if obj.name in bpy.context.view_layer.objects:
            obj.select_set(True)
    if EXPORT_OBJECTS:
        bpy.context.view_layer.objects.active = EXPORT_OBJECTS[0]

    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        use_selection=True,
        export_cameras=False,
        export_lights=False,
    )


def write_info(output_dir: Path, args: argparse.Namespace) -> None:
    mesh_objects = [obj for obj in EXPORT_OBJECTS if obj.type == "MESH"]
    vertex_count = sum(len(obj.data.vertices) for obj in mesh_objects)
    polygon_count = sum(len(obj.data.polygons) for obj in mesh_objects)

    info = (
        "Procedural Mid-Poly Mech\n"
        f"Blender: {bpy.app.version_string}\n"
        f"Resolution: {args.resolution} x {args.resolution}\n"
        f"Requested samples: {args.samples}\n"
        f"Mesh objects: {len(mesh_objects)}\n"
        f"Vertices: {vertex_count}\n"
        f"Polygons: {polygon_count}\n"
        "Files: mech_render.png, mid_poly_mech.blend, mid_poly_mech.glb\n"
    )
    (output_dir / "render_info.txt").write_text(info, encoding="utf-8")
    print(info)


def main() -> None:
    EXPORT_OBJECTS.clear()
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clear_scene()

    materials = {
        "dark": create_material("Gunmetal", (0.028, 0.038, 0.055, 1.0), metallic=0.88, roughness=0.24),
        "armor": create_material("Titanium Armor", (0.25, 0.31, 0.38, 1.0), metallic=0.78, roughness=0.27),
        "accent": create_material("Warning Orange", (0.85, 0.12, 0.018, 1.0), metallic=0.55, roughness=0.25),
        "rubber": create_material("Joint Rubber", (0.012, 0.014, 0.018, 1.0), metallic=0.05, roughness=0.58),
        "glow": create_material(
            "Cyan Energy",
            (0.01, 0.22, 0.34, 1.0),
            metallic=0.18,
            roughness=0.20,
            emission_color=(0.0, 0.75, 1.0, 1.0),
            emission_strength=8.0,
        ),
        "floor": create_material("Stage Floor", (0.018, 0.022, 0.030, 1.0), metallic=0.52, roughness=0.34),
    }

    add_mech(materials)
    add_stage(materials)
    configure_camera_and_lights()
    configure_render(args, output_dir)

    bpy.context.scene.render.filepath = str(output_dir / "mech_render.png")
    bpy.ops.render.render(write_still=True)
    export_assets(output_dir)
    write_info(output_dir, args)


if __name__ == "__main__":
    main()

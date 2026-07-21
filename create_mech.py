"""Create a high-detail stylized chibi mech, render it, and export assets.

Run inside Blender:
    blender --background --factory-startup --python create_mech.py -- \
      --output-dir output --resolution 1024 --samples 96
"""

from __future__ import annotations

import argparse
import math
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
    parser.add_argument("--samples", type=int, default=96)
    args = parser.parse_args(script_args)

    args.resolution = max(512, min(args.resolution, 2160))
    args.samples = max(16, min(args.samples, 256))
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
        bpy.data.images,
    ):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def create_material(
    name: str,
    base_color: tuple[float, float, float, float],
    *,
    metallic: float = 0.0,
    roughness: float = 0.4,
    specular_ior_level: float = 0.5,
    coat: float = 0.0,
    coat_roughness: float = 0.03,
    transmission: float = 0.0,
    emission_color: tuple[float, float, float, float] | None = None,
    emission_strength: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        raise RuntimeError(f"Principled BSDF missing for {name}")

    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = specular_ior_level
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat
    elif "Coat" in bsdf.inputs:
        bsdf.inputs["Coat"].default_value = coat
    if "Coat Roughness" in bsdf.inputs:
        bsdf.inputs["Coat Roughness"].default_value = coat_roughness
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = transmission
    elif "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = transmission

    if emission_color is not None and emission_strength > 0.0:
        emission_input = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
        if emission_input is not None:
            emission_input.default_value = emission_color
        strength_input = bsdf.inputs.get("Emission Strength")
        if strength_input is not None:
            strength_input.default_value = emission_strength

    return mat


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    if obj.data and hasattr(obj.data, "materials"):
        obj.data.materials.clear()
        obj.data.materials.append(material)


def smooth_object(obj: bpy.types.Object, angle_deg: float = 35.0) -> None:
    if obj.type != "MESH":
        return
    for poly in obj.data.polygons:
        poly.use_smooth = True
    if hasattr(obj.data, "use_auto_smooth"):
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = math.radians(angle_deg)


def add_bevel(obj: bpy.types.Object, width: float, segments: int = 3, angle_deg: float = 28.0) -> None:
    mod = obj.modifiers.new(name="Bevel", type="BEVEL")
    mod.width = width
    mod.segments = segments
    mod.limit_method = "ANGLE"
    mod.angle_limit = math.radians(angle_deg)
    mod.profile = 0.7
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)


def add_subsurf(obj: bpy.types.Object, levels: int = 1) -> None:
    mod = obj.modifiers.new(name="Subdivision", type="SUBSURF")
    mod.levels = levels
    mod.render_levels = levels
    mod.subdivision_type = "CATMULL_CLARK"
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)
    smooth_object(obj)


def remember(obj: bpy.types.Object, export: bool = True) -> bpy.types.Object:
    if export:
        EXPORT_OBJECTS.append(obj)
    return obj


def bevelled_box(
    name: str,
    location: tuple[float, float, float],
    half_size: tuple[float, float, float],
    material: bpy.types.Material,
    *,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    bevel: float = 0.06,
    bevel_segments: int = 3,
    subsurf: int = 0,
    export: bool = True,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.scale = half_size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    add_bevel(obj, min(bevel, min(half_size) * 0.45), segments=bevel_segments)
    if subsurf > 0:
        add_subsurf(obj, levels=subsurf)
    assign_material(obj, material)
    smooth_object(obj)
    return remember(obj, export)


def cylinder(
    name: str,
    location: tuple[float, float, float],
    radius: float,
    depth: float,
    material: bpy.types.Material,
    *,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vertices: int = 24,
    bevel: float = 0.04,
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
    smooth_object(obj)
    return remember(obj, export)


def sphere(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    *,
    segments: int = 32,
    rings: int = 18,
    export: bool = True,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    assign_material(obj, material)
    smooth_object(obj)
    return remember(obj, export)


def capsule(
    name: str,
    location: tuple[float, float, float],
    radius: float,
    length: float,
    material: bpy.types.Material,
    *,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    export: bool = True,
) -> bpy.types.Object:
    parts = []
    cyl = cylinder(
        f"{name}_Core",
        location,
        radius,
        length,
        material,
        rotation=rotation,
        vertices=28,
        bevel=min(radius * 0.12, 0.05),
        export=False,
    )
    parts.append(cyl)

    axis = Vector((0.0, 0.0, 1.0))
    rot = Vector((0.0, 0.0, 1.0)).rotation_difference(Vector((0.0, 0.0, 1.0))).to_euler()
    del axis, rot

    # Capsule assumes local Z alignment, so create end spheres and rotate all together.
    end_a = sphere(
        f"{name}_EndA",
        (location[0], location[1], location[2] + length / 2.0),
        (radius, radius, radius),
        material,
        segments=28,
        rings=14,
        export=False,
    )
    end_b = sphere(
        f"{name}_EndB",
        (location[0], location[1], location[2] - length / 2.0),
        (radius, radius, radius),
        material,
        segments=28,
        rings=14,
        export=False,
    )
    parts.extend([end_a, end_b])

    bpy.ops.object.select_all(action="DESELECT")
    for obj in parts:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = cyl
    bpy.ops.object.join()
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = rotation
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    assign_material(obj, material)
    smooth_object(obj)
    return remember(obj, export)


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
    data.shape = "RECTANGLE"
    data.size = size
    data.size_y = size * 0.75
    data.color = color
    obj = bpy.data.objects.new(name=name, object_data=data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    point_at(obj, target)


def add_panel_strips(materials: dict[str, bpy.types.Material]) -> None:
    dark = materials["dark"]
    red = materials["red"]
    for side in (-1, 1):
        for i in range(4):
            bevelled_box(
                f"HeadPanelStrip_{side}_{i}",
                (side * (1.58 + i * 0.06), -0.36 + i * 0.11, 10.94 - i * 0.12),
                (0.05, 0.018, 0.16),
                red if i == 1 else dark,
                rotation=(math.radians(14), math.radians(side * 18), math.radians(side * 14)),
                bevel=0.012,
            )


def add_head(materials: dict[str, bpy.types.Material]) -> None:
    white = materials["white"]
    dark = materials["dark"]
    gray = materials["gray"]
    visor = materials["visor"]
    blue = materials["blue"]
    red = materials["red"]

    sphere("HeadCore", (0.0, 0.08, 9.95), (1.72, 1.46, 1.82), dark, segments=36, rings=20)
    bevelled_box(
        "HelmetMain",
        (0.0, -0.12, 10.46),
        (2.10, 1.55, 1.60),
        white,
        rotation=(math.radians(8), 0.0, 0.0),
        bevel=0.18,
        bevel_segments=4,
        subsurf=1,
    )
    bevelled_box(
        "HelmetTopPlate",
        (0.0, 0.64, 11.78),
        (1.54, 0.76, 0.32),
        white,
        rotation=(math.radians(-8), 0.0, 0.0),
        bevel=0.08,
        subsurf=1,
    )
    bevelled_box(
        "HelmetForeheadBand",
        (0.0, -1.22, 10.94),
        (1.85, 0.18, 0.22),
        gray,
        rotation=(math.radians(-8), 0.0, 0.0),
        bevel=0.05,
    )
    bevelled_box(
        "VisorShell",
        (0.0, -1.08, 10.10),
        (1.34, 0.42, 0.74),
        dark,
        rotation=(math.radians(-12), 0.0, 0.0),
        bevel=0.10,
        subsurf=1,
    )
    bevelled_box(
        "VisorGlass",
        (0.0, -1.33, 10.06),
        (1.06, 0.16, 0.42),
        visor,
        rotation=(math.radians(-14), 0.0, 0.0),
        bevel=0.04,
    )
    for side in (-1, 1):
        sphere(
            f"EyeGlow_{side}",
            (side * 0.44, -1.24, 9.72),
            (0.24, 0.12, 0.34),
            blue,
            segments=24,
            rings=14,
        )
        bevelled_box(
            f"CheekArmor_{side}",
            (side * 1.18, -1.02, 9.72),
            (0.54, 0.42, 0.68),
            white,
            rotation=(math.radians(18), math.radians(side * 16), math.radians(side * 10)),
            bevel=0.08,
            subsurf=1,
        )
        cylinder(
            f"EarModule_{side}",
            (side * 2.00, 0.10, 10.48),
            0.34,
            0.42,
            gray,
            rotation=(0.0, math.radians(90), 0.0),
            vertices=24,
            bevel=0.04,
        )
        cylinder(
            f"EarRing_{side}",
            (side * 2.18, 0.10, 10.48),
            0.16,
            0.10,
            blue,
            rotation=(0.0, math.radians(90), 0.0),
            vertices=20,
            bevel=0.02,
        )
        bevelled_box(
            f"RearPod_{side}",
            (side * 1.84, 0.88, 10.82),
            (0.72, 0.70, 0.84),
            white,
            rotation=(math.radians(-10), math.radians(side * 8), math.radians(side * -8)),
            bevel=0.10,
            bevel_segments=4,
            subsurf=1,
        )
        cylinder(
            f"RearWheel_{side}",
            (side * 2.00, 0.78, 9.10),
            0.28,
            0.30,
            dark,
            rotation=(0.0, math.radians(90), 0.0),
            vertices=24,
            bevel=0.03,
        )

    bevelled_box(
        "HelmetChin",
        (0.0, -0.92, 8.98),
        (0.94, 0.36, 0.32),
        white,
        rotation=(math.radians(-6), 0.0, 0.0),
        bevel=0.06,
        subsurf=1,
    )
    bevelled_box(
        "HelmetMouthGuard",
        (0.0, -1.20, 9.26),
        (0.62, 0.12, 0.20),
        dark,
        rotation=(math.radians(-10), 0.0, 0.0),
        bevel=0.04,
    )
    bevelled_box(
        "ForeMark",
        (0.0, -1.38, 10.72),
        (0.18, 0.05, 0.12),
        red,
        rotation=(math.radians(-10), 0.0, 0.0),
        bevel=0.02,
    )
    add_panel_strips(materials)


def add_torso(materials: dict[str, bpy.types.Material]) -> None:
    white = materials["white"]
    dark = materials["dark"]
    gray = materials["gray"]
    blue = materials["blue"]
    red = materials["red"]

    cylinder("NeckGimbal", (0.0, 0.0, 8.36), 0.42, 0.58, dark, vertices=24, bevel=0.05)
    bevelled_box(
        "TorsoCore",
        (0.0, 0.06, 6.72),
        (1.48, 1.08, 1.72),
        dark,
        bevel=0.12,
        bevel_segments=4,
        subsurf=1,
    )
    bevelled_box(
        "ChestShell",
        (0.0, -0.34, 7.04),
        (1.96, 0.78, 1.32),
        white,
        rotation=(math.radians(8), 0.0, 0.0),
        bevel=0.10,
        bevel_segments=4,
        subsurf=1,
    )
    bevelled_box(
        "ChestCenterPlate",
        (0.0, -1.10, 6.82),
        (0.82, 0.18, 0.72),
        gray,
        rotation=(math.radians(-6), 0.0, 0.0),
        bevel=0.04,
    )
    bevelled_box(
        "ChestCoreGlow",
        (0.0, -1.22, 6.84),
        (0.24, 0.05, 0.28),
        blue,
        bevel=0.02,
    )
    for side in (-1, 1):
        bevelled_box(
            f"ChestSideVent_{side}",
            (side * 1.14, -0.92, 6.88),
            (0.26, 0.10, 0.58),
            dark,
            rotation=(0.0, math.radians(side * 18), math.radians(side * -4)),
            bevel=0.03,
        )
        bevelled_box(
            f"ChestRedAccent_{side}",
            (side * 0.82, -1.04, 5.98),
            (0.08, 0.03, 0.10),
            red,
            rotation=(0.0, 0.0, math.radians(side * 12)),
            bevel=0.015,
        )

    bevelled_box(
        "AbPlate",
        (0.0, -0.52, 5.82),
        (1.12, 0.44, 0.70),
        dark,
        rotation=(math.radians(-6), 0.0, 0.0),
        bevel=0.08,
        bevel_segments=3,
        subsurf=1,
    )
    bevelled_box(
        "PelvisMain",
        (0.0, -0.12, 4.76),
        (1.48, 1.00, 0.86),
        white,
        bevel=0.10,
        bevel_segments=4,
        subsurf=1,
    )
    bevelled_box(
        "PelvisFrontSkirt",
        (0.0, -1.04, 4.78),
        (0.92, 0.14, 0.52),
        white,
        rotation=(math.radians(-8), 0.0, 0.0),
        bevel=0.04,
    )
    for side in (-1, 1):
        bevelled_box(
            f"HipPod_{side}",
            (side * 1.44, -0.02, 4.64),
            (0.40, 0.76, 0.56),
            white,
            rotation=(0.0, 0.0, math.radians(side * 6)),
            bevel=0.05,
            subsurf=1,
        )
        cylinder(
            f"HipJoint_{side}",
            (side * 0.94, 0.12, 4.20),
            0.26,
            0.30,
            gray,
            rotation=(0.0, math.radians(90), 0.0),
            vertices=22,
            bevel=0.03,
        )

    bevelled_box(
        "BackpackCore",
        (0.0, 1.14, 6.86),
        (1.32, 0.42, 1.18),
        dark,
        bevel=0.08,
        bevel_segments=3,
        subsurf=1,
    )
    for side in (-1, 1):
        bevelled_box(
            f"BackpackWing_{side}",
            (side * 1.18, 1.10, 6.96),
            (0.52, 0.52, 1.02),
            white,
            rotation=(0.0, math.radians(side * 10), math.radians(side * 8)),
            bevel=0.05,
            bevel_segments=3,
            subsurf=1,
        )
        cylinder(
            f"Thruster_{side}",
            (side * 0.76, 1.54, 6.18),
            0.24,
            0.46,
            gray,
            rotation=(math.radians(90), 0.0, 0.0),
            vertices=22,
            bevel=0.03,
        )
        cylinder(
            f"ThrusterGlow_{side}",
            (side * 0.76, 1.74, 6.18),
            0.12,
            0.06,
            blue,
            rotation=(math.radians(90), 0.0, 0.0),
            vertices=18,
            bevel=0.015,
        )


def add_arm(side: str, sign: int, materials: dict[str, bpy.types.Material]) -> None:
    white = materials["white"]
    dark = materials["dark"]
    gray = materials["gray"]
    red = materials["red"]

    shoulder_x = sign * 2.22
    cylinder(
        f"{side}_ShoulderJoint",
        (shoulder_x, 0.0, 7.06),
        0.34,
        0.42,
        gray,
        rotation=(0.0, math.radians(90), 0.0),
        vertices=24,
        bevel=0.04,
    )
    bevelled_box(
        f"{side}_ShoulderArmor",
        (shoulder_x + sign * 0.18, -0.10, 7.08),
        (0.84, 0.94, 0.82),
        white,
        rotation=(math.radians(4), math.radians(sign * -8), math.radians(sign * -8)),
        bevel=0.08,
        bevel_segments=4,
        subsurf=1,
    )
    bevelled_box(
        f"{side}_UpperArm",
        (shoulder_x + sign * 0.22, 0.00, 5.92),
        (0.52, 0.56, 0.74),
        dark,
        rotation=(math.radians(2), 0.0, math.radians(sign * -4)),
        bevel=0.06,
        bevel_segments=3,
        subsurf=1,
    )
    cylinder(
        f"{side}_Elbow",
        (shoulder_x + sign * 0.18, 0.00, 5.04),
        0.28,
        0.34,
        gray,
        rotation=(0.0, math.radians(90), 0.0),
        vertices=22,
        bevel=0.03,
    )
    bevelled_box(
        f"{side}_Forearm",
        (shoulder_x + sign * 0.16, -0.04, 4.18),
        (0.60, 0.64, 0.82),
        white,
        rotation=(math.radians(-4), 0.0, math.radians(sign * -3)),
        bevel=0.06,
        bevel_segments=4,
        subsurf=1,
    )
    bevelled_box(
        f"{side}_ForearmInset",
        (shoulder_x + sign * 0.28, -0.56, 4.12),
        (0.22, 0.10, 0.42),
        dark,
        rotation=(math.radians(-6), 0.0, math.radians(sign * -2)),
        bevel=0.03,
    )
    cylinder(
        f"{side}_Wrist",
        (shoulder_x + sign * 0.12, -0.02, 3.32),
        0.22,
        0.26,
        gray,
        rotation=(0.0, math.radians(90), 0.0),
        vertices=20,
        bevel=0.025,
    )

    if side == "R":
        bevelled_box(
            "R_HandPalm",
            (shoulder_x + sign * 0.10, -0.02, 2.92),
            (0.44, 0.46, 0.30),
            dark,
            bevel=0.05,
            bevel_segments=3,
            subsurf=1,
        )
        for idx, (dx, dz) in enumerate(((0.14, -0.22), (0.02, -0.08), (-0.08, 0.06))):
            bevelled_box(
                f"R_Finger_{idx}",
                (shoulder_x + sign * dx, -0.42, 2.60 + dz),
                (0.10, 0.10, 0.26),
                white,
                rotation=(math.radians(-18), 0.0, math.radians(sign * 3)),
                bevel=0.03,
            )
        bevelled_box(
            "R_Thumb",
            (shoulder_x + sign * 0.46, -0.10, 2.78),
            (0.09, 0.10, 0.22),
            white,
            rotation=(math.radians(26), math.radians(sign * -18), math.radians(sign * 26)),
            bevel=0.03,
        )
    else:
        bevelled_box(
            "L_HandPalm",
            (shoulder_x + sign * 0.10, -0.02, 2.96),
            (0.44, 0.46, 0.30),
            dark,
            bevel=0.05,
            bevel_segments=3,
            subsurf=1,
        )
        for idx, dz in enumerate((-0.16, 0.02, 0.20)):
            bevelled_box(
                f"L_Finger_{idx}",
                (shoulder_x + sign * 0.10, -0.44, 2.64 + dz),
                (0.10, 0.10, 0.24),
                white,
                rotation=(math.radians(-12), 0.0, math.radians(sign * -4)),
                bevel=0.03,
            )
        bevelled_box(
            "L_Thumb",
            (shoulder_x + sign * 0.44, -0.08, 2.86),
            (0.09, 0.10, 0.20),
            white,
            rotation=(math.radians(18), math.radians(sign * 16), math.radians(sign * -18)),
            bevel=0.03,
        )

    bevelled_box(
        f"{side}_ShoulderMark",
        (shoulder_x + sign * 0.64, -0.64, 7.02),
        (0.06, 0.02, 0.18),
        red,
        rotation=(math.radians(2), 0.0, math.radians(sign * -12)),
        bevel=0.012,
    )


def add_leg(side: str, sign: int, materials: dict[str, bpy.types.Material]) -> None:
    white = materials["white"]
    dark = materials["dark"]
    gray = materials["gray"]

    hip_x = sign * 0.96
    bevelled_box(
        f"{side}_ThighCore",
        (hip_x, 0.00, 3.32),
        (0.52, 0.54, 0.92),
        dark,
        rotation=(math.radians(3), 0.0, math.radians(sign * -2)),
        bevel=0.06,
        bevel_segments=3,
        subsurf=1,
    )
    bevelled_box(
        f"{side}_ThighArmor",
        (hip_x, -0.58, 3.30),
        (0.40, 0.16, 0.72),
        white,
        rotation=(math.radians(-4), 0.0, math.radians(sign * -4)),
        bevel=0.04,
        subsurf=1,
    )
    cylinder(
        f"{side}_Knee",
        (hip_x, -0.02, 2.32),
        0.28,
        0.34,
        gray,
        rotation=(0.0, math.radians(90), 0.0),
        vertices=22,
        bevel=0.03,
    )
    bevelled_box(
        f"{side}_Shin",
        (hip_x, -0.02, 1.42),
        (0.56, 0.62, 0.90),
        white,
        rotation=(math.radians(-6), 0.0, math.radians(sign * -2)),
        bevel=0.06,
        bevel_segments=4,
        subsurf=1,
    )
    bevelled_box(
        f"{side}_ShinInset",
        (hip_x, -0.66, 1.42),
        (0.24, 0.10, 0.54),
        dark,
        rotation=(math.radians(-8), 0.0, 0.0),
        bevel=0.03,
    )
    cylinder(
        f"{side}_Ankle",
        (hip_x, -0.02, 0.46),
        0.24,
        0.28,
        gray,
        rotation=(0.0, math.radians(90), 0.0),
        vertices=20,
        bevel=0.025,
    )
    bevelled_box(
        f"{side}_FootBase",
        (hip_x, 0.06, -0.04),
        (0.64, 0.98, 0.24),
        dark,
        rotation=(math.radians(4), 0.0, 0.0),
        bevel=0.05,
        bevel_segments=3,
        subsurf=1,
    )
    bevelled_box(
        f"{side}_FootArmor",
        (hip_x, -0.72, 0.10),
        (0.56, 0.26, 0.18),
        white,
        rotation=(math.radians(-12), 0.0, 0.0),
        bevel=0.03,
    )
    bevelled_box(
        f"{side}_Heel",
        (hip_x, 0.72, 0.02),
        (0.28, 0.24, 0.18),
        white,
        rotation=(math.radians(8), 0.0, 0.0),
        bevel=0.03,
    )


def add_logo_decal(materials: dict[str, bpy.types.Material]) -> None:
    dark = materials["dark"]
    for sign in (-1, 1):
        bevelled_box(
            f"ChestBadgeWing_{sign}",
            (sign * 0.14, -1.00, 5.98),
            (0.05, 0.01, 0.08),
            dark,
            rotation=(0.0, 0.0, math.radians(sign * -36)),
            bevel=0.008,
        )
    bevelled_box(
        "ChestBadgeStem",
        (0.0, -1.00, 5.90),
        (0.02, 0.01, 0.10),
        dark,
        bevel=0.008,
    )


def add_stage(materials: dict[str, bpy.types.Material]) -> None:
    floor = materials["floor"]
    ring = materials["ring"]
    blue = materials["blue"]

    cylinder("DisplayBase", (0.0, 0.0, -0.34), 4.40, 0.30, floor, vertices=72, bevel=0.08, export=False)
    cylinder("DisplayTop", (0.0, 0.0, -0.10), 3.90, 0.10, ring, vertices=72, bevel=0.03, export=False)
    for angle in range(0, 360, 30):
        rad = math.radians(angle)
        bevelled_box(
            f"DisplayTick_{angle}",
            (math.cos(rad) * 3.60, math.sin(rad) * 3.60, -0.02),
            (0.18, 0.05, 0.025),
            blue if angle % 60 == 0 else ring,
            rotation=(0.0, 0.0, rad),
            bevel=0.01,
            export=False,
        )

    bpy.ops.mesh.primitive_plane_add(size=40, location=(0.0, 0.0, -0.50))
    plane = bpy.context.object
    plane.name = "GroundPlane"
    assign_material(plane, floor)



def configure_camera_and_lights() -> None:
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (8.0, -10.8, 7.6)
    camera.rotation_euler = (math.radians(73), 0.0, math.radians(34))
    camera_data.lens = 70
    camera_data.sensor_width = 36
    point_at(camera, (0.0, -0.1, 5.4))
    bpy.context.scene.camera = camera

    add_area_light("KeyLight", (5.8, -6.6, 12.8), 2100, 5.4, (1.0, 0.96, 0.92), (0.0, -0.2, 5.0))
    add_area_light("FillLight", (-7.2, -3.8, 7.2), 900, 4.4, (0.62, 0.76, 1.0), (0.0, 0.0, 5.0))
    add_area_light("RimLight", (2.8, 6.8, 11.8), 1650, 4.0, (0.54, 0.80, 1.0), (0.0, 0.2, 5.8))
    add_area_light("KickLight", (-2.6, -7.8, 3.8), 700, 2.5, (1.0, 0.48, 0.42), (0.0, -0.4, 4.5))



def configure_render(args: argparse.Namespace, output_dir: Path) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.transmission_bounces = 4
    scene.cycles.transparent_max_bounces = 6
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.filepath = str(output_dir / "mech_render.png")

    world = scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.92, 0.94, 0.98, 1.0)
        bg.inputs["Strength"].default_value = 0.72

    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except (TypeError, ValueError):
        print("AgX look preset unavailable; using default look.")



def export_assets(output_dir: Path) -> None:
    blend_path = output_dir / "mid_poly_mech.blend"
    glb_path = output_dir / "mid_poly_mech.glb"

    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    bpy.ops.object.select_all(action="DESELECT")
    existing = {obj.name for obj in bpy.context.view_layer.objects}
    selected = []
    for obj in EXPORT_OBJECTS:
        if obj.name in existing:
            obj.select_set(True)
            selected.append(obj)

    if selected:
        bpy.context.view_layer.objects.active = selected[0]

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
        "Stylized Hero Chibi Mech\n"
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
        "white": create_material(
            "WhiteArmor",
            (0.90, 0.91, 0.94, 1.0),
            metallic=0.24,
            roughness=0.28,
            coat=0.18,
            coat_roughness=0.12,
            specular_ior_level=0.6,
        ),
        "gray": create_material(
            "GrayArmor",
            (0.46, 0.50, 0.58, 1.0),
            metallic=0.68,
            roughness=0.30,
            coat=0.08,
            coat_roughness=0.08,
        ),
        "dark": create_material(
            "DarkFrame",
            (0.09, 0.11, 0.14, 1.0),
            metallic=0.82,
            roughness=0.24,
            coat=0.04,
        ),
        "visor": create_material(
            "VisorGlass",
            (0.10, 0.14, 0.19, 1.0),
            metallic=0.0,
            roughness=0.06,
            transmission=0.04,
            coat=0.28,
            coat_roughness=0.02,
            emission_color=(0.28, 0.40, 0.52, 1.0),
            emission_strength=0.4,
        ),
        "blue": create_material(
            "BlueGlow",
            (0.16, 0.40, 0.62, 1.0),
            metallic=0.08,
            roughness=0.16,
            emission_color=(0.55, 0.90, 1.0, 1.0),
            emission_strength=7.5,
        ),
        "red": create_material(
            "RedAccent",
            (0.90, 0.16, 0.18, 1.0),
            metallic=0.22,
            roughness=0.24,
            emission_color=(1.0, 0.12, 0.12, 1.0),
            emission_strength=1.1,
        ),
        "floor": create_material(
            "Floor",
            (0.78, 0.81, 0.86, 1.0),
            metallic=0.18,
            roughness=0.46,
        ),
        "ring": create_material(
            "Ring",
            (0.42, 0.48, 0.56, 1.0),
            metallic=0.74,
            roughness=0.30,
        ),
    }

    add_head(materials)
    add_torso(materials)
    add_arm("L", -1, materials)
    add_arm("R", 1, materials)
    add_leg("L", -1, materials)
    add_leg("R", 1, materials)
    add_logo_decal(materials)
    add_stage(materials)
    configure_camera_and_lights()
    configure_render(args, output_dir)

    bpy.context.scene.render.filepath = str(output_dir / "mech_render.png")
    bpy.ops.render.render(write_still=True)
    export_assets(output_dir)
    write_info(output_dir, args)


if __name__ == "__main__":
    main()

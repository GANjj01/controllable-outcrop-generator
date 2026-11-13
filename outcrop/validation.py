"""Field-addressable validation, shared by JSON boundaries and array consumers."""

from dataclasses import MISSING, fields, is_dataclass
import math
from types import UnionType
from typing import Literal, get_args, get_origin, get_type_hints

import numpy as np

from .contracts import ResolvedConfig, SurfaceMesh, RandomnessSpec


class ContractError(ValueError):
    def __init__(self, field: str, message: str):
        self.field = field or "$"
        self.message = message
        super().__init__(f"{self.field}: {message}")


def require(condition: bool, field: str, message: str) -> None:
    if not condition:
        raise ContractError(field, message)


def decode(value, annotation, path="", *, complete=False):
    """Strict decoder for the small set of types used in our JSON dataclasses.

    Unknown fields fail so typos cannot silently fall back to defaults."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is UnionType:
        if value is None and type(None) in args:
            return None
        concrete = next(a for a in args if a is not type(None))
        return decode(value, concrete, path, complete=complete)
    if origin is Literal:
        require(any(type(value) is type(a) and value == a for a in args), path, f"expected one of {args!r}")
        return value
    if is_dataclass(annotation):
        require(isinstance(value, dict), path, "expected an object")
        declared = {f.name: f for f in fields(annotation)}
        child = lambda key: f"{path}.{key}" if path else key
        if "schema_version" in value and "schema_version" in declared:
            decode(value["schema_version"], Literal[1], child("schema_version"))
        for key in value:
            require(key in declared, child(key), "unknown field")
        hints = get_type_hints(annotation)
        parsed = {}
        for name, f in declared.items():
            if name in value:
                parsed[name] = decode(value[name], hints[name], child(name), complete=complete)
            elif complete or (f.default is MISSING and f.default_factory is MISSING):
                raise ContractError(child(name), "required field is missing")
        return annotation(**parsed)
    if origin is tuple:
        require(isinstance(value, (list, tuple)), path, "expected an array")
        variable = len(args) == 2 and args[1] is Ellipsis
        require(variable or len(value) == len(args), path, f"expected {len(args)} items")
        return tuple(decode(v, args[0] if variable else args[i], f"{path}[{i}]", complete=complete) for i, v in enumerate(value))
    if origin is dict:
        require(isinstance(value, dict), path, "expected an object")
        return {decode(k, args[0], path): decode(v, args[1], f"{path}.{k}", complete=complete) for k, v in value.items()}
    if annotation is float:
        require(type(value) in (int, float), path, "expected a number, not a string or boolean")
        try:
            number = float(value)
        except OverflowError:
            raise ContractError(path, "number is out of range") from None
        require(math.isfinite(number), path, "expected a finite number")
        return number
    if annotation is int:
        require(type(value) is int, path, "expected an integer, not a boolean")
        return value
    if annotation is str:
        require(isinstance(value, str) and bool(value.strip()), path, "expected a nonempty string")
        return value
    raise TypeError(f"unsupported internal contract annotation: {annotation!r}")


def unique(items, attribute, path):
    seen = set()
    for i, item in enumerate(items):
        value = getattr(item, attribute)
        require(value not in seen, f"{path}[{i}].{attribute}", f"duplicate value {value!r}")
        seen.add(value)


def positive(value, path):
    require(value > 0, path, "must be greater than zero")


def interval(value, low, high, path, *, upper_open=False):
    require(low <= value and (value < high if upper_open else value <= high), path,
            f"must be in [{low}, {high}{')' if upper_open else ']'}")


def validate_randomness(spec: RandomnessSpec):
    require(type(spec.root_seed) is int and spec.root_seed >= 0, "randomness.root_seed", "expected a nonnegative integer")
    decode(spec.sample_id, str, "randomness.sample_id")
    require(spec.algorithm == "sha256-seedsequence-pcg64-v1", "randomness.algorithm", "unsupported RNG algorithm")
    require(isinstance(spec.stream_seeds, dict), "randomness.stream_seeds", "expected an object")
    for stream, seed in spec.stream_seeds.items():
        decode(stream, str, "randomness.stream_seeds")
        require(type(seed) is int and seed >= 0, f"randomness.stream_seeds.{stream}", "expected a nonnegative integer")


def validate_config(config: ResolvedConfig):
    """Semantic/reference checks after decode has checked types and versions."""
    validate_randomness(config.randomness)
    catalog = config.lithologies.lithologies
    require(bool(catalog), "lithologies.lithologies", "must not be empty")
    unique(catalog, "lithology_id", "lithologies.lithologies")
    unique(catalog, "code", "lithologies.lithologies")
    lithologies = {v.lithology_id for v in catalog}
    for i, item in enumerate(catalog):
        interval(item.code, 1, 65534, f"lithologies.lithologies[{i}].code")

    model = config.geology
    interval(model.strike_deg, 0, 360, "geology.strike_deg", upper_open=True)
    interval(model.dip_deg, 0, 90, "geology.dip_deg")
    for i, (low, high) in enumerate(zip(model.domain.min_m, model.domain.max_m)):
        require(low < high, f"geology.domain.max_m[{i}]", "must exceed min_m on this axis")
    require(bool(model.layers), "geology.layers", "must not be empty")
    unique(model.layers, "layer_id", "geology.layers")
    for i, layer in enumerate(model.layers):
        prefix = f"geology.layers[{i}]"
        positive(layer.layer_id, prefix + ".layer_id")
        positive(layer.thickness_m, prefix + ".thickness_m")
        require(layer.lithology_id in lithologies, prefix + ".lithology_id", "unknown lithology reference")

    unique(config.materials, "material_id", "materials")
    for i, material in enumerate(config.materials):
        prefix = f"materials[{i}]"
        require(material.lithology_id in lithologies, prefix + ".lithology_id", "unknown lithology reference")
        positive(material.tile_size_m, prefix + ".tile_size_m")
        interval(material.roughness_default, 0, 1, prefix + ".roughness_default")
        require(material.normal_strength >= 0, prefix + ".normal_strength", "must be nonnegative")

    scene = config.scene
    require(scene.model_id == model.model_id, "scene.model_id", "must reference geology.model_id")
    require(not scene.parts or scene.mesh_file is not None, "scene.mesh_file", "required when parts are declared")
    require(len(scene.transform) == 4, "scene.transform", "expected a 4 by 4 affine matrix")
    require(scene.transform[3] == (0., 0., 0., 1.), "scene.transform[3]", "expected affine bottom row [0, 0, 0, 1]")
    require(abs(np.linalg.det(np.asarray(scene.transform)[:3, :3])) > 1e-12,
            "scene.transform", "must be invertible")
    unique(scene.parts, "part_id", "scene.parts")
    unique(scene.parts, "node_id", "scene.parts")
    layers = {v.layer_id: v for v in model.layers}
    materials = {v.material_id: v for v in config.materials}
    for i, part in enumerate(scene.parts):
        prefix = f"scene.parts[{i}]"
        require(part.layer_id in layers, prefix + ".layer_id", "unknown layer reference")
        require(part.material_id in materials, prefix + ".material_id", "unknown material reference")
        require(materials[part.material_id].lithology_id == layers[part.layer_id].lithology_id,
                prefix + ".material_id", "material lithology must match the layer")

    capture = config.capture
    require(capture.scene_ref == scene.scene_id, "capture.scene_ref", "must reference scene.scene_id")
    require(bool(capture.shots), "capture.shots", "must not be empty")
    unique(capture.shots, "shot_id", "capture.shots")
    for i, size in enumerate(capture.resolution):
        positive(size, f"capture.resolution[{i}]")
    require(bool(capture.passes), "capture.passes", "must not be empty")
    require(len(set(capture.passes)) == len(capture.passes), "capture.passes", "duplicate output pass")
    for i, shot in enumerate(capture.shots):
        prefix = f"capture.shots[{i}]"
        camera = shot.camera
        direction = np.subtract(camera.target_m, camera.position_m)
        require(np.linalg.norm(direction) > 0, prefix + ".camera.target_m", "must differ from position_m")
        require(np.linalg.norm(np.cross(direction, camera.up)) > 1e-12,
                prefix + ".camera.up", "must be nonzero and not parallel to the viewing direction")
        for name in ("focal_length_mm", "sensor_width_mm", "near_clip_m"):
            positive(getattr(camera, name), prefix + ".camera." + name)
        require(camera.far_clip_m > camera.near_clip_m, prefix + ".camera.far_clip_m", "must exceed near_clip_m")
        interval(shot.light.azimuth_deg, 0, 360, prefix + ".light.azimuth_deg", upper_open=True)
        interval(shot.light.elevation_deg, -90, 90, prefix + ".light.elevation_deg")
        require(shot.light.intensity_lux >= 0, prefix + ".light.intensity_lux", "must be nonnegative")
        for j, channel in enumerate(shot.light.color_linear_rgb):
            interval(channel, 0, 1, prefix + f".light.color_linear_rgb[{j}]")


def validate_mesh(mesh: SurfaceMesh):
    """Validate array shape/ranges only, not geological correctness or topology."""
    decode(mesh.schema_version, Literal[1], "mesh.schema_version")
    decode(mesh.coordinate_system, Literal["ENU"], "mesh.coordinate_system")
    decode(mesh.length_unit, Literal["m"], "mesh.length_unit")

    def array(name, shape, integer=False):
        value = getattr(mesh, name)
        path = "mesh." + name
        require(isinstance(value, np.ndarray), path, "expected a NumPy array")
        require(value.ndim == len(shape) and all(n is None or n == actual for n, actual in zip(shape, value.shape)),
                path, f"expected shape {shape}")
        require(value.dtype.kind in ("iu" if integer else "f"), path,
                "expected integer array" if integer else "expected floating-point array")
        require(bool(np.isfinite(value).all()), path, "must contain only finite values")
        return value

    vertices = array("vertices", (None, 3))
    faces = array("faces", (None, 3), True)
    n, m = len(vertices), len(faces)
    require(n > 0 and m > 0, "mesh.faces", "a materialized mesh must have vertices and faces")
    require(bool(((faces >= 0) & (faces < n)).all()), "mesh.faces", "vertex index out of range")
    normals = array("normals", (n, 3))
    require(bool(np.allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-6)), "mesh.normals", "expected unit normals")
    array("uv", (n, 2))
    layer = array("face_layer_id", (m,), True)
    lithology = array("face_lithology_code", (m,), True)
    part = array("face_part_id", (m,), True)
    require(bool((layer >= 1).all()), "mesh.face_layer_id", "layer IDs must be positive")
    require(bool(((lithology >= 1) & (lithology <= 65534)).all()), "mesh.face_lithology_code", "expected lithology category codes 1..65534")
    decode(mesh.part_ids, tuple[str, ...], "mesh.part_ids")
    require(len(set(mesh.part_ids)) == len(mesh.part_ids), "mesh.part_ids", "duplicate part IDs")
    require(bool(((part >= 0) & (part < len(mesh.part_ids))).all()), "mesh.face_part_id", "part index out of range")
    require(isinstance(mesh.vertex_attributes, dict), "mesh.vertex_attributes", "expected attribute mapping")
    for name, value in mesh.vertex_attributes.items():
        path = f"mesh.vertex_attributes.{name}"
        decode(name, str, path)
        require(isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == n,
                path, "first dimension must match vertices")
        require(value.dtype.kind == "f" and bool(np.isfinite(value).all()), path, "expected finite floating-point attributes")

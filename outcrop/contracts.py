"""Schema v1 data only. Validate at boundaries; no geological or renderer code.

Frozen dataclasses prevent field reassignment, not mutation of nested arrays/maps.
Consumers must treat validated objects as snapshots and use dataclasses.replace.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Vec3 = tuple[float, float, float]
Matrix4 = tuple[tuple[float, float, float, float], ...]
IDENTITY: Matrix4 = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))


@dataclass(frozen=True, slots=True, kw_only=True)
class Lithology:
    lithology_id: str
    code: int


@dataclass(frozen=True, slots=True, kw_only=True)
class LithologyCatalog:
    lithologies: tuple[Lithology, ...]
    schema_version: Literal[1] = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class Domain:
    min_m: Vec3
    max_m: Vec3


@dataclass(frozen=True, slots=True, kw_only=True)
class Layer:
    layer_id: int
    lithology_id: str
    thickness_m: float


@dataclass(frozen=True, slots=True, kw_only=True)
class GeologicalModel:
    model_id: str
    domain: Domain
    reference_point_m: Vec3
    layers: tuple[Layer, ...]
    strike_deg: float = 0.
    dip_deg: float = 0.
    generator_version: str = "strata-v1"
    coordinate_system: Literal["ENU"] = "ENU"
    length_unit: Literal["m"] = "m"
    schema_version: Literal[1] = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class SurfaceMesh:
    """Array protocol only; seam vertices are duplicated to carry UV/normals.

    face_part_id indexes part_ids; it is not a renderer-generated object ID.
    No mesh generation, serialization or topology operations are implemented.
    """

    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]
    normals: NDArray[np.float64]
    uv: NDArray[np.float64]
    face_layer_id: NDArray[np.int64]
    face_lithology_code: NDArray[np.uint16]
    face_part_id: NDArray[np.int64]
    part_ids: tuple[str, ...]
    vertex_attributes: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    schema_version: Literal[1] = 1
    coordinate_system: Literal["ENU"] = "ENU"
    length_unit: Literal["m"] = "m"


@dataclass(frozen=True, slots=True, kw_only=True)
class TextureSet:
    albedo: str | None = None
    normal: str | None = None
    roughness: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class MaterialAsset:
    material_id: str
    lithology_id: str
    textures: TextureSet = field(default_factory=TextureSet)
    shader_family: Literal["rock_pbr_v1"] = "rock_pbr_v1"
    tile_size_m: float = 1.
    normal_convention: Literal["directx", "opengl"] = "directx"
    roughness_default: float = 0.8
    normal_strength: float = 1.
    schema_version: Literal[1] = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class MeshPart:
    part_id: str
    node_id: str
    layer_id: int
    material_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SceneSpec:
    scene_id: str
    model_id: str
    mesh_file: str | None = None
    parts: tuple[MeshPart, ...] = ()
    transform: Matrix4 = IDENTITY
    schema_version: Literal[1] = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class CameraSpec:
    position_m: Vec3
    target_m: Vec3
    up: Vec3 = (0., 0., 1.)
    focal_length_mm: float = 35.
    sensor_width_mm: float = 36.
    near_clip_m: float = 0.1
    far_clip_m: float = 1000.


@dataclass(frozen=True, slots=True, kw_only=True)
class DirectionalLightSpec:
    azimuth_deg: float
    elevation_deg: float
    intensity_lux: float
    color_linear_rgb: Vec3 = (1., 1., 1.)


@dataclass(frozen=True, slots=True, kw_only=True)
class Shot:
    shot_id: str
    camera: CameraSpec
    light: DirectionalLightSpec


@dataclass(frozen=True, slots=True, kw_only=True)
class CapturePlan:
    scene_ref: str
    shots: tuple[Shot, ...]
    resolution: tuple[int, int] = (640, 480)
    passes: tuple[Literal["rgb", "depth", "lithology"], ...] = ("rgb", "depth", "lithology")
    generator_version: str = "capture-v1"
    schema_version: Literal[1] = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class RandomnessSpec:
    root_seed: int
    sample_id: str
    stream_seeds: dict[str, int] = field(default_factory=dict)
    algorithm: Literal["sha256-seedsequence-pcg64-v1"] = "sha256-seedsequence-pcg64-v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class Provenance:
    package_version: str
    python_version: str
    numpy_version: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedConfig:
    randomness: RandomnessSpec
    lithologies: LithologyCatalog
    geology: GeologicalModel
    materials: tuple[MaterialAsset, ...]
    scene: SceneSpec
    capture: CapturePlan
    provenance: Provenance
    schema_version: Literal[1] = 1
    kind: Literal["resolved"] = "resolved"

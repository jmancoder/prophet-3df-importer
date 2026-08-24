from dataclasses import dataclass
from typing import NamedTuple

from mathutils import Matrix

from .binary_reader import BinaryReader

HEADER_SIZE = 412


class MeshInfo3DF(NamedTuple):
    vertex_bitmask: int
    unk_int: int
    unk_float: float
    vertices_off: int
    faces_off: int


class FaceGroup3DF(NamedTuple):
    face_type: int
    face_idx_count: int
    face_flags: int


@dataclass(frozen=True, slots=True)
class Node3DF:
    name: str
    type_id: int
    child_indexes: list[int]
    transform_type: int
    transform: Matrix


@dataclass(frozen=True, slots=True)
class BoneNode3DF(Node3DF):
    unk_floats: list[float]


@dataclass(frozen=True, slots=True)
class MeshNode3DF(Node3DF):
    vertex_count: int
    face_idx_count: int
    face_groups: list[FaceGroup3DF]


class Material3DF(NamedTuple):
    name: str
    flags_a: int
    unk_count: int
    unk_off: int
    flags_b: int


def read_mesh_info(bs: BinaryReader) -> MeshInfo3DF:
    vertex_bitmask = bs.read_uint32()
    unk_int = bs.read_uint32()
    unk_float = bs.read_float()
    vertices_off = bs.read_uint32()
    faces_off = bs.read_uint32()

    return MeshInfo3DF(
        vertex_bitmask,
        unk_int,
        unk_float,
        vertices_off,
        faces_off,
    )


def read_face_group(bs: BinaryReader) -> FaceGroup3DF:
    face_type = bs.read_uint16()
    face_count = bs.read_uint16()
    face_flags = bs.read_uint32()
    bs.seek(8, 1)

    return FaceGroup3DF(
        face_type,
        face_count,
        face_flags,
    )


def read_node(bs: BinaryReader) -> Node3DF:
    node_name = bs.read_string_block(16)
    node_type = bs.read_uint32()
    bs.read_int32()
    bs.read_int32()
    child_index_count = bs.read_int32()
    unk_id = bs.read_int32()
    child_index_off = bs.read_uint32()
    bs.read_int32()
    unk_vec_0 = bs.read_vec3f()
    unk_vec_1 = bs.read_vec3f()
    unk_floats_off = bs.read_uint32()
    bs.read_int32()
    bs.read_int32()
    transform_type = bs.read_uint32()
    transform = bs.read_loc_rot_scale()
    bs.read_vec3f()
    bs.read_int32()
    bs.read_float()
    bounds_min = bs.read_vec3f()
    bounds_max = bs.read_vec3f()
    bs.seek(68, 1)
    face_groups_off = bs.read_uint32()
    bs.seek(28, 1)

    # Read child indexes
    if child_index_count > 0:
        cur_off = bs.tell()
        bs.seek(child_index_off - HEADER_SIZE)
        child_indexes = [bs.read_uint32() for _ in range(child_index_count)]
        bs.seek(cur_off)
    else:
        child_indexes = []

    match node_type:
        case 0:
            vertex_count = bs.read_uint32()
            face_idx_count = bs.read_uint32()
            face_groups_count = bs.read_uint32()
            bs.seek(92, 1)

            # Read face groups
            cur_off = bs.tell()
            bs.seek(face_groups_off - HEADER_SIZE)
            face_groups = [read_face_group(bs) for _ in range(face_groups_count)]
            bs.seek(cur_off)

            return MeshNode3DF(
                node_name,
                node_type,
                child_indexes,
                transform_type,
                transform,
                vertex_count,
                face_idx_count,
                face_groups,
            )
        case 1:
            unk_floats = [bs.read_float() for _ in range(13)]
            bs.seek(52, 1)

            return BoneNode3DF(
                node_name,
                node_type,
                child_indexes,
                transform_type,
                transform,
                unk_floats,
            )
        case _:
            bs.seek(104, 1)

            return Node3DF(
                node_name,
                node_type,
                child_indexes,
                transform_type,
                transform,
            )


def read_material(bs: BinaryReader) -> Material3DF:
    name = bs.read_string_block(16)
    flags_a = bs.read_uint32()
    unk_count = bs.read_uint32()
    unk_off = bs.read_uint32()
    flags_b = bs.read_uint32()
    bs.read_uint32()
    bs.read_float()
    bs.read_int32()
    bs.read_int32()
    bs.read_int32()
    bs.read_int32()

    return Material3DF(
        name,
        flags_a,
        unk_count,
        unk_off,
        flags_b,
    )

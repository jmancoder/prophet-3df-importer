from dataclasses import dataclass
from typing import NamedTuple

from mathutils import Matrix
import numpy as np
import numpy.typing as npt

from .binary_reader import BinaryReader

HEADER_SIZE = 176


class Header3DF(NamedTuple):
    compress_mode: int
    node_chunk_size: int
    mesh_chunk_size: int
    texture_chunk_size: int
    material_count: int
    material_off: int
    node_count: int
    node_off: int


class Material3DF(NamedTuple):
    name: str
    unk_count: int
    unk_off: int
    color: tuple[int, int, int, int]


class FaceGroup3DF(NamedTuple):
    face_type: int
    face_idx_count: int
    bone_indexes: tuple[int, int, int, int]


class BoneGroup3DF(NamedTuple):
    bone_points: npt.NDArray


@dataclass(frozen=True, slots=True)
class Node3DF:
    name: str
    type_id: int
    internal_index: int
    child_indexes: list[int]
    transform_type: int
    transform: Matrix


@dataclass(frozen=True, slots=True)
class BoneNode3DF(Node3DF):
    unk_floats: list[float]
    bone_groups: list[BoneGroup3DF]


@dataclass(frozen=True, slots=True)
class MeshNode3DF(Node3DF):
    vertex_count: int
    face_idx_count: int
    face_groups: list[FaceGroup3DF]


class MeshInfo3DF(NamedTuple):
    vertices_off: int
    faces_off: int


def read_header(bs: BinaryReader) -> Header3DF:
    bs.read_uint32()
    compress_mode = bs.read_uint32()
    node_chunk_size = bs.read_uint32()
    mesh_chunk_size = bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    texture_chunk_size = bs.read_uint32()
    material_count = bs.read_uint32()
    material_off = bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    nodes_count = bs.read_uint32()
    nodes_off = bs.read_uint32()

    return Header3DF(
        compress_mode,
        node_chunk_size,
        mesh_chunk_size,
        texture_chunk_size,
        material_count,
        material_off,
        nodes_count,
        nodes_off,
    )


def read_material(bs: BinaryReader) -> Material3DF:
    name = bs.read_string_block(12)
    unk_count = bs.read_uint32()
    unk_off = bs.read_uint32()
    color = bs.read_vec4B()
    bs.read_uint32()
    bs.read_float()
    bs.read_float()
    bs.read_int32()
    bs.read_int32()
    bs.read_int32()
    bs.read_int32()
    bs.seek(44, 1)

    return Material3DF(
        name,
        unk_count,
        unk_off,
        color,
    )


def read_face_group(bs: BinaryReader) -> FaceGroup3DF:
    face_type = bs.read_uint16()
    face_count = bs.read_uint16()
    bone_indexes = bs.read_vec4B()
    bs.seek(8, 1)

    return FaceGroup3DF(
        face_type,
        face_count,
        bone_indexes,
    )


def read_bone_group(bs: BinaryReader) -> BoneGroup3DF:
    bone_group_start = bs.tell()
    bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    bone_point_count = bs.read_uint32()
    bone_point_off = bs.read_uint32()

    bs.seek(bone_point_off - HEADER_SIZE)
    bone_point_dtype = np.dtype([("floats", np.float32, 5)])
    bone_points = np.frombuffer(
        bs.getbuffer(), bone_point_dtype, bone_point_count, bs.tell()
    )
    bs.seek(bone_group_start)

    return BoneGroup3DF(bone_points)


def read_node(bs: BinaryReader) -> Node3DF:
    node_name = bs.read_string_block(12)
    node_type = bs.read_uint32()
    bs.read_int32()
    child_index_count = bs.read_int32()
    internal_idx = bs.read_int32()
    child_index_off = bs.read_uint32()
    bs.seek(124, 1)
    unk_floats_off = bs.read_uint32()
    bone_group_count = bs.read_uint32()
    bone_group_off = bs.read_uint32()
    transform_type = bs.read_uint32()
    transform = bs.read_loc_rot_scale()
    bs.read_vec3f()
    bs.read_int32()
    bs.read_float()
    bs.seek(28, 1)
    face_group_off = bs.read_uint32()
    bs.seek(28, 1)

    # Read child indexes
    if child_index_count > 0:
        node_end_off = bs.tell()
        bs.seek(child_index_off - HEADER_SIZE)
        child_indexes = [bs.read_uint32() for _ in range(child_index_count)]
        bs.seek(node_end_off)
    else:
        child_indexes = []

    match node_type:
        case 0:
            vertex_count = bs.read_uint32()
            face_idx_count = bs.read_uint32()
            face_group_count = bs.read_uint32()
            bs.seek(92, 1)

            # Read face groups
            if face_group_off > 0:
                node_end_off = bs.tell()
                bs.seek(face_group_off - HEADER_SIZE)
                face_groups = [read_face_group(bs) for _ in range(face_group_count)]
                bs.seek(node_end_off)
            else:
                face_groups = []

            return MeshNode3DF(
                node_name,
                node_type,
                internal_idx,
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

            if bone_group_count > 0:
                node_end_off = bs.tell()
                bs.seek(bone_group_off - HEADER_SIZE)
                bone_groups = [read_bone_group(bs) for _ in range(bone_group_count)]
                bs.seek(node_end_off)
            else:
                bone_groups = []

            return BoneNode3DF(
                node_name,
                node_type,
                internal_idx,
                child_indexes,
                transform_type,
                transform,
                unk_floats,
                bone_groups,
            )
        case _:
            bs.seek(104, 1)

            return Node3DF(
                node_name,
                node_type,
                internal_idx,
                child_indexes,
                transform_type,
                transform,
            )


def read_mesh_info(bs: BinaryReader) -> MeshInfo3DF:
    bs.read_uint32()
    vertices_off = bs.read_uint32()
    faces_off = bs.read_uint32()

    return MeshInfo3DF(
        vertices_off,
        faces_off,
    )

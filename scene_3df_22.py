from dataclasses import dataclass
from typing import NamedTuple

from mathutils import Matrix
import numpy.typing as npt

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
    transform_type: int


@dataclass(frozen=True, slots=True)
class BoneNode3DF(Node3DF):
    unk_float: float
    unk_matrix: Matrix


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
    bs.seek(60, 1)
    transform_type = bs.read_uint32()
    bs.seek(148, 1)
    face_groups_off = bs.read_uint32()
    bs.seek(28, 1)

    if face_groups_off > 0:
        vertex_count = bs.read_uint32()
        face_idx_count = bs.read_uint32()
        face_groups_count = bs.read_uint32()
        bs.seek(92, 1)
        next_node_off = bs.tell()

        # Read face groups
        bs.seek(face_groups_off - HEADER_SIZE)
        face_groups = [
            read_face_group(bs)
            for _ in range(face_groups_count)
        ]

        bs.seek(next_node_off)

        return MeshNode3DF(
            node_name,
            node_type,
            transform_type,
            vertex_count,
            face_idx_count,
            face_groups,
        )
    else:
        unk_float = bs.read_float()
        unk_matrix = bs.read_mat43()
        bs.seek(52, 1)

        return BoneNode3DF(
            node_name,
            node_type,
            transform_type,
            unk_float,
            unk_matrix,
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

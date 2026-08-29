from dataclasses import dataclass
from typing import NamedTuple

from mathutils import Matrix

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


class Key3DF(NamedTuple):
    time: float
    value: float


class Track3DF(NamedTuple):
    type_id: int
    keys: list[Key3DF]


@dataclass(frozen=True, slots=True)
class Node3DF:
    name: str
    type_id: int
    internal_index: int
    child_indexes: list[int]
    transform_type: int
    transform: Matrix
    tracks: list[Track3DF]


@dataclass(frozen=True, slots=True)
class BoneNode3DF(Node3DF):
    unk_floats: list[float]
    tracks: list[Track3DF]


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


def read_keyframe(bs: BinaryReader) -> Key3DF:
    time = bs.read_float()
    value = bs.read_float()
    bs.read_float()
    bs.read_float()
    bs.read_float()
    bs.read_float()
    return Key3DF(time, value)


def read_track(bs: BinaryReader) -> Track3DF:
    type_id = bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    key_count = bs.read_uint32()
    key_off = bs.read_uint32()

    # Read keyframes
    track_end_off = bs.tell()
    bs.seek(key_off - HEADER_SIZE)
    keys = [read_keyframe(bs) for _ in range(key_count)]
    bs.seek(track_end_off)

    return Track3DF(type_id, keys)


def read_node(bs: BinaryReader) -> Node3DF:
    node_name = bs.read_string_block(12)
    node_type = bs.read_uint32()
    bs.read_int32()
    child_index_count = bs.read_int32()
    internal_idx = bs.read_int32()
    child_index_off = bs.read_uint32()
    bs.seek(124, 1)
    unk_floats_off = bs.read_uint32()
    track_count = bs.read_uint32()
    track_off = bs.read_uint32()
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

    # Read animation tracks
    if track_count > 0:
        node_end_off = bs.tell()
        bs.seek(track_off - HEADER_SIZE)
        tracks = [read_track(bs) for _ in range(track_count)]
        bs.seek(node_end_off)
    else:
        tracks = []

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
                tracks,
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
                internal_idx,
                child_indexes,
                transform_type,
                transform,
                tracks,
                unk_floats,
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
                tracks,
            )


def read_mesh_info(bs: BinaryReader) -> MeshInfo3DF:
    bs.read_uint32()
    vertices_off = bs.read_uint32()
    faces_off = bs.read_uint32()

    return MeshInfo3DF(
        vertices_off,
        faces_off,
    )

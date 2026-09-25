from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from .binary_reader import BinaryReader
from . import scene_3df_20


class Header3DF(NamedTuple):
    compress_mode: int
    node_chunk_size: int
    unk_chunk_size: int
    mesh_chunk_size: int
    texture_chunk_size: int
    material_count: int
    material_off: int
    texture_count: int
    node_count: int
    node_off: int


class MeshInfo3DF(NamedTuple):
    flags: int
    unk_int: int
    unk_float: float
    vertex_off: int
    face_off: int


def read_header(bs: BinaryReader) -> Header3DF:
    bs.read_uint32()
    bs.read_uint32()
    compress_mode = bs.read_uint32()
    node_chunk_size = bs.read_uint32()
    unk_chunk_size = bs.read_uint32()
    mesh_chunk_size = bs.read_uint32()
    bs.seek(124, 1)
    texture_chunk_size = bs.read_uint32()
    bs.seek(128, 1)
    material_count = bs.read_uint32()
    material_off = bs.read_uint32()
    texture_count = bs.read_uint32()
    bs.read_uint32()
    node_count = bs.read_uint32()
    node_off = bs.read_uint32()
    return Header3DF(
        compress_mode,
        node_chunk_size,
        unk_chunk_size,
        mesh_chunk_size,
        texture_chunk_size,
        material_count,
        material_off,
        texture_count,
        node_count,
        node_off,
    )


def read_node(bs: BinaryReader, header_size: int) -> scene_3df_20.Node3DF:
    name = bs.read_string_block(16)
    type_id = bs.read_uint32()
    flags = bs.read_uint32()
    bs.read_int32()
    child_index_count = bs.read_int32()
    internal_idx = bs.read_int32()
    child_index_off = bs.read_uint32()
    bs.read_int32()
    unk_vec_0 = bs.read_vec3f()
    unk_vec_1 = bs.read_vec3f()
    unk_point_off = bs.read_uint32()
    track_count = bs.read_uint32()
    track_off = bs.read_uint32()
    transform_type = bs.read_uint32()
    if transform_type == 0:
        transform = bs.read_loc_rot_scale()
        bs.seek(12, 1)
    else:
        transform = bs.read_matrix_3x4()
    bs.read_int32()
    bs.read_float()
    bounds_min = bs.read_vec3f()
    bounds_max = bs.read_vec3f()
    bs.seek(68, 1)
    face_group_off = bs.read_uint32()
    bs.seek(28, 1)

    # Read child indexes
    if child_index_count > 0:
        node_end_off = bs.tell()
        bs.seek(child_index_off - header_size)
        child_indexes = [bs.read_uint32() for _ in range(child_index_count)]
        bs.seek(node_end_off)
    else:
        child_indexes = []

    # Read animation tracks
    if track_count > 0:
        node_end_off = bs.tell()
        bs.seek(track_off - header_size)
        tracks = [scene_3df_20.read_track(bs, header_size) for _ in range(track_count)]
        bs.seek(node_end_off)
    else:
        tracks = []

    match type_id:
        case 0:
            vertex_count = bs.read_uint32()
            face_idx_count = bs.read_uint32()
            face_groups_count = bs.read_uint32()
            bs.seek(92, 1)

            # Read face groups
            if face_group_off > 0:
                node_end_off = bs.tell()
                bs.seek(face_group_off - header_size)
                face_groups = [
                    scene_3df_20.read_face_group(bs) for _ in range(face_groups_count)
                ]
                bs.seek(node_end_off)
            else:
                face_groups = []
            data = scene_3df_20.MeshNodeData3DF(
                vertex_count,
                face_idx_count,
                face_groups,
            )
        case 1:
            unk_float = bs.read_float()
            bone_transform = bs.read_matrix_3x4()
            bs.seek(52, 1)
            data = scene_3df_20.BoneNodeData3DF(
                unk_float,
                bone_transform,
            )
        case _:
            bs.seek(104, 1)
            data = None
    return scene_3df_20.Node3DF(
        name,
        type_id,
        flags,
        internal_idx,
        child_indexes,
        transform_type,
        transform,
        tracks,
        data,
    )


def read_mesh_info(bs: BinaryReader) -> MeshInfo3DF:
    flags = bs.read_uint32()
    unk_int = bs.read_uint32()
    unk_float = bs.read_float()
    vertex_off = bs.read_uint32()
    face_off = bs.read_uint32()
    return MeshInfo3DF(
        flags,
        unk_int,
        unk_float,
        vertex_off,
        face_off,
    )

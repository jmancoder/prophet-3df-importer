from typing import NamedTuple

from .binary_reader import BinaryReader
from . import scene_3df_22


class Header3DF(NamedTuple):
    unk_int_0: int
    unk_int_1: int
    compress_mode: int
    nodes_chunk_size: int
    unk_chunk_size: int
    mesh_chunk_size: int
    textures_chunk_size: int
    materials_count: int
    materials_off: int
    unk_int_2: int
    unk_int_3: int
    nodes_count: int
    nodes_off: int


def read_node(bs: BinaryReader) -> scene_3df_22.Node3DF:
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
    bs.seek(32, 1)

    # Read child indexes
    if child_index_count > 0:
        cur_off = bs.tell()
        bs.seek(child_index_off - scene_3df_22.HEADER_SIZE)
        child_indexes = [bs.read_uint32() for _ in range(child_index_count)]
        bs.seek(cur_off)
    else:
        child_indexes = []

    if face_groups_off > 0:
        vertex_count = bs.read_uint32()
        face_idx_count = bs.read_uint32()
        face_groups_count = bs.read_uint32()
        bs.seek(92, 1)

        # Read face groups
        cur_off = bs.tell()
        bs.seek(face_groups_off - scene_3df_22.HEADER_SIZE)
        face_groups = [
            scene_3df_22.read_face_group(bs) for _ in range(face_groups_count)
        ]
        bs.seek(cur_off)

        return scene_3df_22.MeshNode3DF(
            node_name,
            node_type,
            child_indexes,
            transform_type,
            transform,
            vertex_count,
            face_idx_count,
            face_groups,
        )
    else:
        unk_floats = [bs.read_float() for _ in range(13)]
        bs.seek(52, 1)

        return scene_3df_22.BoneNode3DF(
            node_name,
            node_type,
            child_indexes,
            transform_type,
            transform,
            unk_floats,
        )


def read_header(bs: BinaryReader) -> Header3DF:
    unk_int_0 = bs.read_uint32()
    unk_int_1 = bs.read_uint32()
    compress_mode = bs.read_uint32()
    nodes_chunk_size = bs.read_uint32()
    unk_chunk_size = bs.read_uint32()
    mesh_chunk_size = bs.read_uint32()
    bs.seek(124, 1)
    textures_chunk_size = bs.read_uint32()
    bs.seek(128, 1)
    materials_count = bs.read_uint32()
    materials_off = bs.read_uint32()
    unk_int_2 = bs.read_uint32()
    unk_int_3 = bs.read_uint32()
    nodes_count = bs.read_uint32()
    nodes_off = bs.read_uint32()

    return Header3DF(
        unk_int_0,
        unk_int_1,
        compress_mode,
        nodes_chunk_size,
        unk_chunk_size,
        mesh_chunk_size,
        textures_chunk_size,
        materials_count,
        materials_off,
        unk_int_2,
        unk_int_3,
        nodes_count,
        nodes_off,
    )

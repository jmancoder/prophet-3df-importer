from typing import NamedTuple

from .binary_reader import BinaryReader
from . import scene_3df_20

HEADER_SIZE = 412


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
    vertex_bitmask: int
    unk_int: int
    unk_float: float
    vertices_off: int
    faces_off: int


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


def read_material(bs: BinaryReader) -> scene_3df_20.Material3DF:
    name = bs.read_string_block(16)
    bs.read_uint32()
    property_count = bs.read_uint32()
    property_off = bs.read_uint32()
    diffuse_color = bs.read_bgra()
    bs.read_int32()
    bs.read_float()
    bs.read_int32()
    bs.read_int32()
    bs.read_int32()
    unk_color = bs.read_bgra()
    bs.seek(48, 1)
    material_end = bs.tell()

    # Read properties
    if property_count > 0 and property_off > 0:
        bs.seek(property_off - HEADER_SIZE)
        properties = [
            scene_3df_20.read_material_property(bs) for _ in range(property_count)
        ]
        bs.seek(material_end)
    else:
        properties = []

    return scene_3df_20.Material3DF(
        name,
        diffuse_color,
        unk_color,
        properties,
    )


def read_track(bs: BinaryReader) -> scene_3df_20.Track3DF:
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
    keys = [scene_3df_20.read_keyframe(bs) for _ in range(key_count)]
    bs.seek(track_end_off)

    return scene_3df_20.Track3DF(type_id, keys)


def read_node(bs: BinaryReader) -> scene_3df_20.Node3DF:
    node_name = bs.read_string_block(16)
    node_type = bs.read_uint32()
    bs.read_int32()
    bs.read_int32()
    child_index_count = bs.read_int32()
    internal_idx = bs.read_int32()
    child_index_off = bs.read_uint32()
    bs.read_int32()
    unk_vec_0 = bs.read_vec3f()
    unk_vec_1 = bs.read_vec3f()
    unk_floats_off = bs.read_uint32()
    track_count = bs.read_uint32()
    track_off = bs.read_uint32()
    transform_type = bs.read_uint32()
    if transform_type != 0:
        raise NotImplementedError(f"Unimplemented transform type {transform_type}")
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
            face_groups_count = bs.read_uint32()
            bs.seek(92, 1)

            # Read face groups
            if face_groups_off > 0:
                node_end_off = bs.tell()
                bs.seek(face_groups_off - HEADER_SIZE)
                face_groups = [
                    scene_3df_20.read_face_group(bs) for _ in range(face_groups_count)
                ]
                bs.seek(node_end_off)
            else:
                face_groups = []

            return scene_3df_20.MeshNode3DF(
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

            return scene_3df_20.BoneNode3DF(
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

            return scene_3df_20.Node3DF(
                node_name,
                node_type,
                internal_idx,
                child_indexes,
                transform_type,
                transform,
                tracks,
            )


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

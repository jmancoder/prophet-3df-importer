from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from .binary_reader import BinaryReader
from . import scene_3df_20


class Header3DF(NamedTuple):
    compress_mode: int
    node_chunk_size: int
    mesh_chunk_size: int
    texture_chunk_size: int
    material_count: int
    material_off: int
    texture_count: int
    node_count: int
    node_off: int


def read_header(bs: BinaryReader) -> Header3DF:
    bs.seek(8, 1)
    compress_mode = bs.read_uint32()
    node_chunk_size = bs.read_uint32()
    mesh_chunk_size = bs.read_uint32()
    bs.seek(120, 1)
    texture_chunk_size = bs.read_uint32()
    bs.seek(132, 1)
    material_count = bs.read_uint32()
    material_off = bs.read_uint32()
    texture_count = bs.read_uint32()
    bs.read_uint32()
    node_count = bs.read_uint32()
    node_off = bs.read_uint32()
    bs.read_int32()
    color = bs.read_bgra()
    bs.read_int32()
    bs.read_int32()
    bs.seek(76, 1)
    return Header3DF(
        compress_mode,
        node_chunk_size,
        mesh_chunk_size,
        texture_chunk_size,
        material_count,
        material_off,
        texture_count,
        node_count,
        node_off,
    )


def read_material(bs: BinaryReader, header_size: int) -> scene_3df_20.Material3DF:
    name = bs.read_string_block(16)
    bs.read_int32()
    property_count = bs.read_uint32()
    property_off = bs.read_uint32()
    diffuse_color = bs.read_bgra()
    bs.read_int32()
    bs.read_float()
    bs.read_float()
    bs.read_int32()
    bs.read_int32()
    unk_color = bs.read_bgra()
    bs.seek(48, 1)
    material_end = bs.tell()

    # Read properties
    if property_count > 0 and property_off > 0:
        bs.seek(property_off - header_size)
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


def create_vertex_dtype(bitmask: int) -> npt.DTypeLike:
    fields = []

    if bitmask & 0x1 != 0:
        fields.append(("position", np.float32, 3))

    blend_weight_count = 0
    if bitmask & 0x2:
        blend_weight_count = 1
    if bitmask & 0x4:
        blend_weight_count = 2
    if bitmask & 0x8:
        blend_weight_count = 3
    if blend_weight_count > 0:
        fields.append((f"blend_weights", np.float32, blend_weight_count))

    if bitmask & 0x10:
        fields.append(("normal", np.float32, 3))
    if bitmask & 0x20:
        fields.append(("diffuse", np.uint8, 4))

    uv_count = 0
    if bitmask & 0x100:
        uv_count = 1
    if bitmask & 0x200:
        uv_count = 2
    if bitmask & 0x400:
        uv_count = 3
    if uv_count > 0:
        fields.append((f"uvs", np.float32, (uv_count, 2)))
    return np.dtype(fields)


def read_node(bs: BinaryReader, header_size: int) -> scene_3df_20.Node3DF:
    name = bs.read_string_block(16)
    type_id = bs.read_uint32()
    flags = bs.read_int32()
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

from typing import NamedTuple

from mathutils import Matrix
import numpy as np
import numpy.typing as npt

from .binary_reader import BinaryReader


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


class MaterialProperty3DF(NamedTuple):
    type_id: int
    value: int
    unk_0: int
    unk_1: int


class Material3DF(NamedTuple):
    name: str
    diffuse_color: tuple[float, float, float, float]
    unk_color: tuple[float, float, float, float]
    properties: list[MaterialProperty3DF]


class FaceGroup3DF(NamedTuple):
    face_type: int
    face_index_count: int
    bone_indexes: tuple[int, int, int, int]
    material_index: int


class Key3DF(NamedTuple):
    time: float
    value: float | Matrix


class Track3DF(NamedTuple):
    type_id: int
    keys: list[Key3DF]


class MeshNodeData3DF(NamedTuple):
    vertex_count: int
    face_idx_count: int
    face_groups: list[FaceGroup3DF]


class BoneNodeData3DF(NamedTuple):
    unk_float: float
    bone_transform: Matrix


class Node3DF(NamedTuple):
    name: str
    type_id: int
    flags: int
    internal_index: int
    child_indexes: list[int]
    transform_type: int
    transform: Matrix
    tracks: list[Track3DF]
    data: MeshNodeData3DF | BoneNodeData3DF | None


class MeshInfo3DF(NamedTuple):
    flags: int
    vertex_off: int
    face_off: int


def read_header(bs: BinaryReader) -> Header3DF:
    bs.read_uint32()
    compress_mode = bs.read_uint32()
    node_chunk_size = bs.read_uint32()
    mesh_chunk_size = bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    texture_chunk_size = bs.read_uint32()
    bs.seek(20, 1)
    material_count = bs.read_uint32()
    material_off = bs.read_uint32()
    texture_count = bs.read_uint32()
    bs.read_uint32()
    node_count = bs.read_uint32()
    node_off = bs.read_uint32()
    bs.read_uint32()
    color = bs.read_bgra()
    bs.read_int32()
    bs.read_int32()
    bs.seek(80, 1)
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


def read_material_property(bs: BinaryReader) -> MaterialProperty3DF:
    type_id = bs.read_uint32()
    value = bs.read_uint32()
    unk_0 = bs.read_uint32()
    unk_1 = bs.read_uint32()
    return MaterialProperty3DF(type_id, value, unk_0, unk_1)


def read_material(bs: BinaryReader, header_size: int) -> Material3DF:
    name = bs.read_string_block(12)
    property_count = bs.read_uint32()
    property_off = bs.read_uint32()
    diffuse_color = bs.read_bgra()
    bs.read_uint32()
    bs.read_float()
    bs.read_float()
    bs.read_int32()
    bs.read_int32()
    bs.read_int32()
    unk_color = bs.read_bgra()
    bs.seek(44, 1)
    material_end = bs.tell()

    # Read properties
    if property_count > 0 and property_off > 0:
        bs.seek(property_off - header_size)
        properties = [read_material_property(bs) for _ in range(property_count)]
        bs.seek(material_end)
    else:
        properties = []
    return Material3DF(
        name,
        diffuse_color,
        unk_color,
        properties,
    )


def read_face_group(bs: BinaryReader) -> FaceGroup3DF:
    face_type = bs.read_uint16()
    face_count = bs.read_uint16()
    bone_indexes = bs.read_vec4B()
    bs.read_uint16()
    bs.read_uint16()
    material_idx = bs.read_uint16()
    bs.read_uint16()
    return FaceGroup3DF(face_type, face_count, bone_indexes, material_idx)


def read_float_keyframe(bs: BinaryReader) -> Key3DF:
    time = bs.read_float()
    value = bs.read_float()
    bs.read_float()
    bs.read_float()
    bs.read_float()
    bs.read_float()
    return Key3DF(time, value)


def read_matrix_keyframe(bs: BinaryReader) -> Key3DF:
    time = bs.read_float()
    value = bs.read_matrix_3x4()
    return Key3DF(time, value)


def read_track(bs: BinaryReader, header_size: int) -> Track3DF:
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
    bs.seek(key_off - header_size)
    if type_id == 9:
        keys = [read_matrix_keyframe(bs) for _ in range(key_count)]
    else:
        keys = [read_float_keyframe(bs) for _ in range(key_count)]
    bs.seek(track_end_off)
    return Track3DF(type_id, keys)


def create_vertex_dtype(bitmask: int) -> npt.DTypeLike:
    fields = []
    if bitmask & 0x800:
        fields.append(("position", np.float32, 3))
        fields.append((f"blend_weights", np.float32, 3))
    elif bitmask & 0x400:
        fields.append(("position", np.float32, 3))
        fields.append((f"blend_weights", np.float32, 2))
    elif bitmask & 0x200:
        fields.append(("position", np.float32, 3))
        fields.append((f"blend_weights", np.float32, 1))
    elif bitmask & 0x100:
        fields.append(("position", np.float32, 3))

    if bitmask & 0x1000:
        fields.append(("normal", np.float32, 3))
    if bitmask & 0x2000:
        fields.append(("diffuse", np.uint8, 4))

    uv_count = 0
    if bitmask & 0x8000:
        uv_count = 1
    if bitmask & 0x10000:
        uv_count = 2
    if bitmask & 0x20000:
        uv_count = 3
    if uv_count > 0:
        fields.append((f"uvs", np.float32, (uv_count, 2)))
    return np.dtype(fields)


def read_node(bs: BinaryReader, header_size: int) -> Node3DF:
    name = bs.read_string_block(12)
    type_id = bs.read_uint32()
    flags = bs.read_uint32()
    bs.read_int32()
    child_index_count = bs.read_int32()
    internal_idx = bs.read_int32()
    child_index_off = bs.read_uint32()
    bs.seek(124, 1)
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
    bs.seek(4, 1)
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
        tracks = [read_track(bs, header_size) for _ in range(track_count)]
        bs.seek(node_end_off)
    else:
        tracks = []

    # Read node data
    match type_id:
        case 0:
            vertex_count = bs.read_uint32()
            face_idx_count = bs.read_uint32()
            face_group_count = bs.read_uint32()
            bs.seek(92, 1)
            # Read face groups
            if face_group_off > 0:
                node_end_off = bs.tell()
                bs.seek(face_group_off - header_size)
                face_groups = [read_face_group(bs) for _ in range(face_group_count)]
                bs.seek(node_end_off)
            else:
                face_groups = []
            data = MeshNodeData3DF(
                vertex_count,
                face_idx_count,
                face_groups,
            )
        case 1:
            unk_float = bs.read_float()
            bone_transform = bs.read_matrix_3x4()
            bs.seek(52, 1)
            data = BoneNodeData3DF(
                unk_float,
                bone_transform,
            )
        case _:
            bs.seek(104, 1)
            data = None
    return Node3DF(
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
    vertex_off = bs.read_uint32()
    face_off = bs.read_uint32()
    return MeshInfo3DF(
        flags,
        vertex_off,
        face_off,
    )

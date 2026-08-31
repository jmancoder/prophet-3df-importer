from typing import NamedTuple

from mathutils import Matrix

from .binary_reader import BinaryReader
from . import scene_3df_20

HEADER_SIZE = 412


class MeshInfo3DF(NamedTuple):
    vertex_bitmask: int
    unk_int: int
    unk_float: float
    vertices_off: int
    faces_off: int
    transform: Matrix


def read_header(bs: BinaryReader) -> scene_3df_20.Header3DF:
    bs.read_uint32()
    bs.read_uint32()
    compress_mode = bs.read_uint32()
    node_chunk_size = bs.read_uint32()
    mesh_chunk_size = bs.read_uint32()
    bs.seek(124, 1)
    texture_chunk_size = bs.read_uint32()
    bs.seek(132, 1)
    material_count = bs.read_uint32()
    material_off = bs.read_uint32()
    texture_count = bs.read_uint32()
    bs.read_uint32()
    node_count = bs.read_uint32()
    node_off = bs.read_uint32()

    return scene_3df_20.Header3DF(
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


def read_mesh_info(bs: BinaryReader) -> MeshInfo3DF:
    vertex_bitmask = bs.read_uint32()
    unk_int = bs.read_uint32()
    unk_float = bs.read_float()
    vertices_off = bs.read_uint32()
    faces_off = bs.read_uint32()
    mesh_transform = bs.read_matrix_3x4()

    return MeshInfo3DF(
        vertex_bitmask,
        unk_int,
        unk_float,
        vertices_off,
        faces_off,
        mesh_transform,
    )

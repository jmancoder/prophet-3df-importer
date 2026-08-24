from typing import NamedTuple

from mathutils import Matrix

from .binary_reader import BinaryReader


class MeshInfo3DF(NamedTuple):
    vertex_bitmask: int
    unk_int: int
    unk_float: float
    vertices_off: int
    faces_off: int
    transform: Matrix


class Header3DF(NamedTuple):
    unk_int_0: int
    unk_int_1: int
    compress_mode: int
    nodes_chunk_size: int
    meshes_chunk_size: int
    textures_chunk_size: int
    materials_count: int
    materials_off: int
    unk_int_2: int
    unk_int_3: int
    nodes_count: int
    nodes_off: int


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


def read_header(bs: BinaryReader) -> Header3DF:
    unk_int_0 = bs.read_uint32()
    unk_int_1 = bs.read_uint32()
    compress_mode = bs.read_uint32()
    nodes_chunk_size = bs.read_uint32()
    meshes_chunk_size = bs.read_uint32()
    bs.seek(124, 1)
    textures_chunk_size = bs.read_uint32()
    bs.seek(132, 1)
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
        meshes_chunk_size,
        textures_chunk_size,
        materials_count,
        materials_off,
        unk_int_2,
        unk_int_3,
        nodes_count,
        nodes_off,
    )

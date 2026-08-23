from typing import NamedTuple

from .binary_reader import BinaryReader

HEADER_SIZE = 56


class Header3DF(NamedTuple):
    compress_mode: int
    mesh_info_count: int
    mesh_info_off: int
    mesh_related_off: int
    node_count: int
    node_off: int


def read_header(bs: BinaryReader) -> Header3DF:
    bs.read_uint32()
    compress_mode = bs.read_uint32()
    mesh_info_count = bs.read_uint32()
    mesh_info_off = bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    mesh_related_off = bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    node_count = bs.read_uint32()
    node_off = bs.read_uint32()

    return Header3DF(
        compress_mode,
        mesh_info_count,
        mesh_info_off,
        mesh_related_off,
        node_count,
        node_off,
    )

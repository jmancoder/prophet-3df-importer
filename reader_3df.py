from io import BufferedReader
from typing import NamedTuple, Sequence
import zlib

import numpy as np
import numpy.typing as npt

from .binary_reader import BinaryReader
from . import image_utils
from . import scene_3df_20
from . import scene_3df_22
from . import scene_3df_23
from . import scene_3df_26_ds
from . import scene_3df_26_pc


class Texture3DF(NamedTuple):
    width: int
    height: int
    pixels: npt.NDArray


class TriangleGroup3DF(NamedTuple):
    bone_indexes: tuple[int, int, int, int]
    triangles: npt.NDArray
    vertex_indices: npt.NDArray
    material_index: int


class MeshData3DF(NamedTuple):
    vertices: npt.NDArray
    triangle_groups: list[TriangleGroup3DF]


class SceneData3DF(NamedTuple):
    materials: list[scene_3df_20.Material3DF]
    nodes: Sequence[scene_3df_20.Node3DF | scene_3df_22.Node3DF]
    mesh_map: dict[int, MeshData3DF]
    textures: list[Texture3DF]


def read_texture(bs: BinaryReader, has_extra_header: bool = False) -> Texture3DF:
    flags = bs.read_uint32()
    type_id = bs.read_uint32()
    data_offset = bs.read_uint32()
    width = bs.read_uint32()
    height = bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    padded_size = bs.read_uint32()
    tex_info_end = bs.tell()

    bs.seek(data_offset)
    data_off = bs.read_uint32()
    data_size = bs.read_uint32()
    bs.seek(data_off)
    match type_id:
        case 0:
            # 8-bit paletted
            palette = np.frombuffer(bs.getbuffer(), np.uint8, 1024, bs.tell()).reshape(
                256, 4
            )
            palette = palette[:, [2, 1, 0, 3]]  # Convert BGRA to RGBA
            bs.seek(palette.nbytes, 1)
            if has_extra_header:
                bs.seek(64, 1)
            indices = np.frombuffer(bs.getbuffer(), np.uint8, width * height, bs.tell())
            pixels = image_utils.rgba_to_floats(palette[indices]).ravel()
        case 3:
            # 4-bit paletted
            palette = np.frombuffer(bs.getbuffer(), np.uint8, 64, bs.tell()).reshape(
                16, 4
            )
            palette = palette[:, [2, 1, 0, 3]]  # Convert BGRA to RGBA
            bs.seek(palette.nbytes, 1)
            if has_extra_header:
                bs.seek(64, 1)
            indices_raw = np.frombuffer(
                bs.getbuffer(), np.uint8, width * height // 2, bs.tell()
            )
            indices = np.empty(indices_raw.size * 2, dtype=np.uint8)
            indices[0::2] = indices_raw >> 4
            indices[1::2] = indices_raw & 0xF
            pixels = image_utils.rgba_to_floats(palette[indices]).ravel()
        case 6:
            # BGRA4444
            raw_pixels = np.frombuffer(
                bs.getbuffer(), np.uint16, width * height, bs.tell()
            )
            pixels = np.empty((raw_pixels.size, 4), dtype=np.float32)
            pixels[:, 0] = ((raw_pixels >> 8) & 0xF) / 15.0
            pixels[:, 1] = ((raw_pixels >> 4) & 0xF) / 15.0
            pixels[:, 2] = (raw_pixels & 0xF) / 15.0
            pixels[:, 3] = ((raw_pixels >> 12) & 0xF) / 15.0
            pixels = pixels.ravel()
        case 8:
            # DXT1
            pixels = image_utils.dxt1_to_rgba(bs.read(data_size), width, height)
        case _:
            print(f"WARNING: Unimplemented texture type {type_id}")
            pixels = np.tile([0.0, 0.0, 0.0, 1.0], width * height)
    bs.seek(tex_info_end)

    return Texture3DF(
        width,
        height,
        pixels,
    )


def tri_strips_to_triangles(indices: npt.NDArray) -> npt.NDArray:
    triangles = []

    for i in range(len(indices) - 2):
        if i % 2 == 0:
            tri = (indices[i], indices[i + 1], indices[i + 2])
        else:
            tri = (indices[i], indices[i + 2], indices[i + 1])

        if tri[0] != tri[1] and tri[1] != tri[2] and tri[0] != tri[2]:
            triangles.append(tri)

    return np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def decompress_chunk_stream(bs: BinaryReader) -> BinaryReader:
    decomp_size = bs.read_uint32()
    comp_size = bs.read_uint32()
    flags = bs.read_uint32()
    mode_a = bs.read_uint16()
    mode_b = bs.read_uint16()

    if mode_a == 0:
        # Copy data directly
        bs_out = BinaryReader(bs.read(decomp_size))
    elif mode_a == 6:
        # Decompress raw zlib data
        bs_out = BinaryReader(zlib.decompress(bs.read(decomp_size), wbits=-15))
    else:
        raise ValueError(f"Unknown compression mode {mode_a}")

    return bs_out


class Reader3DF:
    def __init__(self, platform: str) -> None:
        self.platform: str = platform
        self.version: int = 20

    def read_scene_from_file(self, f: BufferedReader) -> SceneData3DF:
        # Load header chunk
        bs = BinaryReader(f.read(8))

        # Validate signature
        sig = bs.read_string_block(4)
        if sig != "3df":
            raise ValueError("Missing 3df file signature")

        # Load and read self.version-specific header
        self.version = bs.read_uint32()
        if self.version == 20:
            header_size = scene_3df_20.HEADER_SIZE
            bs = BinaryReader(f.read(header_size - 8))
            header = scene_3df_20.read_header(bs)
        elif self.version == 22 or self.version == 23:
            header_size = scene_3df_22.HEADER_SIZE
            bs = BinaryReader(f.read(header_size - 8))
            header = scene_3df_22.read_header(bs)
        elif self.version == 26:
            if self.platform == "DS":
                header_size = scene_3df_26_ds.HEADER_SIZE
                bs = BinaryReader(f.read(header_size - 8))
                header = scene_3df_26_ds.read_header(bs)
                raise NotImplementedError(
                    f"Unimplemented self.platform {self.platform} for self.version {self.version}"
                )
            elif self.platform == "PC":
                header_size = scene_3df_26_pc.HEADER_SIZE
                bs = BinaryReader(f.read(header_size - 8))
                header = scene_3df_26_pc.read_header(bs)
            elif self.platform == "PS2":
                header_size = scene_3df_22.HEADER_SIZE
                bs = BinaryReader(f.read(header_size - 8))
                header = scene_3df_22.read_header(bs)
                raise NotImplementedError(
                    f"Unimplemented platform {self.platform} for version {self.version}"
                )
            else:
                raise NotImplementedError(
                    f"Unimplemented platform {self.platform} for version {self.version}"
                )
        else:
            raise NotImplementedError(f"Unimplemented 3DF version {self.version}")

        # Load node chunk
        bs = BinaryReader(f.read(header.node_chunk_size))
        if header.compress_mode == 1:
            bs = decompress_chunk_stream(bs)

        # Read materials
        bs.seek(header.material_off - header_size)
        if self.version == 20:
            materials = [
                scene_3df_20.read_material(bs) for _ in range(header.material_count)
            ]
        else:
            materials = [
                scene_3df_22.read_material(bs) for _ in range(header.material_count)
            ]

        # Read nodes
        bs.seek(header.node_off - header_size)
        if self.version == 20:
            nodes = [scene_3df_20.read_node(bs) for _ in range(header.node_count)]
        elif self.version == 22:
            nodes = [scene_3df_22.read_node(bs) for _ in range(header.node_count)]
        else:
            nodes = [scene_3df_23.read_node(bs) for _ in range(header.node_count)]

        # Load mesh chunk
        f.seek(header_size + header.node_chunk_size)
        bs = BinaryReader(f.read(header.mesh_chunk_size))
        if header.compress_mode == 1:
            bs = decompress_chunk_stream(bs)

        # Read mesh info entries
        if self.version == 20:
            mesh_info_entries = [
                scene_3df_20.read_mesh_info(bs) for _ in range(header.node_count)
            ]
        elif self.version == 22 or self.version == 23:
            mesh_info_entries = [
                scene_3df_22.read_mesh_info(bs) for _ in range(header.node_count)
            ]
        else:
            mesh_info_entries = [
                scene_3df_26_pc.read_mesh_info(bs) for _ in range(header.node_count)
            ]

        # Read meshes
        mesh_data_map: dict[int, MeshData3DF] = {}
        for i, (node, mesh_info) in enumerate(zip(nodes, mesh_info_entries)):
            if node.type_id != 0:
                continue

            # Read vertices
            if self.version == 20:
                vertex_dtype = scene_3df_20.create_vertex_dtype(node.flags)
                print(node.name, vertex_dtype, node.flags)
            else:
                vertex_dtype = scene_3df_22.create_vertex_dtype(mesh_info.flags)
            bs.seek(mesh_info.vertices_off)
            vertices = np.frombuffer(
                bs.getbuffer(),
                vertex_dtype,
                node.vertex_count,
                bs.tell(),
            )

            # Read faces
            if self.version == 26:
                face_dtype = np.uint32
            else:
                face_dtype = np.uint16
            triangle_groups: list[TriangleGroup3DF] = []
            bs.seek(mesh_info.faces_off)
            for face_group in node.face_groups:
                if face_group.face_type == 1:
                    # Read triangle strips
                    tri_strip_indices = np.frombuffer(
                        bs.getbuffer(),
                        face_dtype,
                        face_group.face_index_count,
                        bs.tell(),
                    )
                    bs.seek(tri_strip_indices.nbytes, 1)
                    vertex_indices = np.unique(tri_strip_indices)
                    triangle_groups.append(
                        TriangleGroup3DF(
                            face_group.bone_indexes,
                            tri_strips_to_triangles(tri_strip_indices),
                            vertex_indices,
                            face_group.material_index,
                        )
                    )
                elif face_group.face_type == 3:
                    # Read triangles
                    tri_indices = np.frombuffer(
                        bs.getbuffer(),
                        face_dtype,
                        face_group.face_index_count,
                        bs.tell(),
                    )
                    bs.seek(tri_indices.nbytes, 1)
                    vertex_indices = np.unique(tri_indices)
                    triangle_groups.append(
                        TriangleGroup3DF(
                            face_group.bone_indexes,
                            tri_indices.reshape(-1, 3),
                            vertex_indices,
                            face_group.material_index,
                        )
                    )
                else:
                    print(
                        "WARNING: Unimplemented face type " + str(face_group.face_type)
                    )

            mesh_data_map[i] = MeshData3DF(
                vertices,
                triangle_groups,
            )

        # Load texture chunk
        bs = BinaryReader(f.read(header.texture_chunk_size))
        if header.compress_mode == 1:
            bs = decompress_chunk_stream(bs)

        # Read textures
        textures: list[Texture3DF] = []
        for i in range(header.texture_count):
            textures.append(read_texture(bs, has_extra_header=self.version == 23))

        return SceneData3DF(materials, nodes, mesh_data_map, textures)

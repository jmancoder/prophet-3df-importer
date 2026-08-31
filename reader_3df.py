from io import BufferedReader
from typing import NamedTuple
import zlib

import numpy as np
import numpy.typing as npt

from .binary_reader import BinaryReader
from . import scene_3df_20
from . import scene_3df_22
from . import scene_3df_23
from . import scene_3df_26_ds
from . import scene_3df_26_pc


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
    nodes: list[scene_3df_20.Node3DF]
    mesh_map: dict[int, MeshData3DF]
    textures: list[scene_3df_20.Texture3DF]


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
        fields.append(("color", np.uint8, 4))

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
        if self.version == 22 or self.version == 23:
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
                    f"Unimplemented self.platform {self.platform} for self.version {self.version}"
                )
            else:
                raise NotImplementedError(
                    f"Unimplemented self.platform {self.platform} for self.version {self.version}"
                )
        else:
            raise NotImplementedError(f"Unimplemented 3DF self.version {self.version}")

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
        if self.version == 22:
            nodes = [scene_3df_22.read_node(bs) for _ in range(header.node_count)]
        else:
            nodes = [scene_3df_23.read_node(bs) for _ in range(header.node_count)]

        # Load mesh chunk
        f.seek(header_size + header.node_chunk_size)
        bs = BinaryReader(f.read(header.mesh_chunk_size))
        if header.compress_mode == 1:
            bs = decompress_chunk_stream(bs)

        # Read mesh info entries
        if self.version == 22 or self.version == 23:
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
            if mesh_info.vertex_bitmask == 0 or not isinstance(
                node, scene_3df_20.MeshNode3DF
            ):
                continue

            # Read vertices
            bs.seek(mesh_info.vertices_off)
            vertex_dtype = create_vertex_dtype(mesh_info.vertex_bitmask)
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

        # Load and read textures
        bs = BinaryReader(f.read(header.texture_chunk_size))
        if header.compress_mode == 1:
            bs = decompress_chunk_stream(bs)
        if self.version == 23:
            textures = [
                scene_3df_23.read_texture(bs) for _ in range(header.texture_count)
            ]
        else:
            textures = [
                scene_3df_20.read_texture(bs) for _ in range(header.texture_count)
            ]

        return SceneData3DF(materials, nodes, mesh_data_map, textures)

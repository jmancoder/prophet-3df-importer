from io import BufferedReader
from typing import NamedTuple
import zlib

import bpy
from bpy.types import Context
import numpy as np
import numpy.typing as npt

from .binary_reader import BinaryReader
from . import scene_3df_22
from . import scene_3df_23
from . import scene_3df_26


class MeshData3DF(NamedTuple):
    node_id: int
    vertices: npt.NDArray
    triangles: list[tuple[int, int, int]]


class SceneData3DF(NamedTuple):
    nodes: list[scene_3df_22.Node3DF]
    meshes: list[MeshData3DF]


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


def tri_strips_to_triangles(indices: list[int]) -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []
    for i in range(len(indices) - 2):
        if i % 2 == 0:
            tri = (indices[i], indices[i+1], indices[i+2])
        else:
            tri = (indices[i], indices[i+2], indices[i+1])

        # Read triangle if it is not degenerate
        if tri[0] != tri[1] and tri[1] != tri[2] and tri[0] != tri[2]:
            triangles.append(tri)

    return triangles


def decompress_chunk_stream(bs: BinaryReader) -> BinaryReader:
    decomp_size = bs.read_uint32()
    comp_size = bs.read_uint32()
    flags = bs.read_uint32()
    mode_a = bs.read_uint16()
    mode_b = bs.read_uint16()

    if mode_a == 0:
        # Copy data directly
        bs_out = BinaryReader(b"\x00" * decomp_size)
        bs.readinto(bs_out.getbuffer())
    elif mode_a == 6:
        # Decompress raw zlib data
        bs_out = BinaryReader(
            zlib.decompress(bs.read(decomp_size),
                            wbits=-15))
    else:
        raise ValueError(f"Unknown compression mode {mode_a}")

    return bs_out


def read_3df(f: BufferedReader) -> SceneData3DF:
    # Load header chunk
    bs = BinaryReader(b"\x00" * 412)
    f.readinto(bs.getbuffer())

    # Validate signature
    sig = bs.read_string_block(4)
    if sig != "3df":
        raise ValueError("Missing 3df file signature")

    # Read version-specific header
    version = bs.read_uint32()
    if version == 22 or version == 23:
        header = scene_3df_23.read_header(bs)
    elif version == 26:
        header = scene_3df_26.read_header(bs)
    else:
        raise NotImplementedError(f"Unimplemented 3DF version {version}")

    # Load nodes chunk
    bs = BinaryReader(b"\x00" * header.nodes_chunk_size)
    f.readinto(bs.getbuffer())
    if header.compress_mode == 1:
        bs = decompress_chunk_stream(bs)

    # Read materials
    bs.seek(header.materials_off - scene_3df_22.HEADER_SIZE)
    materials = [
        scene_3df_22.read_material(bs)
        for _ in range(header.materials_count)
    ]

    # Read nodes
    bs.seek(header.nodes_off - scene_3df_22.HEADER_SIZE)
    if version == 22:
        nodes = [
            scene_3df_22.read_node(bs)
            for _ in range(header.nodes_count)
        ]
    else:
        nodes = [
            scene_3df_23.read_node(bs)
            for _ in range(header.nodes_count)
        ]

    # Load mesh chunk
    bs = BinaryReader(b"\x00" * header.meshes_chunk_size)
    f.seek(scene_3df_22.HEADER_SIZE + header.nodes_chunk_size)
    f.readinto(bs.getbuffer())
    if header.compress_mode == 1:
        bs = decompress_chunk_stream(bs)

    # Read mesh info entries
    if version == 22 or version == 23:
        mesh_info_entries = [
            scene_3df_22.read_mesh_info(bs)
            for _ in range(header.nodes_count)
        ]
    else:
        mesh_info_entries = [
            scene_3df_26.read_mesh_info(bs)
            for _ in range(header.nodes_count)
        ]

    # Read meshes
    meshes: list[MeshData3DF] = []
    for i, (node, mesh_info) in enumerate(zip(nodes, mesh_info_entries)):
        if (mesh_info.vertex_bitmask == 0 or not isinstance(
                node, scene_3df_22.MeshNode3DF)):
            continue

        # Read vertices
        bs.seek(mesh_info.vertices_off)
        vertex_dtype = create_vertex_dtype(mesh_info.vertex_bitmask)
        vertices = np.frombuffer(
            bs.read(node.vertex_count * vertex_dtype.itemsize),
            vertex_dtype,
            node.vertex_count,
        )

        # Read faces
        triangles: list[tuple[int, int, int]] = []
        bs.seek(mesh_info.faces_off)
        for face_group in node.face_groups:
            if face_group.face_type == 3:
                # Read triangles
                if version == 22 or version == 23:
                    triangles.extend([
                        bs.read_vec3H()
                        for _ in range(face_group.face_idx_count // 3)
                    ])
                else:
                    triangles.extend([
                        bs.read_vec3I()
                        for _ in range(face_group.face_idx_count // 3)
                    ])
            elif face_group.face_type == 1:
                # Read triangle strips
                if version == 22 or version == 23:
                    tri_strip_indices = [
                        bs.read_uint16()
                        for _ in range(face_group.face_idx_count)
                    ]
                else:
                    tri_strip_indices = [
                        bs.read_uint32()
                        for _ in range(face_group.face_idx_count)
                    ]
                triangles.extend(tri_strips_to_triangles(tri_strip_indices))
            else:
                print("WARNING: Unimplemented face type "
                      + str(face_group.face_type))

        meshes.append(MeshData3DF(
            i,
            vertices,
            triangles,
        ))

    return SceneData3DF(nodes, meshes)


def import_3df(scene_data: SceneData3DF, context: Context) -> None:
    # Create meshes
    for mesh_3df in scene_data.meshes:
        mesh_node = scene_data.nodes[mesh_3df.node_id]

        # Skip meshes without vertex positions
        if mesh_3df.vertices.dtype.names is None:
            continue
        if "position" not in mesh_3df.vertices.dtype.names:
            continue

        # Import positions and triangles
        mesh = bpy.data.meshes.new(mesh_node.name)
        mesh.from_pydata(
            mesh_3df.vertices["position"],
            [],
            mesh_3df.triangles,
        )

        # Import vertex UV layers
        if "uvs" in mesh_3df.vertices.dtype.names:
            for i in range(mesh_3df.vertices.dtype["uvs"].shape[0]):
                uv_layer = mesh.uv_layers.new(name=f"UV{i}")
                for loop in mesh.loops:
                    uv = mesh_3df.vertices["uvs"][loop.vertex_index][i]
                    uv_layer.data[loop.index].uv = (uv[0], 1.0 - uv[1])

        # Import vertex normals
        if "normal" in mesh_3df.vertices.dtype.names:
            mesh.normals_split_custom_set_from_vertices(
                mesh_3df.vertices["normal"]
            )

        # Import vertex colors
        if "color" in mesh_3df.vertices.dtype.names:
            vertex_color_attr = mesh.color_attributes.new(
                name="vertex_color",
                type="BYTE_COLOR",
                domain="POINT",
            )
            vertex_color_attr.data.foreach_set(
                "color",
                mesh_3df.vertices["color"].flatten(),
            )

        # Validate mesh
        mesh.validate()
        mesh.update()

        # Create mesh object
        mesh_obj = bpy.data.objects.new(mesh_node.name, mesh)
        context.collection.objects.link(mesh_obj)

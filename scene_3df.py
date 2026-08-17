from dataclasses import dataclass
from io import BufferedReader
from typing import NamedTuple

import bpy
from bpy.types import Context
from mathutils import Matrix
import numpy as np
import numpy.typing as npt

from .binary_reader import BinaryReader


HEADER_SIZE = 412


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


class Material3DF(NamedTuple):
    name: str
    flags_a: int
    unk_count: int
    unk_off: int
    flags_b: int


class FaceGroup3DF(NamedTuple):
    face_type: int
    face_idx_count: int
    face_flags: int


@dataclass(frozen=True, slots=True)
class Node3DF:
    name: str
    type_id: int
    transform_type: int


@dataclass(frozen=True, slots=True)
class BoneNode3DF(Node3DF):
    unk_float: float
    unk_matrix: Matrix


@dataclass(frozen=True, slots=True)
class MeshNode3DF(Node3DF):
    vertex_count: int
    face_idx_count: int
    face_groups: list[FaceGroup3DF]


class MeshInfo3DF(NamedTuple):
    vertex_bitmask: int
    unk_int: int
    unk_float: float
    vertices_off: int
    faces_off: int
    transform: Matrix


class MeshData3DF(NamedTuple):
    node_id: int
    vertices: npt.NDArray
    triangles: list[tuple[int, int, int]]


class SceneData3DF(NamedTuple):
    nodes: list[Node3DF]
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


def read_mesh_info(bs: BinaryReader) -> MeshInfo3DF:
    vertex_bitmask = bs.read_uint32()
    unk_int = bs.read_uint32()
    unk_float = bs.read_float()
    vertices_off = bs.read_uint32()
    faces_off = bs.read_uint32()
    mesh_transform = bs.read_mat43()

    return MeshInfo3DF(
        vertex_bitmask,
        unk_int,
        unk_float,
        vertices_off,
        faces_off,
        mesh_transform,
    )


def read_face_group(bs: BinaryReader) -> FaceGroup3DF:
    face_type = bs.read_uint16()
    face_count = bs.read_uint16()
    face_flags = bs.read_uint32()
    bs.seek(8, 1)

    return FaceGroup3DF(
        face_type,
        face_count,
        face_flags,
    )


def read_node(bs: BinaryReader) -> Node3DF:
    node_name = bs.read_string_block(16)
    node_type = bs.read_uint32()
    bs.seek(60, 1)
    transform_type = bs.read_uint32()
    bs.seek(148, 1)
    face_groups_off = bs.read_uint32()
    bs.seek(32, 1)

    if face_groups_off > 0:
        vertex_count = bs.read_uint32()
        face_idx_count = bs.read_uint32()
        face_groups_count = bs.read_uint32()
        bs.seek(92, 1)
        next_node_off = bs.tell()

        # Read face groups
        bs.seek(face_groups_off - HEADER_SIZE)
        face_groups = [
            read_face_group(bs)
            for _ in range(face_groups_count)
        ]

        bs.seek(next_node_off)

        return MeshNode3DF(
            node_name,
            node_type,
            transform_type,
            vertex_count,
            face_idx_count,
            face_groups,
        )
    else:
        unk_float = bs.read_float()
        unk_matrix = bs.read_mat43()
        bs.seek(52, 1)

        return BoneNode3DF(
            node_name,
            node_type,
            transform_type,
            unk_float,
            unk_matrix,
        )


def read_material(bs: BinaryReader) -> Material3DF:
    name = bs.read_string_block(16)
    flags_a = bs.read_uint32()
    unk_count = bs.read_uint32()
    unk_off = bs.read_uint32()
    flags_b = bs.read_uint32()
    bs.read_uint32()
    bs.read_float()
    bs.read_int32()
    bs.read_int32()
    bs.read_int32()
    bs.read_int32()

    return Material3DF(
        name,
        flags_a,
        unk_count,
        unk_off,
        flags_b,
    )


def read_header(bs: BinaryReader) -> Header3DF:
    unk_int_0 = bs.read_uint32()
    unk_int_1 = bs.read_uint32()
    compress_mode = bs.read_uint32()
    nodes_chunk_size = bs.read_uint32()
    meshes_chunk_size = bs.read_uint32()
    bs.seek(0x98)
    textures_chunk_size = bs.read_uint32()
    bs.seek(0x120)
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


def read_3df(f: BufferedReader) -> SceneData3DF:
    # Load header chunk
    bs = BinaryReader(b"\x00" * 412)
    f.readinto(bs.getbuffer())

    # Validate signature
    sig = bs.read_string_block(4)
    if sig != "3df":
        raise ValueError("Missing 3df file signature")

    version = bs.read_uint32()
    if version != 26:
        raise NotImplementedError(f"Unimplemented 3DA version {version}")

    # Read header
    header = read_header(bs)

    # Load nodes chunk
    bs = BinaryReader(b"\x00" * header.nodes_chunk_size)
    f.readinto(bs.getbuffer())

    # Read materials
    bs.seek(header.materials_off - HEADER_SIZE)
    materials = [
        read_material(bs)
        for _ in range(header.materials_count)
    ]

    # Read nodes
    bs.seek(header.nodes_off - HEADER_SIZE)
    nodes = [
        read_node(bs)
        for _ in range(header.nodes_count)
    ]

    # Load mesh chunk
    bs = BinaryReader(b"\x00" * header.meshes_chunk_size)
    f.seek(HEADER_SIZE + header.nodes_chunk_size)
    f.readinto(bs.getbuffer())

    # Read mesh info entries
    mesh_info_entries = [
        read_mesh_info(bs)
        for _ in range(header.nodes_count)
    ]

    # Read meshes
    meshes: list[MeshData3DF] = []
    for i, (node, mesh_info) in enumerate(zip(nodes, mesh_info_entries)):
        if mesh_info.vertex_bitmask == 0 or not isinstance(node, MeshNode3DF):
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
                triangles.extend([
                    bs.read_vec3I()
                    for _ in range(face_group.face_idx_count // 3)
                ])
            elif face_group.face_type == 1:
                # Read triangle strips
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
                type='BYTE_COLOR',
                domain='POINT',
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

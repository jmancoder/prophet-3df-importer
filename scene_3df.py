from dataclasses import dataclass
from typing import NamedTuple

import bpy
from bpy.types import Context
from mathutils import Matrix
import numpy as np
import numpy.typing as npt

from .binary_reader import BinaryReader


@dataclass
class Node3DF:
    name: str
    type_id: int
    transform_type: int


@dataclass
class MeshNode3DF(Node3DF):
    face_groups_off: int
    vertex_count: int
    face_idx_count: int
    face_group_count: int


class MeshInfo3DF(NamedTuple):
    vertex_bitmask: int
    vertices_off: int
    faces_off: int
    transform: Matrix


class MeshData3DF(NamedTuple):
    node_id: int
    vertices: npt.NDArray
    triangles: list[tuple[int, int, int]]


class FaceGroup3DF(NamedTuple):
    face_type: int
    face_idx_count: int
    face_flags: int


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


def parse_3df(data: bytes) -> SceneData3DF:
    bs = BinaryReader(data)

    # Read header
    sig = bs.read_string_block(4)
    if sig != "3df":
        raise ValueError("Missing 3df file signature")
    bs.seek(0x14)
    node_section_size = bs.read_uint32()
    mesh_section_size = bs.read_uint32()
    bs.seek(0x98)
    texture_section_size = bs.read_uint32()
    bs.seek(0x120)
    surface_count = bs.read_uint32()
    surfaces_off = bs.read_uint32()
    bs.read_uint32()
    bs.read_uint32()
    node_count = bs.read_uint32()
    nodes_off = bs.read_uint32()

    # Partially read nodes
    nodes: list[Node3DF] = []
    bs.seek(nodes_off)
    for _ in range(node_count):
        node_name = bs.read_string_block(16)
        node_type = bs.read_uint32()
        bs.seek(60, 1)
        transform_type = bs.read_uint32()
        bs.seek(148, 1)
        face_groups_off = bs.read_uint32()
        bs.seek(32, 1)
        vertex_count = bs.read_uint32()
        face_idx_count = bs.read_uint32()
        face_group_count = bs.read_uint32()
        bs.seek(92, 1)

        nodes.append(MeshNode3DF(
            node_name,
            node_type,
            transform_type,
            face_groups_off,
            vertex_count,
            face_idx_count,
            face_group_count,
        ))

    # Read node-paired mesh info entries
    mesh_info_entries: list[MeshInfo3DF] = []
    mesh_section_off = 0x19C + node_section_size
    bs.seek(mesh_section_off)
    for _ in range(node_count):
        vertex_bitmask = bs.read_uint32()
        bs.read_uint32()
        bs.read_float()
        vertices_off = bs.read_uint32() + mesh_section_off
        faces_off = bs.read_uint32() + mesh_section_off
        mesh_transform = bs.read_mat43()

        mesh_info_entries.append(MeshInfo3DF(
            vertex_bitmask,
            vertices_off,
            faces_off,
            mesh_transform,
        ))

    # Read meshes
    meshes: list[MeshData3DF] = []
    for i, (node, mesh_info) in enumerate(zip(nodes, mesh_info_entries)):
        if mesh_info.vertex_bitmask == 0 or not isinstance(node, MeshNode3DF):
            continue

        # Read face groups
        face_groups: list[FaceGroup3DF] = []
        bs.seek(node.face_groups_off)
        for _ in range(node.face_group_count):
            face_type = bs.read_uint16()
            face_count = bs.read_uint16()
            face_flags = bs.read_uint32()
            bs.seek(8, 1)
            face_groups.append(FaceGroup3DF(
                face_type,
                face_count,
                face_flags,
            ))

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
        for face_group in face_groups:
            if face_group.face_type == 3:
                # Read triangles
                triangles.extend([
                    bs.read_vec3i()
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

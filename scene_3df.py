from io import BufferedReader
from typing import NamedTuple
import zlib

import bpy
from bpy.types import Context, EditBone, Object, VertexGroup
import math
from mathutils import Matrix
import numpy as np
import numpy.typing as npt

from .binary_reader import BinaryReader
from . import scene_3df_22
from . import scene_3df_23
from . import scene_3df_26_ds
from . import scene_3df_26_pc


class TriangleGroup3DF(NamedTuple):
    bone_indexes: tuple[int, int, int, int]
    triangles: npt.NDArray
    vertex_indices: npt.NDArray


class MeshData3DF(NamedTuple):
    vertices: npt.NDArray
    triangle_groups: list[TriangleGroup3DF]


class SceneData3DF(NamedTuple):
    nodes: list[scene_3df_22.Node3DF]
    mesh_map: dict[int, MeshData3DF]


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


def read_3df(f: BufferedReader, platform: str) -> SceneData3DF:
    # Load header chunk
    bs = BinaryReader(f.read(8))

    # Validate signature
    sig = bs.read_string_block(4)
    if sig != "3df":
        raise ValueError("Missing 3df file signature")

    # Load and read version-specific header
    version = bs.read_uint32()
    header_size = scene_3df_22.HEADER_SIZE
    if version == 22 or version == 23:
        bs = BinaryReader(f.read(header_size - 8))
        header = scene_3df_23.read_header(bs)
    elif version == 26:
        if platform == "DS":
            header_size = scene_3df_26_ds.HEADER_SIZE
            bs = BinaryReader(f.read(header_size - 8))
            header = scene_3df_26_ds.read_header(bs)
            raise NotImplementedError(
                f"Unimplemented platform {platform} for version {version}"
            )
        elif platform == "PC":
            bs = BinaryReader(f.read(header_size - 8))
            header = scene_3df_26_pc.read_header(bs)
        elif platform == "PS2":
            bs = BinaryReader(f.read(header_size - 8))
            header = scene_3df_23.read_header(bs)
            raise NotImplementedError(
                f"Unimplemented platform {platform} for version {version}"
            )
        else:
            raise NotImplementedError(
                f"Unimplemented platform {platform} for version {version}"
            )
    else:
        raise NotImplementedError(f"Unimplemented 3DF version {version}")

    # Load nodes chunk
    bs = BinaryReader(f.read(header.nodes_chunk_size))
    if header.compress_mode == 1:
        bs = decompress_chunk_stream(bs)

    # Read materials
    bs.seek(header.materials_off - header_size)
    materials = [scene_3df_22.read_material(bs) for _ in range(header.materials_count)]

    # Read nodes
    bs.seek(header.nodes_off - header_size)
    if version == 22:
        nodes = [scene_3df_22.read_node(bs) for _ in range(header.nodes_count)]
    else:
        nodes = [scene_3df_23.read_node(bs) for _ in range(header.nodes_count)]

    # Load mesh chunk
    f.seek(header_size + header.nodes_chunk_size)
    bs = BinaryReader(f.read(header.mesh_chunk_size))
    if header.compress_mode == 1:
        bs = decompress_chunk_stream(bs)

    # Read mesh info entries
    if version == 22 or version == 23:
        mesh_info_entries = [
            scene_3df_22.read_mesh_info(bs) for _ in range(header.nodes_count)
        ]
    else:
        mesh_info_entries = [
            scene_3df_26_pc.read_mesh_info(bs) for _ in range(header.nodes_count)
        ]

    # Read meshes
    mesh_data_map: dict[int, MeshData3DF] = {}
    for i, (node, mesh_info) in enumerate(zip(nodes, mesh_info_entries)):
        if mesh_info.vertex_bitmask == 0 or not isinstance(
            node, scene_3df_22.MeshNode3DF
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
        if version == 26:
            face_dtype = np.uint32
        else:
            face_dtype = np.uint16
        triangle_groups: list[TriangleGroup3DF] = []
        bs.seek(mesh_info.faces_off)
        for face_group in node.face_groups:
            if face_group.face_type == 1:
                # Read triangle strips
                tri_strip_indices = np.frombuffer(
                    bs.getbuffer(), face_dtype, face_group.face_idx_count, bs.tell()
                )
                bs.seek(tri_strip_indices.nbytes, 1)
                vertex_indices = np.unique(tri_strip_indices)
                triangle_groups.append(
                    TriangleGroup3DF(
                        face_group.bone_indexes,
                        tri_strips_to_triangles(tri_strip_indices),
                        vertex_indices,
                    )
                )
            elif face_group.face_type == 3:
                # Read triangles
                tri_indices = np.frombuffer(
                    bs.getbuffer(), face_dtype, face_group.face_idx_count, bs.tell()
                )
                bs.seek(tri_indices.nbytes, 1)
                vertex_indices = np.unique(tri_indices)
                triangle_groups.append(
                    TriangleGroup3DF(
                        face_group.bone_indexes,
                        tri_indices.reshape(-1, 3),
                        vertex_indices,
                    )
                )
            else:
                print("WARNING: Unimplemented face type " + str(face_group.face_type))

        mesh_data_map[i] = MeshData3DF(
            vertices,
            triangle_groups,
        )

    return SceneData3DF(nodes, mesh_data_map)


def import_empty_object(context: Context, node: scene_3df_22.Node3DF) -> Object:
    node_obj = bpy.data.objects.new(node.name, None)
    node_obj.empty_display_size = 0.2
    context.collection.objects.link(node_obj)

    return node_obj


def import_camera_object(context: Context, node: scene_3df_22.Node3DF):
    camera = bpy.data.cameras.new(node.name)
    camera_obj = bpy.data.objects.new(node.name, camera)
    context.collection.objects.link(camera_obj)

    return camera_obj


def import_mesh_object(
    context: Context, scene_data: SceneData3DF, node_index: int
) -> Object:
    node = scene_data.nodes[node_index]
    if node_index not in scene_data.mesh_map:
        return import_empty_object(context, node)
    mesh_data = scene_data.mesh_map[node_index]

    # Create empty objects for meshes without vertex positions
    if (
        mesh_data.vertices.dtype.names is None
        or "position" not in mesh_data.vertices.dtype.names
    ):
        print(f"WARNING: Mesh node {node.name} contains no positions")
        return import_empty_object(context, node)

    # Combine triangle groups
    triangles = np.concatenate(
        [tri_group.triangles for tri_group in mesh_data.triangle_groups]
    )

    # Import positions and triangles
    mesh = bpy.data.meshes.new(node.name)
    mesh.from_pydata(
        mesh_data.vertices["position"],
        [],
        triangles,
    )
    mesh.validate()
    mesh.update()

    # Import vertex UV layers
    if "uvs" in mesh_data.vertices.dtype.names:
        for i in range(mesh_data.vertices.dtype["uvs"].shape[0]):
            uv_layer = mesh.uv_layers.new(name=f"UV{i}")
            for loop in mesh.loops:
                uv = mesh_data.vertices["uvs"][loop.vertex_index][i]
                uv_layer.data[loop.index].uv = (uv[0], 1.0 - uv[1])

    # Import vertex normals
    if "normal" in mesh_data.vertices.dtype.names:
        mesh.normals_split_custom_set_from_vertices(mesh_data.vertices["normal"])

    # Import vertex colors
    if "color" in mesh_data.vertices.dtype.names:
        vertex_color_attr = mesh.color_attributes.new(
            name="vertex_color",
            type="BYTE_COLOR",
            domain="POINT",
        )
        vertex_color_attr.data.foreach_set(
            "color",
            mesh_data.vertices["color"].flatten(),
        )

    # Create mesh object
    mesh_obj = bpy.data.objects.new(node.name, mesh)
    context.collection.objects.link(mesh_obj)

    # Create and assign vertex groups
    if "blend_weights" not in mesh_data.vertices.dtype.names:
        return mesh_obj
    vertex_group_map: dict[int, VertexGroup] = {}
    weight_groups = mesh_data.vertices["blend_weights"]
    for tri_group in mesh_data.triangle_groups:
        bone_indexes = tri_group.bone_indexes
        for vertex_idx in tri_group.vertex_indices:
            weights: list[float] = weight_groups[vertex_idx].tolist()
            if len(weights) == 0:
                continue
            if len(weights) < 4:
                weights.append(1.0 - sum(weights))
                for _ in range(4 - len(weights)):
                    weights.append(0.0)

            for i, weight in enumerate(weights):
                if weight <= 0.0:
                    continue
                # Treat bone indexes as relative to the mesh node
                bone_node_idx = bone_indexes[i] + node_index
                if bone_node_idx not in vertex_group_map:
                    vertex_group_map[bone_node_idx] = mesh_obj.vertex_groups.new(
                        name=scene_data.nodes[bone_node_idx].name
                    )
                vertex_group_map[bone_node_idx].add(
                    [int(vertex_idx)],
                    weight,
                    "ADD",
                )

    return mesh_obj


def create_objects(
    context: Context,
    scene_data: SceneData3DF,
    object_map: dict[int, Object],
    armature_indexes: list[int],
    node_index: int,
    parent_world_transform: Matrix,
) -> None:
    node = scene_data.nodes[node_index]
    world_transform = parent_world_transform @ node.transform
    match node.type_id:
        case 1:
            # Skip bones until next pass
            obj = None
        case 0:
            obj = import_mesh_object(
                context,
                scene_data,
                node_index,
            )
            obj.matrix_world = world_transform

            # Parent mesh to armature if it has any child bones
            if obj.data is not None and any(
                scene_data.nodes[child_idx].type_id == 1
                for child_idx in scene_data.nodes[node_index].child_indexes
            ):
                armature = bpy.data.armatures.new(node.name)
                armature_obj = bpy.data.objects.new(node.name, armature)
                armature_obj.matrix_world = obj.matrix_world
                context.collection.objects.link(armature_obj)
                armature_indexes.append(node_index)

                modifier = obj.modifiers.new("Armature", "ARMATURE")
                modifier.object = armature_obj

                # Replace mesh reference with armature
                obj.parent = armature_obj
                obj = armature_obj
        case 3:
            obj = import_camera_object(context, node)
            # Flip camera
            flip_matrix = Matrix.Rotation(math.radians(180), 4, "Y")
            obj.matrix_world = world_transform @ flip_matrix
        case _:
            obj = import_empty_object(context, node)
            obj.matrix_world = world_transform

    # Store object for next two passes
    if obj is not None:
        object_map[node_index] = obj

    # Create child objects
    for child_idx in node.child_indexes:
        create_objects(
            context,
            scene_data,
            object_map,
            armature_indexes,
            child_idx,
            world_transform,
        )


def create_bones(
    scene_data: SceneData3DF,
    bone_map: dict[int, tuple[str, Object]],
    node_index: int,
    armature_object: Object,
    parent_bone: EditBone | None,
    parent_world_transform: Matrix,
) -> None:
    node = scene_data.nodes[node_index]
    transform = parent_world_transform @ node.transform

    # Create edit bone
    bone = armature_object.data.edit_bones.new(node.name)
    bone.length = 0.2
    if parent_bone:
        bone.parent = parent_bone
    bone.matrix = armature_object.matrix_world.inverted() @ transform

    # Store bone name and armature for next pass
    bone_map[node_index] = (bone.name, armature_object)

    # Create child bones
    for child_idx in node.child_indexes:
        if scene_data.nodes[child_idx].type_id == 1:
            create_bones(
                scene_data,
                bone_map,
                child_idx,
                armature_object,
                bone,
                transform,
            )


def import_3df(context: Context, scene_data: SceneData3DF) -> None:
    object_map: dict[int, Object] = {}
    armature_indexes: list[int] = []
    create_objects(
        context, scene_data, object_map, armature_indexes, 0, Matrix.Identity(4)
    )

    bone_map: dict[int, tuple[str, Object]] = {}
    for armature_idx in armature_indexes:
        armature_node = scene_data.nodes[armature_idx]
        armature_obj = object_map[armature_idx]
        context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode="EDIT")
        for root_bone_idx in armature_node.child_indexes:
            create_bones(
                scene_data,
                bone_map,
                root_bone_idx,
                armature_obj,
                None,
                Matrix.Identity(4),
            )
        bpy.ops.object.mode_set(mode="OBJECT")

    # Reparent objects
    for parent_idx, parent_node in enumerate(scene_data.nodes):
        for child_idx in parent_node.child_indexes:
            if child_idx not in object_map:
                continue
            child_obj = object_map[child_idx]
            if parent_idx in bone_map:
                bone_name, armature_obj = bone_map[parent_idx]
                parent_bone = armature_obj.data.bones[bone_name]

                child_obj.parent = armature_obj
                child_obj.parent_type = "BONE"
                child_obj.parent_bone = bone_name

                child_obj.matrix_world = (
                    armature_obj.matrix_world
                    @ parent_bone.matrix_local
                    @ scene_data.nodes[child_idx].transform
                )
            else:
                child_obj.parent = object_map[parent_idx]

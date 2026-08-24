from dataclasses import dataclass
from io import BufferedReader
import math
from typing import NamedTuple
import zlib

import bpy
from bpy.types import Context, EditBone, Object
from mathutils import Matrix
import numpy as np
import numpy.typing as npt

from .binary_reader import BinaryReader
from . import scene_3df_22
from . import scene_3df_23
from . import scene_3df_26_ds
from . import scene_3df_26_pc


class MeshData3DF(NamedTuple):
    vertices: npt.NDArray
    triangles: list[tuple[int, int, int]]


class SceneData3DF(NamedTuple):
    nodes: list[scene_3df_22.Node3DF]
    mesh_map: dict[int, MeshData3DF]


@dataclass
class ImportRecord:
    obj: Object | None = None
    armature_obj: Object | None = None
    mesh_obj: Object | None = None


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
            tri = (indices[i], indices[i + 1], indices[i + 2])
        else:
            tri = (indices[i], indices[i + 2], indices[i + 1])

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
                    triangles.extend(
                        [bs.read_vec3H() for _ in range(face_group.face_idx_count // 3)]
                    )
                else:
                    triangles.extend(
                        [bs.read_vec3I() for _ in range(face_group.face_idx_count // 3)]
                    )
            elif face_group.face_type == 1:
                # Read triangle strips
                if version == 22 or version == 23:
                    tri_strip_indices = [
                        bs.read_uint16() for _ in range(face_group.face_idx_count)
                    ]
                else:
                    tri_strip_indices = [
                        bs.read_uint32() for _ in range(face_group.face_idx_count)
                    ]
                triangles.extend(tri_strips_to_triangles(tri_strip_indices))
            else:
                print("WARNING: Unimplemented face type " + str(face_group.face_type))

        mesh_data_map[i] = MeshData3DF(
            vertices,
            triangles,
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
    context: Context,
    node: scene_3df_22.Node3DF,
    mesh_data: MeshData3DF,
) -> Object:
    # Create empty objects for meshes without vertex positions
    if (
        mesh_data.vertices.dtype.names is None
        or "position" not in mesh_data.vertices.dtype.names
    ):
        print(f"WARNING: Mesh node {node.name} contains no positions")
        return import_empty_object(context, node)

    # Import positions and triangles
    mesh = bpy.data.meshes.new(node.name)
    mesh.from_pydata(
        mesh_data.vertices["position"],
        [],
        mesh_data.triangles,
    )

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

    # Validate mesh
    mesh.validate()
    mesh.update()

    # Create mesh object
    mesh_obj = bpy.data.objects.new(node.name, mesh)
    context.collection.objects.link(mesh_obj)

    return mesh_obj


def import_edit_bone(
    context: Context,
    scene_data: SceneData3DF,
    node_index: int,
    armature_obj: Object,
    parent_bone: EditBone | None,
    parent_transform: Matrix,
    pending_armature_indexes: list[int],
) -> None:
    node = scene_data.nodes[node_index]
    edit_bone = armature_obj.data.edit_bones.new(node.name)

    # Set bone parent and armature-space transform
    edit_bone.length = 0.2
    if parent_bone is not None:
        edit_bone.parent = parent_bone
    edit_bone.matrix = parent_transform @ node.transform

    for child_index in node.child_indexes:
        child_node = scene_data.nodes[child_index]
        if child_node.type_id == 1:
            import_edit_bone(
                context,
                scene_data,
                child_index,
                armature_obj,
                edit_bone,
                edit_bone.matrix,
                pending_armature_indexes,
            )
        else:
            # Wait until next pass to import bone-parented nodes
            pending_armature_indexes.append(node_index)


def calculate_world_transforms(
    scene_data: SceneData3DF,
) -> dict[int, Matrix]:
    world_transforms: dict[int, Matrix] = {}

    def visit(node_index: int, parent_world: Matrix) -> None:
        node = scene_data.nodes[node_index]

        world = parent_world @ node.transform
        world_transforms[node_index] = world

        for child_index in node.child_indexes:
            visit(child_index, world)

    visit(0, Matrix.Identity(4))
    return world_transforms


def has_bone_children(
    scene_data: SceneData3DF,
    node_index: int,
) -> bool:
    return any(
        scene_data.nodes[child_index].type_id == 1
        for child_index in scene_data.nodes[node_index].child_indexes
    )


def create_node_objects(
    context: Context,
    scene_data: SceneData3DF,
    world_transforms: dict[int, Matrix],
) -> tuple[
    dict[int, ImportRecord],
    list[int],
]:
    node_records: dict[int, ImportRecord] = {}
    armature_node_indexes: list[int] = []

    for node_index, node in enumerate(scene_data.nodes):
        if node.type_id == 1:
            # Bone nodes do not have Blender Objects.
            continue

        if node.type_id == 0:
            if node_index not in scene_data.mesh_map:
                obj = import_empty_object(context, node)
                node_records[node_index] = ImportRecord(obj=obj)
                continue

            mesh_obj = import_mesh_object(
                context,
                node,
                scene_data.mesh_map[node_index],
            )

            if has_bone_children(scene_data, node_index):
                armature_data = bpy.data.armatures.new(node.name)
                armature_obj = bpy.data.objects.new(
                    node.name,
                    armature_data,
                )
                context.collection.objects.link(armature_obj)

                armature_obj.matrix_world = world_transforms[node_index]

                # Keep the actual mesh under the armature object.
                mesh_obj.parent = armature_obj
                mesh_obj.matrix_world = world_transforms[node_index]

                node_records[node_index] = ImportRecord(
                    obj=armature_obj,
                    armature_obj=armature_obj,
                    mesh_obj=mesh_obj,
                )

                armature_node_indexes.append(node_index)

            else:
                mesh_obj.matrix_world = world_transforms[node_index]

                node_records[node_index] = ImportRecord(
                    obj=mesh_obj,
                )

        elif node.type_id == 3:
            obj = import_camera_object(context, node)
            obj.matrix_world = world_transforms[node_index]
            # Flip camera objects
            obj.rotation_euler = (0.0, math.radians(180.0), 0.0)

            node_records[node_index] = ImportRecord(obj=obj)

        else:
            obj = import_empty_object(context, node)
            obj.matrix_world = world_transforms[node_index]

            node_records[node_index] = ImportRecord(obj=obj)

    return node_records, armature_node_indexes


def import_armature_bones(
    context: Context,
    scene_data: SceneData3DF,
    armature_node_index: int,
    armature_obj: Object,
    world_transforms: dict[int, Matrix],
    bone_records: dict[int, tuple[Object, str]],
) -> None:
    context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)

    bpy.ops.object.mode_set(mode="EDIT")

    try:
        armature_inverse = armature_obj.matrix_world.inverted()

        def create_bone(
            bone_node_index: int,
            parent_edit_bone: EditBone | None,
        ) -> None:
            node = scene_data.nodes[bone_node_index]

            edit_bone = armature_obj.data.edit_bones.new(node.name)

            if parent_edit_bone is not None:
                edit_bone.parent = parent_edit_bone

            bone_matrix = armature_inverse @ world_transforms[bone_node_index]

            edit_bone.length = 0.2
            edit_bone.matrix = bone_matrix

            # Do not retain the EditBone itself.
            bone_records[bone_node_index] = (
                armature_obj,
                edit_bone.name,
            )

            for child_index in node.child_indexes:
                child_node = scene_data.nodes[child_index]

                if child_node.type_id == 1:
                    create_bone(
                        child_index,
                        edit_bone,
                    )

        armature_node = scene_data.nodes[armature_node_index]

        for child_index in armature_node.child_indexes:
            child_node = scene_data.nodes[child_index]

            if child_node.type_id == 1:
                create_bone(child_index, None)

    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


def import_node_object(
    context: Context,
    scene_data: SceneData3DF,
    node_index: int,
) -> Object:
    node = scene_data.nodes[node_index]
    match node.type_id:
        case 0:
            if node_index in scene_data.mesh_map:
                node_obj = import_mesh_object(
                    context, node, scene_data.mesh_map[node_index]
                )
                # Replace mesh with armature if it has any child bones
                skinned = False
                for child_index in node.child_indexes:
                    if scene_data.nodes[child_index].type_id == 1:
                        skinned = True
                if skinned:
                    armature = bpy.data.armatures.new(node.name)
                    armature_obj = bpy.data.objects.new(node.name, armature)
                    context.collection.objects.link(armature_obj)
                    node_obj.parent = armature_obj
                    node_obj = armature_obj
            else:
                print(f"WARNING: No mesh info entry found for node {node.name}")
                node_obj = import_empty_object(context, node)
        case 3:
            node_obj = import_camera_object(context, node)
        case _:
            node_obj = import_empty_object(context, node)

    return node_obj


def build_parent_map(
    scene_data: SceneData3DF,
) -> dict[int, int]:
    parent_map: dict[int, int] = {}

    for parent_index, node in enumerate(scene_data.nodes):
        for child_index in node.child_indexes:
            if child_index in parent_map:
                raise ValueError(f"Node {child_index} has multiple parents")

            parent_map[child_index] = parent_index

    return parent_map


def apply_object_parenting(
    scene_data: SceneData3DF,
    node_records: dict[int, ImportRecord],
    bone_records: dict[int, tuple[Object, str]],
    world_transforms: dict[int, Matrix],
) -> None:
    parent_map = build_parent_map(scene_data)

    for node_index, record in node_records.items():
        parent_index = parent_map.get(node_index)

        if parent_index is None:
            record.obj.matrix_world = world_transforms[node_index]
            continue

        parent_node = scene_data.nodes[parent_index]

        if parent_node.type_id == 1:
            # Source parent is a bone.
            armature_obj, bone_name = bone_records[parent_index]

            record.obj.parent = armature_obj
            record.obj.parent_type = "BONE"
            record.obj.parent_bone = bone_name

        else:
            # Normal object parenting.
            parent_obj = node_records[parent_index].obj

            record.obj.parent = parent_obj
            record.obj.parent_type = "OBJECT"

        # Restore the source transform after setting the parent.
        record.obj.matrix_world = world_transforms[node_index]


def import_3df(
    context: Context,
    scene_data: SceneData3DF,
) -> None:
    world_transforms = calculate_world_transforms(scene_data)
    node_records, armature_node_indexes = create_node_objects(
        context,
        scene_data,
        world_transforms,
    )

    bone_records: dict[int, tuple[Object, str]] = {}

    for armature_node_index in armature_node_indexes:
        armature_obj = node_records[armature_node_index].armature_obj
        if armature_obj is None:
            continue

        import_armature_bones(
            context,
            scene_data,
            armature_node_index,
            armature_obj,
            world_transforms,
            bone_records,
        )

    apply_object_parenting(
        scene_data,
        node_records,
        bone_records,
        world_transforms,
    )

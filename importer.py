import math

import bpy
from bpy.types import Context, EditBone, Image, Material, Object, PoseBone, VertexGroup
from mathutils import Matrix
import numpy as np

from . import scene_3df_20

from .reader import SceneData3DF


class Importer3DF:
    def __init__(self, context: Context) -> None:
        self.context: Context = context
        self.images: list[Image] = []
        self.materials: list[Material] = []

    def import_empty_object(self, node: scene_3df_20.Node3DF) -> Object:
        node_obj = bpy.data.objects.new(node.name, None)
        node_obj.empty_display_size = 0.2
        self.context.collection.objects.link(node_obj)

        return node_obj

    def import_camera_object(self, node: scene_3df_20.Node3DF):
        camera = bpy.data.cameras.new(node.name)
        camera_obj = bpy.data.objects.new(node.name, camera)
        self.context.collection.objects.link(camera_obj)

        return camera_obj

    def import_mesh_object(self, scene_data: SceneData3DF, node_index: int) -> Object:
        node = scene_data.nodes[node_index]
        if node_index not in scene_data.mesh_map:
            return self.import_empty_object(node)
        mesh_data = scene_data.mesh_map[node_index]

        # Create empty objects for meshes without vertex positions
        if (
            mesh_data.vertices.dtype.names is None
            or "position" not in mesh_data.vertices.dtype.names
        ):
            print(f"WARNING: Mesh node {node.name} contains no positions")
            return self.import_empty_object(node)

        # Combine triangle groups
        if len(mesh_data.triangle_groups) > 0:
            triangles = np.concatenate(
                [tri_group.triangles for tri_group in mesh_data.triangle_groups]
            )
        else:
            triangles = []

        # Import positions and triangles
        mesh = bpy.data.meshes.new(node.name)
        mesh.from_pydata(
            mesh_data.vertices["position"],
            [],
            triangles,
        )
        mesh.validate()
        mesh.update()

        # Assign materials
        for material in self.materials:
            mesh.materials.append(material)
        tri_cursor = 0
        for tri_group in mesh_data.triangle_groups:
            tri_count = len(tri_group.triangles)
            for i in range(tri_count):
                mesh.polygons[tri_cursor + i].material_index = tri_group.material_index
            tri_cursor += tri_count

        # Import vertex UV layers
        if "uvs" in mesh_data.vertices.dtype.names:
            for i in range(mesh_data.vertices.dtype["uvs"].shape[0]):
                uv_layer = mesh.uv_layers.new(name=f"UV{i}")
                for loop in mesh.loops:
                    uv = mesh_data.vertices["uvs"][loop.vertex_index][i]
                    uv_layer.data[loop.index].uv = uv

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
        self.context.collection.objects.link(mesh_obj)

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
                    bone_idx = bone_indexes[i]
                    if bone_idx not in vertex_group_map:
                        bone_name = None
                        for j in range(node_index, len(scene_data.nodes)):
                            bone_node = scene_data.nodes[j]
                            if bone_node.internal_index == bone_idx:
                                bone_name = bone_node.name
                                break
                        if bone_name is None:
                            print(
                                "WARNING: Failed to find node with index "
                                f"{bone_idx} for vertex group"
                            )
                            bone_name = str(bone_idx)
                        vertex_group_map[bone_idx] = mesh_obj.vertex_groups.new(
                            name=bone_name
                        )
                    vertex_group_map[bone_idx].add(
                        [int(vertex_idx)],
                        weight,
                        "ADD",
                    )

        return mesh_obj

    def create_objects(
        self,
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
                obj = self.import_mesh_object(
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
                    self.context.collection.objects.link(armature_obj)
                    armature_indexes.append(node_index)

                    modifier = obj.modifiers.new("Armature", "ARMATURE")
                    modifier.object = armature_obj

                    # Replace mesh reference with armature
                    obj.parent = armature_obj
                    obj = armature_obj
            case 3:
                obj = self.import_camera_object(node)
                # Flip camera
                flip_matrix = Matrix.Rotation(math.radians(180), 4, "Y")
                obj.matrix_world = world_transform @ flip_matrix
            case _:
                obj = self.import_empty_object(node)
                obj.matrix_world = world_transform

        # Store object for next two passes
        if obj is not None:
            object_map[node_index] = obj

        # Create child objects
        for child_idx in node.child_indexes:
            self.create_objects(
                scene_data,
                object_map,
                armature_indexes,
                child_idx,
                world_transform,
            )

    def create_bones(
        self,
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
                self.create_bones(
                    scene_data,
                    bone_map,
                    child_idx,
                    armature_object,
                    bone,
                    transform,
                )

    def import_material(self, material_data: scene_3df_20.Material3DF) -> Material:
        mat = bpy.data.materials.new(material_data.name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        links = mat.node_tree.links

        # Import property nodes
        bsdf.inputs["Base Color"].default_value = material_data.diffuse_color
        for prop in material_data.properties:
            if prop.type_id == 0:
                img_node = nodes.new("ShaderNodeTexImage")
                img_node.image = self.images[prop.value]
                img_node.location = (-600.0, 100.0)
                links.new(img_node.outputs["Color"], bsdf.inputs["Base Color"])
                links.new(img_node.outputs["Alpha"], bsdf.inputs["Alpha"])
                break

        return mat

    def import_scene(self, scene_data: SceneData3DF) -> None:
        # Import images/textures
        for i, texture in enumerate(scene_data.textures):
            img = bpy.data.images.new(
                f"image_{i}", texture.width, texture.height, alpha=True
            )
            img.pixels = texture.pixels
            img.update()
            self.images.append(img)

        # Import materials
        self.materials = [
            self.import_material(mat_data) for mat_data in scene_data.materials
        ]

        # Create objects
        object_map: dict[int, Object] = {}
        armature_indexes: list[int] = []
        self.create_objects(
            scene_data,
            object_map,
            armature_indexes,
            0,
            Matrix.Identity(4),
        )

        # Create bones
        bone_map: dict[int, tuple[str, Object]] = {}
        for armature_idx in armature_indexes:
            armature_node = scene_data.nodes[armature_idx]
            armature_obj = object_map[armature_idx]
            self.context.view_layer.objects.active = armature_obj
            bpy.ops.object.mode_set(mode="EDIT")
            for root_bone_idx in armature_node.child_indexes:
                self.create_bones(
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

        # Import bone animation tracks
        self.context.scene.render.fps = 30
        for node_idx, node in enumerate(scene_data.nodes):
            if node_idx not in bone_map:
                continue
            bone_name, armature_obj = bone_map[node_idx]
            self.context.view_layer.objects.active = armature_obj
            bpy.ops.object.mode_set(mode="POSE")
            bone: PoseBone = armature_obj.pose.bones[bone_name]
            for track in node.tracks:
                if track.type_id <= 2:
                    axis = track.type_id
                    for key in track.keys:
                        frame = round(key.time * 30)
                        bone.location[axis] = key.value
                        bone.keyframe_insert(
                            "location",
                            index=axis,
                            frame=frame,
                        )
                elif track.type_id <= 5:
                    axis = track.type_id - 3
                    for key in track.keys:
                        frame = round(key.time * 30)
                        bone.rotation_mode = "XYZ"
                        bone.rotation_euler[axis] = key.value
                        bone.keyframe_insert(
                            "rotation_euler",
                            index=axis,
                            frame=frame,
                        )
                else:
                    axis = track.type_id - 6
                    for key in track.keys:
                        frame = round(key.time * 30)
                        bone.scale[axis] = key.value
                        bone.keyframe_insert(
                            "scale",
                            index=axis,
                            frame=frame,
                        )
            bpy.ops.object.mode_set(mode="OBJECT")

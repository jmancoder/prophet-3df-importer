from .binary_reader import BinaryReader
from . import scene_3df_20
from . import scene_3df_22


def read_node(bs: BinaryReader) -> scene_3df_20.Node3DF:
    node_name = bs.read_string_block(16)
    node_type = bs.read_uint32()
    bs.read_int32()
    bs.read_int32()
    child_index_count = bs.read_int32()
    internal_idx = bs.read_int32()
    child_index_off = bs.read_uint32()
    bs.read_int32()
    unk_vec_0 = bs.read_vec3f()
    unk_vec_1 = bs.read_vec3f()
    unk_floats_off = bs.read_uint32()
    bone_group_count = bs.read_uint32()
    bone_group_off = bs.read_uint32()
    transform_type = bs.read_uint32()
    transform = bs.read_loc_rot_scale()
    bs.read_vec3f()
    bs.read_int32()
    bs.read_float()
    bounds_min = bs.read_vec3f()
    bounds_max = bs.read_vec3f()
    bs.seek(68, 1)
    face_groups_off = bs.read_uint32()
    bs.seek(32, 1)

    # Read child indexes
    if child_index_count > 0:
        node_end_off = bs.tell()
        bs.seek(child_index_off - scene_3df_22.HEADER_SIZE)
        child_indexes = [bs.read_uint32() for _ in range(child_index_count)]
        bs.seek(node_end_off)
    else:
        child_indexes = []

    match node_type:
        case 0:
            vertex_count = bs.read_uint32()
            face_idx_count = bs.read_uint32()
            face_groups_count = bs.read_uint32()
            bs.seek(92, 1)

            # Read face groups
            if face_groups_off > 0:
                node_end_off = bs.tell()
                bs.seek(face_groups_off - scene_3df_22.HEADER_SIZE)
                face_groups = [
                    scene_3df_20.read_face_group(bs) for _ in range(face_groups_count)
                ]
                bs.seek(node_end_off)
            else:
                face_groups = []

            return scene_3df_20.MeshNode3DF(
                node_name,
                node_type,
                internal_idx,
                child_indexes,
                transform_type,
                transform,
                vertex_count,
                face_idx_count,
                face_groups,
            )
        case 1:
            unk_floats = [bs.read_float() for _ in range(13)]
            bs.seek(52, 1)

            if bone_group_count > 0:
                node_end_off = bs.tell()
                bs.seek(bone_group_off - scene_3df_22.HEADER_SIZE)
                bone_groups = [
                    scene_3df_20.read_bone_group(bs) for _ in range(bone_group_count)
                ]
                bs.seek(node_end_off)
            else:
                bone_groups = []

            return scene_3df_20.BoneNode3DF(
                node_name,
                node_type,
                internal_idx,
                child_indexes,
                transform_type,
                transform,
                unk_floats,
                bone_groups,
            )
        case _:
            bs.seek(104, 1)

            return scene_3df_20.Node3DF(
                node_name,
                node_type,
                internal_idx,
                child_indexes,
                transform_type,
                transform,
            )

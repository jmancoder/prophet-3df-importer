from .binary_reader import BinaryReader
from . import scene_3df_20


def read_node(bs: BinaryReader, header_size: int) -> scene_3df_20.Node3DF:
    name = bs.read_string_block(16)
    type_id = bs.read_uint32()
    flags = bs.read_uint32()
    bs.read_int32()
    child_index_count = bs.read_int32()
    internal_idx = bs.read_int32()
    child_index_off = bs.read_uint32()
    bs.read_int32()
    unk_vec_0 = bs.read_vec3f()
    unk_vec_1 = bs.read_vec3f()
    unk_point_off = bs.read_uint32()
    track_count = bs.read_uint32()
    track_off = bs.read_uint32()
    transform_type = bs.read_uint32()
    if transform_type != 0:
        raise NotImplementedError(f"Unimplemented transform type {transform_type}")
    transform = bs.read_loc_rot_scale()
    bs.read_vec3f()
    bs.read_int32()
    bs.read_float()
    bounds_min = bs.read_vec3f()
    bounds_max = bs.read_vec3f()
    bs.seek(68, 1)
    face_group_off = bs.read_uint32()
    bs.seek(32, 1)

    # Read child indexes
    if child_index_count > 0:
        node_end_off = bs.tell()
        bs.seek(child_index_off - header_size)
        child_indexes = [bs.read_uint32() for _ in range(child_index_count)]
        bs.seek(node_end_off)
    else:
        child_indexes = []

    # Read animation tracks
    if track_count > 0:
        node_end_off = bs.tell()
        bs.seek(track_off - header_size)
        tracks = [
            scene_3df_20.read_track(bs, header_size)
            for _ in range(track_count)
        ]
        bs.seek(node_end_off)
    else:
        tracks = []

    match type_id:
        case 0:
            vertex_count = bs.read_uint32()
            face_idx_count = bs.read_uint32()
            face_groups_count = bs.read_uint32()
            bs.seek(92, 1)

            # Read face groups
            if face_group_off > 0:
                node_end_off = bs.tell()
                bs.seek(face_group_off - header_size)
                face_groups = [
                    scene_3df_20.read_face_group(bs) for _ in range(face_groups_count)
                ]
                bs.seek(node_end_off)
            else:
                face_groups = []
            return scene_3df_20.MeshNode3DF(
                name,
                type_id,
                flags,
                internal_idx,
                child_indexes,
                transform_type,
                transform,
                tracks,
                vertex_count,
                face_idx_count,
                face_groups,
            )
        case 1:
            unk_float = bs.read_float()
            bone_transform = bs.read_matrix_3x4()
            bs.seek(52, 1)
            return scene_3df_20.BoneNode3DF(
                name,
                type_id,
                flags,
                internal_idx,
                child_indexes,
                transform_type,
                transform,
                tracks,
                unk_float,
                bone_transform,
            )
        case _:
            bs.seek(104, 1)
            return scene_3df_20.Node3DF(
                name,
                type_id,
                flags,
                internal_idx,
                child_indexes,
                transform_type,
                transform,
                tracks,
            )

from typing import List, Optional, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break

        motion_data = []
        for line in lines[i + 1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1, -1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def pose_from_bvh_lines(lines: List[str],
                        joint_names: List[str],
                        joint_parents: List[int],
                        joint_offsets: List[List[float]],
                        i: int = 0,
                        parent_index: int = -1) -> int:
    my_joint_name_read = False
    my_name_index = len(joint_names)

    line = lines[i].strip()
    while line != "}":

        if line.startswith("JOINT") or line.startswith("ROOT"):
            if my_joint_name_read:
                i = pose_from_bvh_lines(lines,
                                        joint_names,
                                        joint_parents,
                                        joint_offsets,
                                        i,
                                        my_name_index)
            else:
                joint_parents.append(parent_index)

                joint_names.append(line.split()[-1])
                my_joint_name_read = True

        elif line.startswith("OFFSET"):
            joint_offsets.append([float(offset) for offset in line.split()[-3:]])

        elif line.startswith("End"):
            joint_parents.append(my_name_index)

            joint_names.append(joint_names[my_name_index] + "_end")
            joint_offsets.append([float(offset) for offset in lines[i + 2].split()[-3:]])
            i += 3

        i += 1
        line = lines[i].strip()

    return i


def part1_calculate_T_pose(bvh_file_path: str) -> Tuple[List[str], List[int], np.ndarray]:
    """请填写以下内容
        输入： bvh 文件路径
        输出:
            joint_name: List[str]，字符串列表，包含着所有关节的名字
            joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
            joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_names = []
    joint_parents = []
    joint_offsets = []

    with open(bvh_file_path, 'r') as bvh:
        lines = bvh.readlines()

        pose_from_bvh_lines(lines,
                            joint_names,
                            joint_parents,
                            joint_offsets)

    return joint_names, joint_parents, np.array(joint_offsets, dtype=np.float64)


def pose_joint_orientations(joint_names: List[str],
                            joint_parents: List[int],
                            pose: np.ndarray) -> List[R]:
    rotations = pose.reshape(-1, 3)[1:]
    orientations = [R.from_euler("XYZ", rotations[0], degrees=True)]
    orientation_indices = np.empty_like(joint_names, dtype=np.int64)
    orientation_indices[0] = 0

    joint_index = 1
    rotation_index = 1

    while joint_index < len(joint_names):
        orientations.append(orientations[orientation_indices[joint_parents[joint_index]]]
                            * R.from_euler("XYZ", rotations[rotation_index], degrees=True))
        orientation_indices[joint_index] = rotation_index

        joint_index += 1
        if joint_names[joint_index].endswith("end"):
            joint_index += 1
        rotation_index += 1

    return orientations


def pose_joint_positions(joint_parents: List[int],
                         joint_offsets: np.ndarray,
                         joint_orientations: List[R]) -> List[np.ndarray]:
    positions = [joint_offsets[0]]

    for i in range(1, len(joint_parents)):
        positions[joint_parents[i]] + joint_orientations[i].apply(joint_offsets[i])

    return positions


def part2_forward_kinematics(joint_names: List[str],
                             joint_parents: List[int],
                             joint_offsets: np.ndarray,
                             motion_data: np.ndarray,
                             frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_orientations = pose_joint_orientations(joint_parents, motion_data[frame_id])
    return (np.stack(pose_joint_positions(joint_parents, joint_offsets, joint_orientations), axis=0, dtype=np.float64),
            np.stack([o.as_quat() for o in joint_orientations], axis=0, dtype=np.float64))


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = None
    return motion_data

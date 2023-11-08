from typing import List
import numpy as np
from scipy.spatial.transform import Rotation as R


def update_chain_orientations(path: List[int],
                              start: int,
                              rotation,
                              joint_orientations: np.ndarray) -> None:
    """
    Update the orientations of joints from index "start" to the end effector (exclusive) in a chain of joints defined by "path".
    """
    for i in path[start:-1]:
        joint_orientations[i] = (rotation * R.from_quat(joint_orientations[i])).as_quat()


def joint_offsets_from_positions(joint_positions: np.ndarray,
                                 joint_parents: List[int]) -> np.ndarray:
    offsets = np.empty_like(joint_positions)
    offsets[0] = np.zeros_like(offsets[0])

    for i in range(1, joint_positions.shape[0]):
        offsets[i] = joint_positions[i] - joint_positions[joint_parents[i]]

    return offsets


def part1_inverse_kinematics(meta_data,
                             joint_positions: np.ndarray,
                             joint_orientations: np.ndarray,
                             target_pose: np.ndarray):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """

    path, path_names, end_to_root, fixed_to_root = meta_data


    return joint_positions, joint_orientations


def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """

    return joint_positions, joint_orientations


def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """

    return joint_positions, joint_orientations

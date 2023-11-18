from typing import List, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R
from lab1.Lab1_FK_answers import *
from lab1.inverse_kinematics_methods.methods import *


def part1_inverse_kinematics(meta_data,
                             joint_positions: np.ndarray,
                             joint_orientations: np.ndarray,
                             target_end: np.ndarray):
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
    start2end, _, root_index = meta_data.get_path_from_root_to_end()
    joint_offsets = joint_offsets_from_positions(meta_data.joint_initial_position, meta_data.joint_parent)

    return gradient_descent(joint_positions,
                            joint_orientations,
                            meta_data.joint_parent,
                            joint_offsets,
                            root_index,
                            start2end,
                            target_end,
                            learning_rate=0.5, # When the manipulator has many joints, smaller learning rate provides stability
                            max_iterations=40)
    # return cyclic_coordinate_descent(meta_data.get_path_from_root_to_end()[0],
    #                                  target_end,
    #                                  meta_data.joint_initial_position,
    #                                  meta_data.joint_parent,
    #                                  joint_positions,
    #                                  joint_orientations,
    #                                  max_iterations=10,
    #                                  order=1)


def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    start2end, _, root_index = meta_data.get_path_from_root_to_end()
    joint_offsets = joint_offsets_from_positions(meta_data.joint_initial_position, meta_data.joint_parent)

    return gradient_descent(joint_positions,
                            joint_orientations,
                            meta_data.joint_parent,
                            joint_offsets,
                            root_index,
                            start2end,
                            np.array([joint_positions[0, 0] + relative_x,
                                      target_height,
                                      joint_positions[0, 2] + relative_z]),
                            learning_rate=1,
                            max_iterations=20)


def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """

    return joint_positions, joint_orientations

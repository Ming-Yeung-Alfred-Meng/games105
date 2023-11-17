from typing import List, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R
from lab1.inverse_kinematics_methods.scripts import *
from lab1.Lab1_FK_answers import *


# TODO: no longer working after improvements on its helper functions. More work to do.
#  The quality of the implementation of ccd and its helper functions is overall poor.
def cyclic_coordinate_descent(path: List[int],
                              target: np.ndarray,
                              initial_joint_positions: np.ndarray,
                              joint_parents: List[int],
                              joint_positions: np.ndarray,
                              joint_orientations: np.ndarray,
                              max_iterations: int = 20,
                              max_error: float = 0.01,
                              order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    assert order == 1 or order == -1
    assert 2 <= len(path)

    joint_offsets = joint_offsets_from_positions(initial_joint_positions,
                                                 joint_parents)
    joint_orientations = [R.from_quat(o) for o in joint_orientations]

    iteration_count = 0
    if order == 1:
        i = 0
    else:
        i = len(path) - 2

    while iteration_count < max_iterations and max_error < np.linalg.norm(joint_positions[path[-1]] - target):
        joint_to_end = joint_positions[path[-1]] - joint_positions[path[i]]
        joint_to_target = target - joint_positions[path[i]]
        axis_of_rotation = np.cross(joint_to_end, joint_to_target)
        angle_of_rotation = np.arccos(np.dot(joint_to_end, joint_to_target)
                                      / (np.linalg.norm(joint_to_end) * np.linalg.norm(joint_to_target)))

        update_chain_orientations(path,
                                  i,
                                  R.from_rotvec(
                                      angle_of_rotation * axis_of_rotation / np.linalg.norm(axis_of_rotation)),
                                  joint_orientations)

        joint_positions = pose_joint_positions(initial_joint_positions[0],
                                               joint_parents,
                                               joint_offsets,
                                               joint_orientations)

        iteration_count += 1
        i = (i + order) % (len(path) - 1)

    return joint_positions, np.stack([o.as_quat() for o in joint_orientations])


def gradient_descent(positions: np.ndarray,
                     joint_orientations: np.ndarray,
                     parents: List[int],
                     joint_offsets: np.ndarray,
                     root_index: int,
                     start2end: List[int],
                     target: np.ndarray,
                     learning_rate: float = 1,
                     max_iterations: int = 10,
                     max_error: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    orientations, indices = link_orientations(joint_orientations, parents, start2end, root_index)
    orientations = R.from_quat(orientations)
    # the same orientation may occur twice, as desired.
    links = manipulator_links(positions, start2end)
    # include all links connect the root, as desired.
    error, error_norm = forward(positions, start2end, links, target)

    iteration_count = 0
    while iteration_count < max_iterations and max_error < error_norm:
        loss_gradient = backward(links, root_index, error)

        links, orientations = step(orientations, links, loss_gradient, learning_rate)

        error, error_norm = forward(positions, start2end, links, target)

        iteration_count += 1

    joint_orientations = update_joint_orientations(joint_orientations, orientations.as_quat(), indices)

    return (pose_joint_positions(link_position(positions, start2end, links, root_index),
                                 parents, joint_offsets, joint_orientations),
            joint_orientations)

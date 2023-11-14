import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from scipy.spatial.transform import Rotation as R
from lab1.Lab1_FK_answers import *


def plot_manipulator(positions: np.ndarray,
                     parents: List[int],
                     path: List[int],
                     target: np.ndarray,
                     names: List[str]) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    offsets = positions - positions[parents]
    offsets[0].fill(0)

    ax.quiver(positions[parents][path, 0],
              positions[parents][path, 1],
              positions[parents][path, 2],
              offsets[path, 0],
              offsets[path, 1],
              offsets[path, 2], label=names)

    ax.scatter(target[0], target[1], target[2], c='r', marker='o', label='target')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(0, 5)  # Set X-axis limits
    ax.set_ylim(0, 5)  # Set Y-axis limits
    ax.set_zlim(0, 5)  # Set Z-axis limits

    ax.legend()


def gradient_descent(joint_positions: np.ndarray,
                     joint_orientations: np.ndarray,
                     joint_parents: List[int],
                     joint_offsets: np.ndarray,
                     initial_root: np.ndarray,
                     path: List[int],
                     target: np.ndarray,
                     learning_rate: int = 0.1) -> Tuple[np.ndarray, np.ndarray]:

    current_joint_offsets = (joint_positions - joint_positions[joint_parents])
    current_joint_offsets[0].fill(0.)

    I = np.eye(3)
    jacobian_transpose = np.cross(I, current_joint_offsets[path[1:], None, :]).reshape(-1, 3)
    error = joint_positions[path[-1]] - target

    loss_gradient = jacobian_transpose @ error

    update = - learning_rate * loss_gradient

    rotations = R.from_euler("XYZ", update.reshape(-1, 3))

    result_orientations = R.from_quat(joint_orientations)
    result_orientations[path[:-1]] = rotations * result_orientations[path[:-1]]

    result_positions = pose_joint_positions(initial_root,
                                            joint_parents,
                                            joint_offsets,
                                            result_orientations)

    return result_positions, result_orientations.as_quat()


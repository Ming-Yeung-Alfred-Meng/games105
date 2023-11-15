import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from scipy.spatial.transform import Rotation as R
from lab1.Lab1_FK_answers import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def plot_manipulator(positions: np.ndarray,
                     parents: List[int],
                     path: List[int],
                     points: np.ndarray,
                     names: List[str] = None,
                     xlim: Tuple[int, int] = (0, 5),
                     ylim: Tuple[int, int] = (0, 5),
                     zlim: Tuple[int, int] = (0, 5)) -> None:
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

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_xlim(ylim[0], ylim[1])
    ax.set_xlim(zlim[0], zlim[1])
    ax.legend()
    ax.view_init(elev=20, azim=10, vertical_axis='y')


def jacobian_transpose(joint_positions: np.ndarray,
                       joint_parents: List[int],
                       path: List[int]) -> np.ndarray:
    current_joint_offsets = joint_positions - joint_positions[joint_parents]
    current_joint_offsets[0].fill(0.)

    return np.cross(np.eye(3), current_joint_offsets[path[1:], None, :]).reshape(-1, 3)


def gradient_descent(joint_positions: np.ndarray,
                     joint_orientations: np.ndarray,
                     joint_parents: List[int],
                     joint_offsets: np.ndarray,
                     initial_root: np.ndarray,
                     path: List[int],
                     target: np.ndarray,
                     learning_rate: float = 1,
                     max_iterations: int = 10,
                     max_error: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    iteration_count = 0
    while iteration_count < max_iterations and max_error < np.linalg.norm(joint_positions[path[-1]] - target):
        loss_gradient = jacobian_transpose(joint_positions, joint_parents, path) @ (joint_positions[path[-1]] - target)

        change_in_angles = - learning_rate * loss_gradient

        change_in_orientations = R.from_euler("XYZ", change_in_angles.reshape(-1, 3))

        new_orientations = R.from_quat(joint_orientations)
        new_orientations[path[:-1]] = change_in_orientations * new_orientations[path[:-1]]
        # Here is wrong, i don't think this is how you update orientations. you might need to use parent index

        joint_positions = pose_joint_positions(initial_root,
                                               joint_parents,
                                               joint_offsets,
                                               new_orientations)
        iteration_count += 1

    return joint_positions, new_orientations.as_quat()

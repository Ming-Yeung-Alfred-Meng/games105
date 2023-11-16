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


# def jacobian_transpose(joint_positions: np.ndarray,
#                        joint_parents: List[int],
#                        path: List[int]) -> np.ndarray:
#     current_joint_offsets = joint_positions - joint_positions[joint_parents]
#     current_joint_offsets[0].fill(0.)
#
#     return np.cross(np.eye(3), current_joint_offsets[path[1:], None, :]).reshape(-1, 3)


def jacobian_transpose(links: np.ndarray,
                       root_index: int):
    """
    Compute the transpose of the Jacobian of the manipulator formed by "links". Assume the links use Euler rotations
    in the same canonical coordinate system.
    @param links: m x 3 numpy array of links.
    @return: 3m x 3 numpy array. Transpose of the Jacobian of end effector position w.r.t. all joint angles.
    """
    result = np.cross(np.eye(3), links[:, None, :]).reshape(-1, 3)
    if 0 < root_index < links.shape[0]:
        result[3 * (root_index - 1): 3 * (root_index - 1) + 3] += result[3 * root_index: 3 * root_index + 3]
        result[3 * root_index: 3 * root_index + 3] = result[3 * (root_index - 1): 3 * (root_index - 1) + 3]

    return result


def backward(links: np.ndarray,
             root_index: int,
             error: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the loss function i.e. L2 distance between end effector and target w.r.t. joint angles.
    @param links: m x 3 numpy array of links.
    @param error: 3 numpy array of displacement from target to end effector, i.e. end - target.
    @return: 3m numpy array of loss gradient.
    """
    return jacobian_transpose(links, root_index) @ error


def step(loss_gradient,
         learning_rate,
         links: np.ndarray,
         orientations):
    """
    Update links and their orientations.
    @param loss_gradient: m array of gradient of loss function w.r.t. joint angles.
    @param learning_rate: amount to step in the direction of loss_gradient.
    @param links: m x 3 array of links in the manipulator.
    @param orientations: m scipy rotations defining the orientations of links.
    @return: updated links and orientations.
    """
    rotations = R.from_euler("XYZ", (- learning_rate * loss_gradient).reshape(-1, 3))

    return rotations.apply(links), rotations * orientations


def forward(joint_positions: np.ndarray,
            start2end: List[int],
            links: np.ndarray,
            target: np.ndarray):
    """
    Compute the output of the loss function, i.e. displacement (error) from target to end effector position, and its L2
    norm.
    @param joint_positions: n x 3 numpy array of positions of all joints.
    @param start2end: indices of joints in the manipulator.
    @param links: m x 3 numpy array of links.
    @param target: 3 array of target position.
    @return: 3 numpy array of end effector position.
    """
    error = link_position(joint_positions, start2end, links) - target
    return error, np.linalg.norm(error)


def link_orientations(orientations: np.ndarray,
                      parents: List[int],
                      start2end: List[int],
                      root_index: int):
    """
    Return scipy rotations representing the orientations of links in the manipulator, and their indices into the
    array of all orientations.
    @param orientations: n x 4 numpy array of orientations of all joints.
    @param parents: parents[i] stores the index of i-th joint's parent.
    @param start2end: indices of joints in the manipulator.
    @param root_index: index of the root, i.e. 0 in "start2end".
    @return: m scipy rotations of orientations of joints in the manipulator, and their indices into "orientations".
    """
    assert 0 <= root_index <= len(start2end) - 1 or root_index == -1

    if root_index == 0:
        indices = start2end[1:]
    elif 0 < root_index < len(start2end) - 1:
        indices = start2end
    elif root_index == len(start2end) - 1:
        indices = start2end[:-1]
    else:  # root_index == -1
        if parents[start2end[0]] != start2end[0]:
            indices = start2end[1:]
        else:
            indices = start2end[:-1]

    return R.from_quat(orientations[indices]), indices.copy()


def manipulator_links(joint_positions: np.ndarray,
                      start2end: List[int]) -> np.ndarray:
    """
    Compute the links in the manipulator defined by a list of joint indices, one for every consecutive pair of joints.
    @param joint_positions: n x 3 numpy array of positions of all joints.
    @param start2end: indices (relative to joint_positions) of joints in the manipulator.
    @return: m x 3 array of links.
    """
    return joint_positions[start2end][1:] - joint_positions[start2end][:-1]


def update_joint_orientations(joint_orientations, orientations):
    pass


def link_position(joint_positions: np.ndarray,
                  start2end: List[int],
                  links: np.ndarray,
                  i: int = None):
    """
    Return the position of a joint in the manipulator.
    @param joint_positions: n x 3 array of all joint positions.
    @param start2end: indices (relative to joint_positions) of joints in the manipulator.
    @param links: m x 3 numpy array of links.
    @param i: index of the joint of which we are computing the position for.
    @return: 3 array of position of i-th joint
    """
    return joint_positions[start2end[0]] + np.sum(links[:i], axis=0)


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
    orientations = link_orientations(joint_orientations, parents,
                                     start2end, root_index)  # the same orientation may occur twice, as desired.
    links = manipulator_links(positions, start2end)  # include all links connect the root, as desired.
    error, error_norm = forward(positions, start2end, links, target)

    iteration_count = 0
    while iteration_count < max_iterations and max_error < error_norm:
        loss_gradient = backward(links, error)

        links, orientations = step(loss_gradient, learning_rate, links, orientations)

        error, error_norm = forward(positions, start2end, links, target)

        iteration_count += 1

    joint_orientations = update_joint_orientations(joint_orientations, orientations, start2end, root_index)
    new_root = link_position(positions, start2end, links, root_index)

    return (pose_joint_positions(new_root, parents,
                                 joint_offsets, joint_orientations),
            joint_orientations)

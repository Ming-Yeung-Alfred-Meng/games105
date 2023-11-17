from typing import List, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R


def joint_offsets_from_positions(joint_positions: np.ndarray,
                                 joint_parents: List[int]) -> np.ndarray:
    offsets = np.empty_like(joint_positions)
    offsets[0] = np.zeros_like(offsets[0])

    for i in range(1, joint_positions.shape[0]):
        offsets[i] = joint_positions[i] - joint_positions[joint_parents[i]]

    return offsets


def update_chain_orientations(path: List[int],
                              start: int,
                              rotation,
                              joint_orientations: np.ndarray) -> None:
    """
    Update the orientations of joints from index "start" to the end effector (exclusive) in a chain of joints defined by "path".
    """
    for i in path[start:-1]:
        joint_orientations[i] = (rotation * R.from_quat(joint_orientations[i])).as_quat()


def jacobian_transpose(links: np.ndarray,
                       root_index: int):
    """
    Compute the transpose of the Jacobian of the manipulator formed by "links". Assume the links use Euler rotations
    in the same canonical coordinate system.
    @param root_index: index of the root, i.e. 0 in "start2end".
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


def step(orientations: R,
         links: np.ndarray,
         loss_gradient: np.ndarray,
         learning_rate: float = 1) -> Tuple[np.ndarray, R]:
    """
    Update links and their orientations using gradient descent.
    @param orientations: m scipy rotations defining the orientations of links.
    @param links: m x 3 array of links in the manipulator.
    @param loss_gradient: m array of gradient of loss function w.r.t. joint angles.
    @param learning_rate: amount to step in the direction of loss_gradient.
    @return: updated links and orientations.
    """
    rotations = R.from_euler("XYZ", (- learning_rate * loss_gradient).reshape(-1, 3))

    return rotations.apply(links), rotations * orientations


def forward(joint_positions: np.ndarray,
            start2end: List[int],
            links: np.ndarray,
            target: np.ndarray):
    """
    Compute the output of the loss function, i.e. displacement (error) from target to end effector, and its L2
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
                      root_index: int) -> Tuple[R, np.ndarray]:
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

    parents = np.array(parents)

    if root_index == 0:
        indices = parents[start2end[1:]]
    elif 0 < root_index < len(start2end) - 1:
        indices = parents[start2end]
    elif root_index == len(start2end) - 1:
        indices = parents[start2end[:-1]]
    else:  # root_index == -1
        if parents[start2end[0]] != start2end[0]:
            indices = parents[start2end[1:]]
        else:
            indices = parents[start2end[:-1]]

    return R.from_quat(orientations[indices]), indices


def manipulator_links(joint_positions: np.ndarray,
                      start2end: List[int]) -> np.ndarray:
    """
    Compute the links in the manipulator defined by a list of joint indices, one for every consecutive pair of joints.
    @param joint_positions: n x 3 numpy array of positions of all joints.
    @param start2end: indices (relative to joint_positions) of joints in the manipulator.
    @return: m x 3 array of links.
    """
    return joint_positions[start2end][1:] - joint_positions[start2end][:-1]


def update_joint_orientations(joint_orientations: np.ndarray,
                              orientations: np.ndarray,
                              indices: np.ndarray):
    """
    Update selected orientations.
    @param joint_orientations: n x 4 array of orientations, in which some of them are being updated.
    @param orientations: o x 4 new orientations.
    @param indices: indices into joint_orientations of orientations to be updated.
    @return: n x 4 array of updated orientations.
    """
    result = joint_orientations.copy()
    result[indices] = orientations
    return result


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

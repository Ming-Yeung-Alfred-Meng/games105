from typing import List
import numpy as np


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
        joint_orientations[i] = rotation * joint_orientations[i]

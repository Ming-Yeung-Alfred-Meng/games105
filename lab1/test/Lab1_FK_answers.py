import unittest
from lab1.Lab1_FK_answers import *
import numpy as np


class CalculateTPose(unittest.TestCase):
    def test_multiple_child_multiple_depth(self):
        bvh_path = "../data/truncated_walk60.bvh"
        (joint_names_actual,
         joint_parents_actual,
         joint_offsets_actual) = part1_calculate_T_pose(bvh_path)

        joint_names_expected = ["RootJoint",
                                "lHip", "lHip_end",
                                "rHip", "rKnee", "rKnee_end"]
        joint_parents_expected = [-1, 0, 1, 0, 3, 4]
        joint_offsets_expected = np.array([[0.000000, 0.000000, 0.000000],
                                           [0.100000, -0.051395, 0.000000],
                                           [0.010000, 0.002000, 0.060000],
                                           [-0.100000, -0.051395, 0.000000],
                                           [0.000000, -0.410000, 0.000000],
                                           [-0.010000, 0.002000, 0.060000]])

        self.assertEqual(joint_names_actual, joint_names_expected)
        self.assertEqual(joint_parents_actual, joint_parents_expected)
        self.assertTrue(np.allclose(joint_offsets_actual, joint_offsets_expected))


class PoseJointOrientation(unittest.TestCase):
    def test_small_bvh(self):
        joint_names = ["RootJoint",
                       "lHip", "lHip_end",
                       "rHip", "rKnee", "rKnee_end"]
        joint_parents = [-1, 0, 1, 0, 3, 4]

        bvh_path = "../data/truncated_walk60.bvh"
        pose = load_motion_data(bvh_path)[0]

        orientations_actual = np.stack([o.as_quat() for o in pose_joint_orientations(joint_names,
                                                                                     joint_parents,
                                                                                     pose)],
                                       axis=0,
                                       dtype=np.float64)
        orientations_expected = np.array([[0.017026, 0.06643844, -0.00578745, 0.99762847],
                                          [0.11351008, 0.03868307, -0.02932178, 0.9923504],
                                          [0.13208152, 0.0602358, -0.05127781, 0.98807728],
                                          [-0.01987982, 0.04116234, 0.03973097, 0.99816427]])

        self.assertTrue(np.allclose(orientations_actual, orientations_expected))


if __name__ == '__main__':
    unittest.main()

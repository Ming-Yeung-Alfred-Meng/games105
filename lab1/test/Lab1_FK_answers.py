import unittest
import lab1.Lab1_FK_answers as l
import numpy as np


class CalculateTPose(unittest.TestCase):
    def test_multiple_child_multiple_depth(self):
        bvh_path = "../data/truncated_walk60.bvh"
        (joint_names_actual,
         joint_parents_actual,
         joint_offsets_actual) = l.part1_calculate_T_pose(bvh_path)

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


if __name__ == '__main__':
    unittest.main()

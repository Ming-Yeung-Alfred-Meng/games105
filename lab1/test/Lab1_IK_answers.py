import unittest
from lab1.Lab1_IK_answers import *
import numpy as np


class JointOffsetsFromPositions(unittest.TestCase):

    def test_small_bvh(self):
        joint_positions = np.array([[-0.001735, 0.855388, 0.315499],
                                    [0.09666573, 0.80309773, 0.30051669],
                                    [0.11095962, 0.7908955, 0.3584027],
                                    [-0.10155524, 0.80495475, 0.32706845],
                                    [-0.14962567, 0.4114162, 0.2225857],
                                    [-0.15488649, 0.41520896, 0.28309966]])
        joint_parents = [-1, 0, 1, 0, 3, 4]

        offsets_expected = np.array([[0.000000, 0.000000, 0.000000],
                                     [ 0.09840073, -0.05229027, -0.01498231],
                                     [0.01429389, -0.01220223,  0.05788601],
                                     [-0.09982024, -0.05043325,  0.01156945],
                                     [-0.04807043, -0.39353855, -0.10448275],
                                     [-0.00526082,  0.00379276,  0.06051396]])

        offsets_actual = joint_offsets_from_positions(joint_positions,
                                                      joint_parents)

        self.assertTrue(np.allclose(offsets_actual, offsets_expected))


if __name__ == '__main__':
    unittest.main()

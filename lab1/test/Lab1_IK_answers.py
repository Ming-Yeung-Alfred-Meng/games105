import unittest
from lab1.Lab1_IK_answers import *
from lab1.Lab1_FK_answers import *
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
                                     [0.09840073, -0.05229027, -0.01498231],
                                     [0.01429389, -0.01220223, 0.05788601],
                                     [-0.09982024, -0.05043325, 0.01156945],
                                     [-0.04807043, -0.39353855, -0.10448275],
                                     [-0.00526082, 0.00379276, 0.06051396]])

        offsets_actual = joint_offsets_from_positions(joint_positions,
                                                      joint_parents)

        self.assertTrue(np.allclose(offsets_actual, offsets_expected))

    def test_reversed(self):
        root_position = np.array([-0.001735, 0.855388, 0.315499])
        joint_parents = [-1, 0, 1, 0, 3, 4]
        orientations = np.array([R.from_quat([0., 0., 0., 1.]),
                                 R.from_quat([0., 0., 0., 1.]),
                                 R.from_quat([0., 0., 0., 1.]),
                                 R.from_quat([0., 0., 0., 1.]),
                                 R.from_quat([0., 0., 0., 1.]),
                                 R.from_quat([0., 0., 0., 1.])])
        offsets = np.array([[0.000000, 0.000000, 0.000000],
                            [0.09840073, -0.05229027, -0.01498231],
                            [0.01429389, -0.01220223, 0.05788601],
                            [-0.09982024, -0.05043325, 0.01156945],
                            [-0.04807043, -0.39353855, -0.10448275],
                            [-0.00526082, 0.00379276, 0.06051396]])
        positions_expected = np.array([[-0.001735, 0.855388, 0.315499],
                                       [0.09666573, 0.80309773, 0.30051669],
                                       [0.11095962, 0.7908955, 0.3584027],
                                       [-0.10155524, 0.80495475, 0.32706845],
                                       [-0.14962567, 0.4114162, 0.2225857],
                                       [-0.15488649, 0.41520896, 0.28309966]])
        positions_actual = pose_joint_positions(root_position,
                                                joint_parents,
                                                offsets,
                                                orientations)
        self.assertTrue(np.allclose(positions_actual, positions_expected))


class UpdateChainOrientation(unittest.TestCase):

    def test_small_bvh(self):
        path = [2, 0, 1, 4, 3]
        start = 1
        rotation = R.from_quat([0.52784149, 0.00947183, 0.63056979, 0.74099276])
        joint_orientations = np.array([[0.35434237, 0.66826034, 0.05824902, 0.19663584],
                                       [0.7755293, 0.40326017, 0.02373202, 0.38239167],
                                       [0.33911014, 0.74349115, 0.97532171, 0.52664232],
                                       [0.87143483, 0.59879948, 0.60969387, 0.86172419],
                                       [0.30543437, 0.74410066, 0.00692545, 0.98328112]])
        joint_orientations_expected = np.array([[-0.06279299, 0.7950419, 0.5954014, -0.09727584],
                                                [0.4945142, 0.73729047, 0.43940367, -0.13705062],
                                                [0.33911014, 0.74349115, 0.97532171, 0.52664232],
                                                [0.87143483, 0.59879948, 0.60969387, 0.86172419],
                                                [0.19640433, 0.53305811, 0.72178616, 0.39534686]])

        update_chain_orientations(path,
                                  start,
                                  rotation,
                                  joint_orientations)

        self.assertTrue(np.allclose(joint_orientations, joint_orientations_expected))


if __name__ == '__main__':
    unittest.main()

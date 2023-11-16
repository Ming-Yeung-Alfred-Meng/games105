import unittest
from lab1.inverse_kinematics_methods.scripts import *
from lab1.task2_inverse_kinematics import *
from lab1.Lab1_FK_answers import *

root_as_start = None
root_as_intermediate_joint = None


class Character:
    def __init__(self, bvh: str, target: np.ndarray, start: str, end: str):
        self.start = start
        self.end = end
        self.target = target
        self.joint_name, self.joint_parents, self.joint_offsets = part1_calculate_T_pose(bvh)
        self.motion_data = load_motion_data(bvh)
        self.joint_positions, self.joint_orientations = part2_forward_kinematics(self.joint_name,
                                                                                 self.joint_parents,
                                                                                 self.joint_offsets,
                                                                                 self.motion_data,
                                                                                 0)
        self.meta_data = MetaData(self.joint_name, self.joint_parents, self.joint_positions, self.start, self.end)
        self.start2end, self.start2end_names, self.root_index = self.meta_data.get_path_from_root_to_end()


def setUpModule():
    global root_as_start
    global root_as_intermediate_joint

    root_as_start = Character("..\\..\\data\\simple.bvh",
                              np.array([1., 1., 1.]),
                              "RootJoint",
                              "Link1_end")
    root_as_intermediate_joint = Character("..\\..\\data\\general_simple.bvh",
                                           np.array([2., 2., 2.]),
                                           'Link2_end', 'Link1_end')


class JacobianTranspose(unittest.TestCase):
    def setUp(self):
        self.links = np.array([[-4, -10, -10],
                               [-6, 2, 1],
                               [2, -9, -5]])

    def test_root_as_start(self):
        root_index = 0

        jacobian_expected = np.array([[0, 10, -10],
                                      [-10, 0, 4],
                                      [10, -4, 0],
                                      [0, -1, 2],
                                      [1, 0, 6],
                                      [-2, -6, 0],
                                      [0, 5, -9],
                                      [-5, 0, -2],
                                      [9, 2, 0]])

        jacobian_actual = jacobian_transpose(self.links, root_index)
        self.assertTrue(np.allclose(jacobian_actual, jacobian_expected))

    def test_root_as_intermediate_joint(self):
        root_index = 1

        jacobian_expected = np.array([[0, 9, -8],
                                      [-9, 0, 10],
                                      [8, -10, 0],
                                      [0, 9, -8],
                                      [-9, 0, 10],
                                      [8, -10, 0],
                                      [0, 5, -9],
                                      [-5, 0, -2],
                                      [9, 2, 0]])

        jacobian_actual = jacobian_transpose(self.links, root_index)
        self.assertTrue(np.allclose(jacobian_actual, jacobian_expected))


class Backward(unittest.TestCase):
    def test_simple(self):
        links = np.array([[-4, -10, -10],
                          [-6, 2, 1],
                          [2, -9, -5]])
        error = np.array([7, -8, -4])
        root_index = 1
        gradient_expected = np.array([-40, -103, 136, -40, -103, 136, -4, -27, 47])

        gradient_actual = backward(links, root_index, error)
        self.assertTrue(np.allclose(gradient_actual, gradient_expected))


class Step(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(True, False)


class Forward(unittest.TestCase):
    def test_something(self):

        self.assertEqual(True, False)


class LinkOrientations(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


class ManipulatorLinks(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


class UpdateJointOrientation(unittest.TestCase):
    def test_something(self):
        joint_orientations = np.array([[5, -9, -2, 8],
                                       [9, -2, -6, -8],
                                       [-4, 9, 2, 2],
                                       [-2, 3, 0, -7],
                                       [6, 5, 2, 3]])
        new_orientations = R.from_quat(np.array([[-7, -8, 3, 6],
                                                 [0, 1, 1, -2],
                                                 [0, -8, 7, 5]]))
        indices = np.array([0, 3, 2])
        orientations_expected = np.array([])

        orientations_actual = update_joint_orientations(joint_orientations,
                                                        new_orientations,
                                                        indices)
        self.assertEqual(True, False)


class LinkPosition(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()

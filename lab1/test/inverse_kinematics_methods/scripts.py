import unittest
from lab1.task2_inverse_kinematics import *
from lab1.Lab1_FK_answers import *


class JacobianTranspose(unittest.TestCase):

    def test_root_as_start(self):
        links = np.array([[-4, -10, -10],
                          [-6, 2, 1],
                          [2, -9, -5]])
        root_index = 0

        jacobian_expected = np.array([[0, 14, -17],
                                      [-14, 0, 8],
                                      [17, -8, 0],
                                      [0, 4, -7],
                                      [-4, 0, 4],
                                      [7, -4, 0],
                                      [0, 5, -9],
                                      [-5, 0, -2],
                                      [9, 2, 0]])

        jacobian_actual = jacobian_transpose(links, root_index)
        self.assertTrue(np.allclose(jacobian_actual, jacobian_expected))

    def test_root_as_intermediate_joint(self):
        links = np.array([[49, -98, -9],
                          [99, -49, 91],
                          [-17, 15, -25]])
        root_index = 1

        jacobian_expected = np.array([[0, -57, -132],
                                      [57, 0, -131],
                                      [132, 131, 0],
                                      [0, -57, -132],
                                      [57, 0, -131],
                                      [132, 131, 0],
                                      [0, 25, 15],
                                      [-25, 0, 17],
                                      [-15, -17, 0]])

        jacobian_actual = jacobian_transpose(links, root_index)
        self.assertTrue(np.allclose(jacobian_actual, jacobian_expected))

    def test_root_as_end(self):
        links = np.array([[19, -59, 83],
                          [-40, 0, 66]])
        root_index = 2

        jacobian_expected = np.array([[0, -149, -59],
                                      [149, 0, 21],
                                      [59, -21, 0],
                                      [0, -66, 0],
                                      [66, 0, 40],
                                      [0, -40, 0]])

        jacobian_actual = jacobian_transpose(links, root_index)
        self.assertTrue(np.allclose(jacobian_actual, jacobian_expected))

    def test_root_not_in_manipulator(self):
        links = np.array([[79, 86, 17],
                          [71, 83, 75]])
        root_index = -1

        jacobian_expected = np.array([[0, -92, 169],
                                      [92, 0, -150],
                                      [-169, 150, 0],
                                      [0, -75, 83],
                                      [75, 0, -71],
                                      [-83, 71, 0]])

        jacobian_actual = jacobian_transpose(links, root_index)
        self.assertTrue(np.allclose(jacobian_actual, jacobian_expected))


class Backward(unittest.TestCase):
    def test_simple(self):
        links = np.array([[-4, -10, -10],
                          [-6, 2, 1],
                          [2, -9, -5]])
        error = np.array([7, -8, -4])
        root_index = 1
        gradient_expected = np.array([-44, -130, 183, -44, -130, 183, -4, -27, 47])

        gradient_actual = backward(links, root_index, error)
        self.assertTrue(np.allclose(gradient_actual, gradient_expected))


class Step(unittest.TestCase):
    def test_simple(self):
        orientations = R.from_quat(np.array([[83, 65, 98, 45],
                                             [-13, -41, 18, -64],
                                             [95, -94, 45, 23],
                                             [32, -23, -39, -87]]))
        links = np.array([[56, -93, -43],
                          [-83, 83, -76],
                          [93, -24, -81],
                          [-53, 88, 24]])
        gradient = np.array([34, 98, -2, 35, 35, 38, -66, -44, -50, -73, 61, 21])
        learning_rate = 0.1940226470258245

        links_expected = np.array([[92.89717194, -70.41588183, -6.76158501],
                                   [50.20442353, 61.49383734, -115.11743494],
                                   [-104.03731298, 20.69907949, 67.33339154],
                                   [-75.71978084, 68.75894965, -25.84031023]])
        orientations_expected = np.array([[0.36047663, 0.62122638, 0.64125761, 0.27004272],
                                          [0.23974381, 0.01832774, -0.34639601, 0.90675068],
                                          [-0.54044477, -0.34434155, 0.11950791, 0.7583312],
                                          [-0.57084761, 0.24832479, 0.40450021, 0.66996073]])

        links_actual, orientations_actual = step(orientations, links, gradient, learning_rate)
        self.assertTrue(np.allclose(links_actual, links_expected))
        self.assertTrue(np.allclose(orientations_actual.as_quat(), orientations_expected))


class Forward(unittest.TestCase):
    def test_simple(self):
        positions = np.array([[72, -35, 25],
                              [19, -85, -80],
                              [-93, -8, 3],
                              [39, 62, 96],
                              [69, -55, 99]])
        start2end = [3, 4, 2, 1]
        links = np.array([[30, -117, 3], [-162, 47, -96], [112, -77, -83]])
        target = np.array([12, 57, 27])
        error_expected = np.array([7, -142, -107])
        error_norm_expected = 177.93819151604302

        error_actual, error_norm_actual = forward(positions, start2end, links, target)
        self.assertTrue(np.allclose(error_actual, error_expected))
        self.assertTrue(np.allclose(error_norm_actual, error_norm_expected))


class LinkOrientations(unittest.TestCase):

    def setUp(self):
        # 0, 1, 2, 3, 4, 5
        self.joint_orientations = np.array([[64, -73, -25, 44],
                                            [84, 52, 5, -70],
                                            [-11, 59, 95, 43],
                                            [21, 22, -22, -43],
                                            [87, 23, 98, -43],
                                            [91, 51, 48, -94]])
        self.parents = [-1, 0, 1, 0, 3, 4]

    def test_root_as_start(self):
        start2end = [0, 3, 4, 5]
        root_index = 0

        orientations_expected = np.array([[64, -73, -25, 44],
                                          [21, 22, -22, -43],
                                          [87, 23, 98, -43]])
        indices_expected = np.array([0, 3, 4])

        orientations_actual, indices_actual = link_orientations(self.joint_orientations, self.parents,
                                                                start2end, root_index)
        self.assertTrue(np.allclose(orientations_actual, orientations_expected))
        self.assertTrue(np.allclose(indices_actual, indices_expected))

    def test_root_as_intermediate_joint(self):
        start2end = [2, 1, 0, 3, 4, 5]
        root_index = 2
        # 0, 1, 2, 3, 4, 5
        orientations_expected = np.array([[84, 52, 5, -70],
                                          [64, -73, -25, 44],
                                          [64, -73, -25, 44],
                                          [21, 22, -22, -43],
                                          [87, 23, 98, -43]])
        indices_expected = np.array([1, 0, 0, 3, 4])

        orientations_actual, indices_actual = link_orientations(self.joint_orientations, self.parents,
                                                                start2end, root_index)
        self.assertTrue(np.allclose(orientations_actual, orientations_expected))
        self.assertTrue(np.allclose(indices_actual, indices_expected))

    def test_root_as_intermediate_joint_left_edge(self):
        start2end = [1, 0, 3, 4, 5]
        root_index = 1
        # 0, 1, 2, 3, 4, 5
        orientations_expected = np.array([[64, -73, -25, 44],
                                          [64, -73, -25, 44],
                                          [21, 22, -22, -43],
                                          [87, 23, 98, -43]])
        indices_expected = np.array([0, 0, 3, 4])

        orientations_actual, indices_actual = link_orientations(self.joint_orientations, self.parents,
                                                                start2end, root_index)
        self.assertTrue(np.allclose(orientations_actual, orientations_expected))
        self.assertTrue(np.allclose(indices_actual, indices_expected))

    def test_root_as_intermediate_joint_right_edge(self):
        start2end = [2, 1, 0, 3]
        root_index = 2
        # 0, 1, 2, 3, 4, 5
        orientations_expected = np.array([[84, 52, 5, -70],
                                          [64, -73, -25, 44],
                                          [64, -73, -25, 44]])
        indices_expected = np.array([1, 0, 0])

        orientations_actual, indices_actual = link_orientations(self.joint_orientations, self.parents,
                                                                start2end, root_index)
        self.assertTrue(np.allclose(orientations_actual, orientations_expected))
        self.assertTrue(np.allclose(indices_actual, indices_expected))

    def test_root_as_end(self):
        start2end = [5, 4, 3, 0]
        root_index = 3
        # 0, 1, 2, 3, 4, 5
        orientations_expected = np.array([[87, 23, 98, -43],
                                          [21, 22, -22, -43],
                                          [64, -73, -25, 44]])
        indices_expected = np.array([4, 3, 0])

        orientations_actual, indices_actual = link_orientations(self.joint_orientations, self.parents,
                                                                start2end, root_index)
        self.assertTrue(np.allclose(orientations_actual, orientations_expected))
        self.assertTrue(np.allclose(indices_actual, indices_expected))

    def test_root_not_in_manipulator_left_to_right(self):
        start2end = [2, 1]
        root_index = -1
        # 0, 1, 2, 3, 4, 5
        orientations_expected = np.array([[84, 52, 5, -70]])
        indices_expected = np.array([1])

        orientations_actual, indices_actual = link_orientations(self.joint_orientations, self.parents,
                                                                start2end, root_index)
        self.assertTrue(np.allclose(orientations_actual, orientations_expected))
        self.assertTrue(np.allclose(indices_actual, indices_expected))

    def test_root_not_in_manipulator_right_to_left(self):
        start2end = [3, 4, 5]
        root_index = -1
        # 0, 1, 2, 3, 4, 5
        orientations_expected = np.array([[21, 22, -22, -43],
                                          [87, 23, 98, -43]])
        indices_expected = np.array([3, 4])

        orientations_actual, indices_actual = link_orientations(self.joint_orientations, self.parents,
                                                                start2end, root_index)
        self.assertTrue(np.allclose(orientations_actual, orientations_expected))
        self.assertTrue(np.allclose(indices_actual, indices_expected))


class ManipulatorLinks(unittest.TestCase):
    def test_simple(self):
        positions = np.array([[-56, -32, 98],
                              [18, -59, -30],
                              [29, -53, 46],
                              [-91, 69, 55],
                              [31, -38, -89]])
        start2end = [3, 4, 1]

        links_expected = np.array([[122, -107, -144],
                                   [-13, -21, 59]])
        links_actual = manipulator_links(positions, start2end)
        self.assertTrue(np.allclose(links_actual, links_expected))


class UpdateJointOrientation(unittest.TestCase):
    def test_simple(self):
        joint_orientations = np.array([[5, -9, -2, 8],
                                       [9, -2, -6, -8],
                                       [-4, 9, 2, 2],
                                       [-2, 3, 0, -7],
                                       [6, 5, 2, 3]])
        new_orientations = np.array([[-7, -8, 3, 6],
                                     [0, 1, 1, -2],
                                     [0, -8, 7, 5]])
        indices = np.array([0, 3, 2])
        orientations_expected = np.array([[-7, -8, 3, 6],
                                          [9, -2, -6, -8],
                                          [0, -8, 7, 5],
                                          [0, 1, 1, -2],
                                          [6, 5, 2, 3]])

        orientations_actual = update_joint_orientations(joint_orientations,
                                                        new_orientations,
                                                        indices)
        self.assertTrue(np.allclose(orientations_actual, orientations_expected))


class LinkPosition(unittest.TestCase):

    def setUp(self):
        self.positions = np.array([[92, -51, -31],
                                   [-3, 64, -50],
                                   [-38, 2, -11],
                                   [90, -14, 95],
                                   [85, 60, 14]])
        self.start2end = [2, 4, 3, 1]
        self.links = np.array([[123, 58, 25],
                               [5, -74, 81],
                               [-93, 78, -145]])

    def test_first_joint(self):
        i = 0
        position_expected = np.array([-38, 2, -11])

        position_actual = link_position(self.positions, self.start2end, self.links, i)
        self.assertTrue(np.allclose(position_actual, position_expected))

    def test_intermediate_joint(self):
        i = 1
        position_expected = np.array([85, 60, 14])

        position_actual = link_position(self.positions, self.start2end, self.links, i)
        self.assertTrue(np.allclose(position_actual, position_expected))

    def test_last_joint_use_index(self):
        i = 3
        position_expected = np.array([-3, 64, -50])

        position_actual = link_position(self.positions, self.start2end, self.links, i)
        self.assertTrue(np.allclose(position_actual, position_expected))

    def test_last_joint_no_index(self):
        position_expected = np.array([-3, 64, -50])

        position_actual = link_position(self.positions, self.start2end, self.links)
        self.assertTrue(np.allclose(position_actual, position_expected))


class UpdateRoot(unittest.TestCase):

    def setUp(self):
        self.positions = np.array([[-71, -33, 37],
                                   [64, -85, 70],
                                   [15, 0, -4],
                                   [55, 54, -66],
                                   [-36, -72, -16]])

    def test_root_as_start(self):
        start2end = [0, 4, 2, 3]
        links = np.array([[-92, 78, 35],
                          [89, 18, 13],
                          [60, -52, 42]])
        # do NOT compute links from self.positions, as links are in general not offsets.
        root_index = 0

        root_expected = np.array([-71, -33, 37])

        root_actual = update_root(self.positions, start2end, links, root_index)
        self.assertTrue(np.allclose(root_actual, root_expected))

    def test_root_as_end(self):
        start2end = [3, 4, 2, 0]
        links = np.array([[90, 30, 15],
                          [31, -75, 35],
                          [-4, 57, 44]])
        root_index = 3

        root_expected = np.array([-71, -33, 37])

        root_actual = update_root(self.positions, start2end, links, root_index)
        self.assertTrue(np.allclose(root_actual, root_expected))

    def test_root_as_intermediate_joint(self):
        start2end = [1, 3, 0, 4]
        links = np.array([[38, -43, 17],
                          [10, 18, 73],
                          [8, -59, -95]])
        root_index = 2

        root_expected = np.array([112, -110, 160])

        root_actual = update_root(self.positions, start2end, links, root_index)
        self.assertTrue(np.allclose(root_actual, root_expected))

    def test_root_not_in_manipulator(self):
        start2end = [4, 3, 2, 1]
        links = np.array([[-4, -99, 99],
                          [78, -52, 89],
                          [19, -84, 6]])
        root_index = -1

        root_expected = np.array([-71, -33, 37])

        root_actual = update_root(self.positions, start2end, links, root_index)
        self.assertTrue(np.allclose(root_actual, root_expected))


if __name__ == '__main__':
    unittest.main()

from typing import Dict
import numpy as np

from pydrake.all import ModelInstanceIndex
from qsim.simulator import QuasistaticSimulator


class ConfigurationSpace:
    def __init__(self, joint_limits: Dict[ModelInstanceIndex, np.ndarray],
                 q_sim: QuasistaticSimulator):
        """
        For each model instance with n DOFs,
        joint_limits[model] is an (n, 2) array, where joint_limits[model][i, 0] is the
        lower bound of joint i and joint_limits[model][i, 1] the upper bound.
        """
        self.joint_limits = joint_limits
        self.q_sim = q_sim

    def sample(self):
        """
        returns a collision-free configuration for self.qsim.plant.
        """
        q_dict = {}
        while True:
            for model, bounds in self.joint_limits.items():
                n = len(bounds)
                lb = bounds[:, 0]
                ub = bounds[:, 1]
                q_model = np.random.rand(n) * (ub - lb) + lb
                q_dict[model] = q_model

            if not self.has_collision(q_dict):
                break
        return q_dict

    def dist(self, q1: Dict[ModelInstanceIndex, np.ndarray],
             q2: Dict[ModelInstanceIndex, np.ndarray]):
        d = 0
        for model in q1.keys():
            dq = q1[model] - q2[model]
            d += np.sqrt((dq**2).sum())
        return d

    def has_collision(self, q_dict: Dict[ModelInstanceIndex, np.ndarray]):
        self.q_sim.update_mbp_positions(q_dict)  # this also updates query_object.
        return self.q_sim.query_object.HasCollisions()


class TreeNode:
    def __init__(self, q, parent):
        self.q = q
        self.parent = parent
        self.children = []


class RRT:
    class RRT:
        """
        RRT Tree.
        """
    def __init__(self, root: TreeNode, cspace: ConfigurationSpace):
        self.root = root  # root TreeNode
        self.cspace = cspace  # robot.ConfigurationSpace
        self.size = 1  # int length of path
        self.max_recursion = 1000  # int length of longest possible path

    def add_node(self, parent_node: TreeNode,
                 q_child: Dict[ModelInstanceIndex, np.ndarray]):
        child_node = TreeNode(q_child, parent_node)
        parent_node.children.append(child_node)
        self.size += 1
        return child_node

    # Brute force nearest, handles general distance functions
    def nearest(self, q_target: Dict[ModelInstanceIndex, np.ndarray]):
        """
        Finds the nearest node in the tree by distance to q_target in the
             configuration space.
        Args:
            q_target: dictionary of arrays representing a configuration of the MBP.
        Returns:
            closest: TreeNode. the closest node in the configuration space
                to q_target
            distance: float. distance from q_target to closest.q
        """

        def recur(node, depth=0):
            closest, distance = node, self.cspace.dist(node.q, q_target)
            if depth < self.max_recursion:
                for child in node.children:
                    (child_closest, child_distance) = recur(child, depth+1)
                    if child_distance < distance:
                        closest = child_closest
                        child_distance = child_distance
            return closest, distance
        return recur(self.root)[0]


from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional
from numpy.typing import NDArray
import numpy as np
from .base import BaseOptimisation


class SOO(BaseOptimisation):
    """Simultaneous Optimistic Optimization (SOO) algorithm."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def single_rollout(
        self,
        X: NDArray,
        x_current: NDArray,
        rollout_round: int,
        method_args: dict = {},
    ) -> NDArray:
        """Perform a single rollout of the SOO algorithm."""
        domain = np.array([[self.f.lb[0]] * self.f.dims, [self.f.ub[0]] * self.f.dims])
        soo_tree = BinarySOOTree(domain)

        for _ in range(rollout_round):
            next_node = soo_tree.get_next_point_and_node_to_evaluate()
            x_to_evaluate = next_node.cell_mid_point
            next_node.evaluated_x = x_to_evaluate
            x_to_evaluate = np.round(x_to_evaluate, int(-np.log10(self.f.turn)))
            fval = self.model.predict(
                np.array(x_to_evaluate).reshape(1, self.f.dims, 1), verbose=False
            )
            fval = np.array(fval).reshape(1)
            soo_tree.expand_node(fval, next_node)
            self.all_proposed.append(x_to_evaluate)

        return self.get_top_X(X, self.num_samples_per_acquisition)


@dataclass
class SOOTreeNode:
    """Node in the SOO tree."""

    cell_mid_point: NDArray
    cell_min: NDArray
    cell_max: NDArray
    height: int
    parent: Optional[SOOTreeNode] = None
    evaluated_x: Optional[NDArray] = None
    l_child: Optional[SOOTreeNode] = None
    r_child: Optional[SOOTreeNode] = None
    delta_h: float = 0.0
    f_value: Optional[float] = None

    def update_node_f_value(self, fval: float) -> None:
        """Update the node's f-value."""
        self.f_value = fval


@dataclass
class BinarySOOTree:
    """Binary tree for the SOO algorithm."""

    domain: NDArray
    root: Optional[SOOTreeNode] = None
    leaves: list[SOOTreeNode] = field(default_factory=list)
    nodes: list[SOOTreeNode] = field(default_factory=list)
    x_to_node: dict = field(default_factory=dict)
    vmax: float = -np.inf
    tree_traversal_height: int = 0
    tree_height: int = 0

    @staticmethod
    def create_node(
        cell_mid_point: NDArray,
        cell_min: NDArray,
        cell_max: NDArray,
        parent_node: Optional[SOOTreeNode],
    ) -> SOOTreeNode:
        """Create a new SOOTreeNode."""
        height = 0 if parent_node is None else parent_node.height + 1
        return SOOTreeNode(cell_mid_point, cell_min, cell_max, height, parent_node)

    def add_left_child(self, parent_node: SOOTreeNode) -> None:
        """Add a left child to the given parent node."""
        left_child_cell_mid_point_x = self.compute_left_child_cell_mid_point(
            parent_node
        )
        cell_min, cell_max = self.compute_left_child_cell_limits(parent_node)

        node = self.create_node(
            left_child_cell_mid_point_x, cell_min, cell_max, parent_node
        )
        self.add_node_to_tree(node, parent_node, "left")

    def add_right_child(self, parent_node: SOOTreeNode) -> None:
        """Add a right child to the given parent node."""
        right_child_cell_mid_point_x = self.compute_right_child_cell_mid_point(
            parent_node
        )
        cell_min, cell_max = self.compute_right_child_cell_limits(parent_node)

        node = self.create_node(
            right_child_cell_mid_point_x, cell_min, cell_max, parent_node
        )
        self.add_node_to_tree(node, parent_node, "right")

    def find_leaf_with_max_value_at_given_height(
        self, height: int
    ) -> Optional[SOOTreeNode]:
        """Find the leaf node with the maximum value at the given height."""
        leaves = self.get_leaves_at_height(height)
        if not leaves:
            return None

        best_leaf = max(leaves, key=lambda l: l.f_value)

        if best_leaf.f_value >= self.vmax:
            self.vmax = best_leaf.f_value
            if best_leaf.l_child is not None:
                if best_leaf.l_child.f_value is None:
                    return best_leaf.l_child
                elif best_leaf.r_child.f_value is None:
                    return best_leaf.r_child
                else:
                    raise AssertionError(
                        "When both children have been evaluated, the node should not be in self.leaves"
                    )
            else:
                return best_leaf
        return None

    def get_leaves_at_height(self, height: int) -> list[SOOTreeNode]:
        """Get all leaf nodes at the given height."""
        return [l for l in self.leaves if l.height == height]

    def get_next_point_and_node_to_evaluate(self) -> SOOTreeNode:
        """Get the next point and node to evaluate."""
        if self.root is None:
            cell_mid_point = np.random.uniform(
                self.domain[0], self.domain[1], (1, len(self.domain[0]))
            ).squeeze()
            node = self.create_node(
                cell_mid_point, self.domain[0], self.domain[1], None
            )
            self.leaves.append(node)
            self.root = node
        else:
            node = self.find_leaf_node_whose_value_is_greater_than_vmax()

        return node

    def find_leaf_node_whose_value_is_greater_than_vmax(self) -> SOOTreeNode:
        """Find a leaf node whose value is greater than vmax."""
        while self.tree_traversal_height <= self.tree_height:
            node = self.find_leaf_with_max_value_at_given_height(
                self.tree_traversal_height
            )
            if node is not None:
                return node
            self.tree_traversal_height += 1

        self.vmax = -np.inf
        self.tree_traversal_height = 0
        return self.find_leaf_node_whose_value_is_greater_than_vmax()

    def expand_node(self, fval: float, node: SOOTreeNode) -> None:
        """Expand the node if it has not been evaluated, but evaluates its children if they have not been evaluated."""
        node.update_node_f_value(fval)
        self.nodes.append(node)

        self.add_left_child(parent_node=node)
        self.add_right_child(parent_node=node)

        not_root_node = node.parent is not None
        if not_root_node:
            self.add_parent_children_to_leaves(node)

    def add_parent_children_to_leaves(self, node: SOOTreeNode) -> None:
        is_parent_node_children_all_evaluated = (
            node.parent.l_child.f_value is not None
            and node.parent.r_child.f_value is not None
        )
        if is_parent_node_children_all_evaluated:
            # note that parent is not a leaf until its children have been evaluated
            self.add_to_leaf(node.parent.l_child)
            self.add_to_leaf(node.parent.r_child)
            self.tree_traversal_height += 1  # increment the current height only when we evaluated the current node fully
            self.tree_height += 1

    def add_to_leaf(self, node: SOOTreeNode) -> None:
        parent_node = node.parent
        self.leaves.append(node)
        self.leaves = [leaf for leaf in self.leaves if leaf is not parent_node]

    def find_evaled_f_value(
        self, target_x_value: NDArray, evaled_x: NDArray, evaled_y: NDArray
    ) -> float:
        # it all gets stuck here most of the time.
        # This is likely because there are so many self.nodes and that there are so many evaled_x
        # create a mapping between the node to the evaled_x value
        is_in_array = [np.all(np.isclose(target_x_value, a)) for a in evaled_x]
        is_action_included = np.any(is_in_array)
        assert (
            is_action_included
        ), "action that needs to be updated does not have a value"
        return evaled_y[np.where(is_in_array)[0][0]]

    @staticmethod
    def add_node_to_tree(
        node: SOOTreeNode, parent_node: SOOTreeNode, side: str
    ) -> None:
        node.parent = parent_node
        if side == "left":
            parent_node.l_child = node
        else:
            parent_node.r_child = node

    @staticmethod
    def compute_left_child_cell_mid_point(node: SOOTreeNode) -> NDArray:
        cell_mid_point = copy.deepcopy(node.cell_mid_point)
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        cell_mid_point[cutting_dimension] = (
            node.cell_min[cutting_dimension] + node.cell_mid_point[cutting_dimension]
        ) / 2.0
        return cell_mid_point

    @staticmethod
    def compute_right_child_cell_mid_point(node: SOOTreeNode) -> NDArray:
        cell_mid_point = copy.deepcopy(node.cell_mid_point)
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        cell_mid_point[cutting_dimension] = (
            node.cell_max[cutting_dimension] + node.cell_mid_point[cutting_dimension]
        ) / 2.0

        return cell_mid_point

    @staticmethod
    def compute_left_child_cell_limits(
        node: SOOTreeNode,
    ) -> tuple[NDArray, NDArray]:
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        cell_min = copy.deepcopy(node.cell_min)
        cell_max = copy.deepcopy(node.cell_max)
        cell_max[cutting_dimension] = node.cell_mid_point[cutting_dimension]
        return cell_min, cell_max

    @staticmethod
    def compute_right_child_cell_limits(
        node: SOOTreeNode,
    ) -> tuple[NDArray, NDArray]:
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        cell_max = copy.deepcopy(node.cell_max)
        cell_min = copy.deepcopy(node.cell_min)
        cell_min[cutting_dimension] = node.cell_mid_point[cutting_dimension]
        return cell_min, cell_max

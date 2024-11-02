import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Self
from numpy.typing import NDArray
import numpy as np

from .base import BaseOptimisation


class DOO(BaseOptimisation):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def single_rollout(
        self,
        X: NDArray,
        x_current: NDArray,
        rollout_round: int,
        method_args: dict[str, Any] = None,
    ) -> NDArray:
        """Perform a single rollout of the DOO algorithm."""
        if not method_args:
            method_args = {"explr_p": 0.01}

        domain = np.array([[self.f.lb[0]] * self.f.dims, [self.f.ub[0]] * self.f.dims])
        distance_fn = lambda x, y: np.linalg.norm(x - y)
        doo_tree = BinaryDOOTree(domain=domain, distance_fn=distance_fn, **method_args)

        for _ in range(rollout_round):
            next_node = doo_tree.get_next_point_and_node_to_evaluate()
            x_to_evaluate = np.round(
                next_node.cell_mid_point, int(-np.log10(self.f.turn))
            )
            next_node.evaluated_x = x_to_evaluate

            fval = self.model.predict(
                x_to_evaluate.reshape(1, self.f.dims, 1), verbose=False
            ).flatten()
            doo_tree.expand_node(fval, next_node)
            self.all_proposed.append(x_to_evaluate)

        return self.get_top_X(X, self.num_samples_per_acquisition)


@dataclass
class DOOTreeNode:
    """Represents a node in the DOO tree."""

    cell_mid_point: NDArray
    cell_min: NDArray
    cell_max: NDArray
    parent_node: Self | None
    distance_fn: Callable[[NDArray, NDArray], float]
    idx: int
    evaluated_x: NDArray | None = None
    l_child: Self | None = None
    r_child: Self | None = None
    f_value: float | None = None
    delta_h: float = field(init=False)

    def __post_init__(self):
        self.delta_h = self.distance_fn(
            self.cell_mid_point, self.cell_min
        ) + self.distance_fn(self.cell_mid_point, self.cell_max)

    def update_node_f_value(self, fval: float) -> None:
        """Update the f-value of the node."""
        self.f_value = fval


@dataclass
class BinaryDOOTree:
    """Represents a binary DOO tree."""

    domain: NDArray
    distance_fn: Callable[[NDArray, NDArray], float]
    explr_p: float
    root: DOOTreeNode | None = None
    leaves: list[DOOTreeNode] = field(default_factory=list)
    nodes: list[DOOTreeNode] = field(default_factory=list)
    node_to_update: DOOTreeNode | None = None
    evaled_x_to_node: dict[tuple, DOOTreeNode] = field(default_factory=dict)

    def create_node(
        self,
        cell_mid_point: NDArray,
        cell_min: NDArray,
        cell_max: NDArray,
        parent_node: Self | None,
    ) -> DOOTreeNode:
        """Create a new DOOTreeNode."""
        new_node = DOOTreeNode(
            cell_mid_point,
            cell_min,
            cell_max,
            parent_node,
            self.distance_fn,
            idx=len(self.nodes),
        )
        return new_node

    def add_left_child(self, parent_node: DOOTreeNode) -> None:
        """Add a left child to the given parent node."""
        left_child_cell_mid_point_x = self.compute_left_child_cell_mid_point(
            parent_node
        )
        cell_min, cell_max = self.compute_left_child_cell_limits(parent_node)

        node = self.create_node(
            left_child_cell_mid_point_x, cell_min, cell_max, parent_node
        )
        self.add_node_to_tree(node, parent_node, "left")

    def add_right_child(self, parent_node: DOOTreeNode) -> None:
        """Add a right child to the given parent node."""
        right_child_cell_mid_point_x = self.compute_right_child_cell_mid_point(
            parent_node
        )
        cell_min, cell_max = self.compute_right_child_cell_limits(parent_node)

        node = self.create_node(
            right_child_cell_mid_point_x, cell_min, cell_max, parent_node
        )
        self.add_node_to_tree(node, parent_node, "right")

    def find_leaf_with_max_upper_bound_value(self) -> DOOTreeNode:
        """Find the leaf node with the maximum upper bound value."""
        max_upper_bound = -np.inf
        best_leaf = None
        for leaf_node in self.leaves:
            if leaf_node.f_value is None:
                return leaf_node
            if leaf_node.f_value == "update_me":
                continue
            node_upper_bound = leaf_node.f_value + self.explr_p * leaf_node.delta_h
            if node_upper_bound > max_upper_bound:
                best_leaf = leaf_node
                max_upper_bound = node_upper_bound

        assert best_leaf is not None, "No valid leaf found"

        if best_leaf.l_child is not None:
            if best_leaf.l_child.f_value is None:
                return best_leaf.l_child
            elif best_leaf.r_child.f_value is None:
                return best_leaf.r_child
            else:
                raise ValueError(
                    "When both children have been evaluated, the node should not be in self.leaves"
                )
        return best_leaf

    def get_next_point_and_node_to_evaluate(self) -> DOOTreeNode:
        """Get the next point and node to evaluate."""
        if self.root is None:
            dim_domain = len(self.domain[0])
            cell_mid_point = np.random.uniform(
                self.domain[0], self.domain[1], (1, dim_domain)
            ).squeeze()
            node = self.create_node(
                cell_mid_point, self.domain[0], self.domain[1], None
            )
            self.leaves.append(node)
            self.root = node
        else:
            node = self.find_leaf_with_max_upper_bound_value()
        return node

    def update_evaled_x_to_node(self, x: NDArray, node: DOOTreeNode) -> None:
        """Update the mapping of evaluated x to node."""
        self.evaled_x_to_node[tuple(x)] = node

    def expand_node(self, fval: float, node: DOOTreeNode) -> None:
        """Expand the given node with the provided f-value."""
        if fval == "update_me":
            self.node_to_update = node
        else:
            self.node_to_update = None

        node.update_node_f_value(fval)
        self.nodes.append(node)

        self.add_left_child(node)
        self.add_right_child(node)

        if node.parent_node is not None:
            is_parent_node_children_all_evaluated = (
                node.parent_node.l_child.f_value is not None
                and node.parent_node.r_child.f_value is not None
            )
            if is_parent_node_children_all_evaluated:
                self.add_to_leaf(node.parent_node.l_child)
                self.add_to_leaf(node.parent_node.r_child)

    def add_to_leaf(self, node: DOOTreeNode) -> None:
        """Add the given node to the leaves list."""
        parent_node = node.parent_node
        self.leaves.append(node)
        self.leaves = [leaf for leaf in self.leaves if leaf is not parent_node]

    def update_evaled_values(
        self,
        evaled_x: NDArray,
        evaled_y: NDArray,
        infeasible_reward: float,
        idx_to_update: NDArray,
    ) -> None:
        """Update the evaluated values in the tree."""
        if len(evaled_x) == 0:
            return

        feasible_idxs = np.zeros(len(evaled_x), dtype=bool)
        feasible_idxs[idx_to_update] = True

        evaled_x_to_update = np.array(evaled_x)[feasible_idxs]
        evaled_y_to_update = np.array(evaled_y)[feasible_idxs]
        for x, y in zip(evaled_x_to_update, evaled_y_to_update):
            node_to_update = self.evaled_x_to_node[tuple(x)]
            node_to_update.f_value = y

        fvals_in_tree = np.array([n.f_value for n in self.nodes])
        sorted_evaled_y = np.array(evaled_y)
        assert np.array_equal(
            np.sort(fvals_in_tree), np.sort(sorted_evaled_y)
        ), "Are you using N_r?"

    @staticmethod
    def add_node_to_tree(
        node: DOOTreeNode, parent_node: DOOTreeNode, side: str
    ) -> None:
        """Add a node to the tree as a child of the parent node."""
        node.parent = parent_node
        if side == "left":
            parent_node.l_child = node
        else:
            parent_node.r_child = node

    @staticmethod
    def compute_left_child_cell_mid_point(node: DOOTreeNode) -> NDArray:
        """Compute the mid-point of the left child cell."""
        cell_mid_point = copy.deepcopy(node.cell_mid_point)
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        cell_mid_point[cutting_dimension] = (
            node.cell_min[cutting_dimension] + node.cell_mid_point[cutting_dimension]
        ) / 2.0
        return cell_mid_point

    @staticmethod
    def compute_right_child_cell_mid_point(node: DOOTreeNode) -> NDArray:
        """Compute the mid-point of the right child cell."""
        cell_mid_point = copy.deepcopy(node.cell_mid_point)
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        cell_mid_point[cutting_dimension] = (
            node.cell_max[cutting_dimension] + node.cell_mid_point[cutting_dimension]
        ) / 2.0
        return cell_mid_point

    @staticmethod
    def compute_left_child_cell_limits(
        node: DOOTreeNode,
    ) -> tuple[NDArray, NDArray]:
        """Compute the limits of the left child cell."""
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        cell_min = copy.deepcopy(node.cell_min)
        cell_max = copy.deepcopy(node.cell_max)
        cell_max[cutting_dimension] = node.cell_mid_point[cutting_dimension]
        return cell_min, cell_max

    @staticmethod
    def compute_right_child_cell_limits(
        node: DOOTreeNode,
    ) -> tuple[NDArray, NDArray]:
        """Compute the limits of the right child cell."""
        cutting_dimension = np.argmax(node.cell_max - node.cell_min)
        cell_max = copy.deepcopy(node.cell_max)
        cell_min = copy.deepcopy(node.cell_min)
        cell_min[cutting_dimension] = node.cell_mid_point[cutting_dimension]
        return cell_min, cell_max

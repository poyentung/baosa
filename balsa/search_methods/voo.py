from dataclasses import dataclass
from typing import Callable, List, Optional
from numpy.typing import NDArray
import numpy as np

from .base import BaseOptimisation


@dataclass
class VOO(BaseOptimisation):
    """Voronoi Optimistic Optimisation algorithm."""

    def single_rollout(
        self,
        X: NDArray,
        Y: NDArray,
        x_current: NDArray,
        rollout_round: int,
        method_args: dict = {
                "explr_p": 0.001,
                "sampling_mode": "centered_uniform",
                "switch_counter": 100,
            }
    ) -> NDArray:
        """Perform a single rollout of the VOO algorithm."""
        domain = np.array([[self.f.lb[0]] * self.f.dims, [self.f.ub[0]] * self.f.dims])
        voo = BaseVOO(domain, **method_args)
        evaled_x = X.tolist()
        evaled_y = Y.tolist()

        for _ in range(rollout_round):
            x = voo.choose_next_point(evaled_x, evaled_y)
            x = np.round(x, int(-np.log10(self.f.turn)))
            y = self.model.predict(np.array(x).reshape(1, self.f.dims, 1), verbose=False)
            y = np.array(y).item()
            evaled_x.append(x)
            evaled_y.append(y)
            self.all_proposed.append(x)

        return self.get_top_X(X, self.num_samples_per_acquisition)

    def rollout(
        self, X: NDArray, y: NDArray, rollout_round: int, method_args: dict, num_top_samples:int=20
    ) -> NDArray:
        """Perform rollout based on the function type."""
        if self.name in ["rastrigin", "ackley"]:
            index_max = np.argmax(y)
            initial_X = X[index_max, :]
            top_X = self.single_rollout(X, y, initial_X, rollout_round)
        else:
            ind = np.argsort(y)
            x_current_top = X[ind[-3:]]
            x_current_top = np.unique(x_current_top, axis=0)
            i = 0
            while len(x_current_top) < 3:
                x_current_top = np.concatenate(
                    (
                        x_current_top.reshape(-1, self.f.dims),
                        X[ind[i - 4]].reshape(-1, self.f.dims),
                    ),
                    axis=0,
                )
                i -= 1
                x_current_top = np.unique(x_current_top, axis=0)

            X_top = []
            for i in range(3):
                initial_X = x_current_top[i]
                top_X = self.single_rollout(
                    X,
                    y,
                    initial_X,
                    rollout_round,
                    top_n=6,
                    top_n2=1,
                    method_args=method_args,
                )
                X_top.append(top_X)

            top_X = np.vstack(X_top)
            top_X = top_X[-num_top_samples:]
        return top_X


@dataclass
class BaseVOO:
    """Base class for Voronoi Optimistic Optimisation."""

    domain: NDArray
    explr_p: float
    sampling_mode: str
    switch_counter: int
    distance_fn: Optional[Callable] = None

    def __post_init__(self):
        self.dim_x = self.domain.shape[-1]
        if self.distance_fn is None:
            self.distance_fn = lambda x, y: np.linalg.norm(x - y)

        self.GAUSSIAN = False
        self.CENTERED_UNIFORM = False
        self.UNIFORM = False

        if self.sampling_mode == "centered_uniform":
            self.CENTERED_UNIFORM = True
        elif self.sampling_mode == "gaussian":
            self.GAUSSIAN = True
        elif "hybrid" in self.sampling_mode or "uniform" in self.sampling_mode:
            self.UNIFORM = True
        else:
            raise NotImplementedError(f"Unknown sampling mode: {self.sampling_mode}")

        self.UNIFORM_TOUCHING_BOUNDARY = False

    def sample_next_point(
        self, evaled_x: List[NDArray], evaled_y: List[float]
    ) -> NDArray:
        """Sample the next point based on exploration probability."""
        is_sample_from_best_v_region = (np.random.random() < 1 - self.explr_p) and len(
            evaled_x
        ) > 1
        if is_sample_from_best_v_region:
            return self.sample_from_best_voronoi_region(evaled_x, evaled_y)
        return self.sample_from_uniform()

    def choose_next_point(
        self, evaled_x: List[NDArray], evaled_y: List[float]
    ) -> NDArray:
        """Choose the next point to evaluate."""
        return self.sample_next_point(evaled_x, evaled_y)

    def sample_from_best_voronoi_region(
        self, evaled_x: List[NDArray], evaled_y: List[float]
    ) -> NDArray:
        """Sample a point from the best Voronoi region."""
        best_dist = np.inf
        other_dists = np.array([-1])
        counter = 1

        best_evaled_x_idxs = np.argwhere(evaled_y == np.amax(evaled_y)).flatten()
        best_evaled_x_idx = best_evaled_x_idxs[0]
        best_evaled_x = evaled_x[best_evaled_x_idx]
        other_best_evaled_xs = evaled_x

        curr_closest_dist = np.inf
        curr_closest_pt = None

        while np.any(best_dist > other_dists):
            if self.GAUSSIAN:
                new_x = self._sample_gaussian(best_evaled_x, counter)
            elif self.CENTERED_UNIFORM:
                new_x = self._sample_centered_uniform(best_evaled_x, counter)
            elif self.UNIFORM:
                new_x = self._sample_uniform(counter)
            else:
                raise NotImplementedError("Unknown sampling mode")

            best_dist = self.distance_fn(new_x, best_evaled_x)
            other_dists = np.array(
                [self.distance_fn(other, new_x) for other in other_best_evaled_xs]
            )
            counter += 1
            if best_dist < curr_closest_dist:
                curr_closest_dist = best_dist
                curr_closest_pt = new_x

        return curr_closest_pt

    def _sample_gaussian(self, best_evaled_x: NDArray, counter: int) -> NDArray:
        """Sample a point using Gaussian distribution."""
        possible_max = (self.domain[1] - best_evaled_x) / np.exp(counter)
        possible_min = (self.domain[0] - best_evaled_x) / np.exp(counter)
        possible_values = np.max(
            np.vstack([np.abs(possible_max), np.abs(possible_min)]), axis=0
        )
        new_x = np.random.normal(best_evaled_x, possible_values)
        while np.any(new_x > self.domain[1]) or np.any(new_x < self.domain[0]):
            new_x = np.random.normal(best_evaled_x, possible_values)
        return new_x

    def _sample_centered_uniform(
        self, best_evaled_x: NDArray, counter: int
    ) -> NDArray:
        """Sample a point using centered uniform distribution."""
        possible_max = (self.domain[1] - best_evaled_x) / np.exp(counter)
        possible_min = (self.domain[0] - best_evaled_x) / np.exp(counter)
        possible_values = np.random.uniform(possible_min, possible_max, (self.dim_x,))
        new_x = best_evaled_x + possible_values
        while np.any(new_x > self.domain[1]) or np.any(new_x < self.domain[0]):
            possible_values = np.random.uniform(
                possible_min, possible_max, (self.dim_x,)
            )
            new_x = best_evaled_x + possible_values
        return new_x

    def _sample_uniform(self, counter: int) -> NDArray:
        """Sample a point using uniform distribution."""
        new_x = np.random.uniform(self.domain[0], self.domain[1])
        if counter > self.switch_counter:
            if "hybrid" in self.sampling_mode:
                if "gaussian" in self.sampling_mode:
                    self.GAUSSIAN = True
                else:
                    self.CENTERED_UNIFORM = True
            else:
                return new_x
        return new_x

    def sample_from_uniform(self) -> NDArray:
        """Sample a point from uniform distribution within the domain."""
        return np.random.uniform(
            self.domain[0], self.domain[1], (1, self.dim_x)
        ).squeeze()

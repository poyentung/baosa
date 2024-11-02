from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
from numpy.typing import NDArray
import numpy as np


@dataclass
class BaseOptimisation:
    """Base class for optimisation methods."""

    f: Optional[Callable] = None
    dims: int = 20
    num_samples_per_acquisition: int = 20
    model: Optional[object] = None
    name: Optional[str] = None
    search_method: Optional[str] = None
    turn: float = field(init=False)
    bounds: List[Tuple[float, float]] = field(init=False)
    mode: Optional[str] = None
    all_proposed: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        """Initialize attributes after object creation."""
        if self.f is not None:
            self.turn = self.f.turn
            self.bounds = [
                (float(self.f.lb[idx]), float(self.f.ub[idx]))
                for idx in range(len(self.f.lb))
            ]

    def data_process(self, X: NDArray, boards: List[NDArray]) -> NDArray:
        """Process and filter unique boards."""
        unique_boards = np.unique(np.array(boards), axis=0)
        new_x = [
            board for board in unique_boards if not np.any(np.all(board == X, axis=1))
        ]
        print(f"Unique number of boards: {len(new_x)}")
        return np.array(new_x)

    def predict(self, x: NDArray) -> float:
        """Make predictions using the model."""
        x = np.round(x, int(-np.log10(self.turn)))
        self.all_proposed.append(x)

        try:
            pred = self.model.predict(
                x.reshape(len(x), self.dims, 1), verbose=False
            ).flatten()
        except ValueError:
            pred = self.model.predict(
                x.reshape(1, self.dims, 1), verbose=False
            ).flatten()
        return float(pred)

    def get_top_X(self, X: NDArray, num_top_samples: int) -> NDArray:
        """Get top X values based on model predictions."""
        new_x = self.data_process(X, self.all_proposed)
        new_pred = self.model.predict(
            new_x.reshape(len(new_x), -1, 1), verbose=False
        ).flatten()
        if len(new_x) == num_top_samples:
            return new_x
        if len(new_x) > num_top_samples:
            ind = np.argsort(new_pred)
            top_X = new_x[ind[-num_top_samples:]]
        else:
            dummy = np.arange(self.f.lb[0], self.f.ub[0] + self.turn, self.turn).round(
                5
            )
            random_X = np.random.choice(
                dummy, size=(num_top_samples - len(new_x), self.dims)
            )
            top_X = np.concatenate((new_x, random_X), axis=0)

        return top_X

    def single_rollout(self, *args, **kwargs):
        """Placeholder for single rollout method."""
        raise NotImplementedError("Subclasses must implement single_rollout method.")

    def rollout(
        self, X: NDArray, y: NDArray, rollout_round: int, method_args: dict = {}
    ) -> NDArray:
        """Perform rollout based on the optimisation problem."""
        if self.name in ["rastrigin", "ackley"]:
            index_max = np.argmax(y)
            initial_X = X[index_max, :]
            top_X = self.single_rollout(
                X, initial_X, rollout_round, method_args=method_args
            )
        else:
            top_X = self._rollout_for_other_problems(X, y, rollout_round, method_args)

        return top_X

    def _rollout_for_other_problems(
        self, X: NDArray, y: NDArray, rollout_round: int, method_args: dict
    ) -> NDArray:
        """Perform rollout for problems other than rastrigin and ackley."""
        ind = np.argsort(y)
        x_current_top = self._get_unique_top_points(X, ind)

        X_top = []
        for initial_X in x_current_top:
            top_X = self.single_rollout(X, initial_X, rollout_round, top_n=6, top_n2=1)
            X_top.append(top_X)

        top_X = np.vstack(X_top)
        return top_X[-self.num_samples_per_acquisition :]

    def _get_unique_top_points(self, X: NDArray, ind: NDArray) -> NDArray:
        """Get unique top points for rollout."""
        x_current_top = X[ind[-3:]]
        x_current_top = np.unique(x_current_top, axis=0)
        i = -4
        while len(x_current_top) < 3:
            x_current_top = np.concatenate((x_current_top, X[ind[i]].reshape(1, -1)))
            i -= 1
            x_current_top = np.unique(x_current_top, axis=0)
        return x_current_top[:3]

import numpy as np
from numpy.typing import NDArray
from bayes_opt import BayesianOptimization

from balsa.search_methods.base import BaseOptimisation


class BO(BaseOptimisation):
    """Bayesian Optimisation.

    This class provides an interface for Bayesian Optimisation using the `bayes_opt`
    library, supporting both minimisation and maximisation problems through
    automatic conversion.
    """

    def interface(self, **kwargs: float) -> float:
        """Interfaces between `bayes_opt` and the objective function.

        Args:
            **kwargs: Parameter dictionary where keys are dimension names
                and values are parameter values.

        Returns:
            float: Converted objective function value for maximisation.
        """
        x = np.array(list(kwargs.values()))
        return self.exact_f(x)

    def exact_f(self, x: NDArray | list[float]) -> float:
        if isinstance(self.f(x), float):
            return self.min_max_conversion(self.f(x))
        return self.min_max_conversion(self.f(x)[0])

    def min_max_conversion(self, y: float) -> float:
        return 1 / y

    def run(self, num_samples: int, num_init_samples: int = 200) -> None:
        bo_optimiser = BayesianOptimization(
            f=self.interface,
            pbounds={
                f"dim_{p}": (lb, ub)
                for p, lb, ub in zip(range(self.f.dims), self.f.lb, self.f.ub)
            },
            random_state=1,
        )

        bo_optimiser.maximize(init_points=num_init_samples, n_iter=num_samples)

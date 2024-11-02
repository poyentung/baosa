from numpy.typing import NDArray
from .base import BaseOptimisation
from lamcts import MCTS


class LaMCTS(BaseOptimisation):
    """Wrapper for Latent Multi-Armed Bandit Tree Search (LaMCTS) optimisation.

    This class wraps the implementation of the LaMCTS algorithm for optimisation
    problems, extending the BaseOptimisation class.
    """

    def exact_f(self, x: NDArray | list[float]) -> float:
        """Ensures the objective function returns a single float value.

        Args:
            x: Input vector as numpy array or list of floats.

        Returns:
            float: Single objective function value.
        """
        if isinstance(self.f(x), float):
            return self.f(x)
        return self.f(x)[0]

    def run(
        self,
        num_samples: int,
        num_init_samples: int = 200,
        Cp: float = 1,
        leaf_size: int = 10,
        kernel_type: str = "linear",
        gamma_type: str = "auto",
    ) -> None:
        """Runs the LaMCTS optimization process.

        Args:
            num_samples: Number of samples for the search process.
            num_init_samples: Number of initial random samples.
            Cp: Exploration constant for MCTS.
            leaf_size: Maximum size of tree leaves.
            kernel_type: SVM kernel type ('linear', 'rbf', etc.).
            gamma_type: SVM gamma parameter ('auto' or 'scale').
        """
        if num_samples <= 0 or num_init_samples <= 0:
            raise ValueError("Sample counts must be positive integers")
        if Cp <= 0:
            raise ValueError("Cp must be positive")
        if leaf_size <= 0:
            raise ValueError("Leaf size must be positive")
        if kernel_type not in ["linear", "rbf", "poly"]:
            raise ValueError("Unsupported kernel type")
        if gamma_type not in ["auto", "scale"]:
            raise ValueError("Unsupported gamma type")

        try:
            agent = MCTS(
                lb=self.f.lb,  # the lower bound of each problem dimensions
                ub=self.f.ub,  # the upper bound of each problem dimensions
                dims=self.dims,  # the problem dimensions
                ninits=num_init_samples,  # the number of random samples used in initialisations
                func=self.exact_f,  # function object to be optimized
                Cp=Cp,  # Cp for MCTS
                leaf_size=leaf_size,  # Tree leaf size
                kernel_type=kernel_type,  # SVM configruation
                gamma_type=gamma_type,  # SVM configruation
            )

            agent.search(iterations=num_samples)
        except Exception as e:
            raise RuntimeError(f"MCTS optimization failed: {str(e)}") from e

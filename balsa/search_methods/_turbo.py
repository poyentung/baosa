from numpy.typing import NDArray
from turbo.turbo_1 import Turbo1
from turbo.turbo_m import TurboM

from .base import BaseOptimisation


class TuRBO(BaseOptimisation):
    """Trust Region Bayesian Optimization (TuRBO) implementation.

    This class wraps both TuRBO-1 and TuRBO-M variants for Bayesian optimisation
    with trust regions.
    """

    def exact_f(self, x: NDArray | list[float]) -> float:
        if isinstance(self.f(x), float):
            return self.f(x)
        return self.f(x)[0]

    def run(
        self,
        num_samples: int,
        num_init_samples: int = 200,
        n_trust_regions: int = 5,
        n_repeat: int = 1,
        batch_size: int = 1,
        verbose: bool = True,
        use_ard: bool = True,
        max_cholesky_size: int = 2000,
        n_training_steps: int = 50,
        min_cuda: int = 1024,
        device: str = "cpu",
        dtype: str = "float32",
    ) -> None:
        """Run the TuRBO optimisation process.

        Args:
            num_samples: Maximum number of evaluations.
            num_init_samples: Number of initial bounds from a Latin hypercube design.
            n_trust_regions: Number of trust regions (1 for TuRBO-1, >1 for TuRBO-M).
            n_repeat: Number of repeat times for the same condition.
            batch_size: Batch size for parallel evaluations.
            verbose: Whether to print information from each batch.
            use_ard: Whether to use ARD for the GP kernel.
            max_cholesky_size: Threshold for switching from Cholesky to Lanczos.
            n_training_steps: Number of ADAM steps to learn hyperparameters.
            min_cuda: Minimum dataset size for GPU usage.
            device: Computation device ('cpu' or 'cuda').
            dtype: Data type precision ('float32' or 'float64').
        """

        for _ in range(n_repeat):
            agent = self._create_agent(
                num_init_samples,
                num_samples,
                n_trust_regions,
                batch_size,
                verbose,
                use_ard,
                max_cholesky_size,
                n_training_steps,
                min_cuda,
                device,
                dtype,
            )
            agent.optimize()

    def _create_agent(
        self,
        num_init_samples: int,
        num_samples: int,
        n_trust_regions: int,
        batch_size: int,
        verbose: bool,
        use_ard: bool,
        max_cholesky_size: int,
        n_training_steps: int,
        min_cuda: int,
        device: str,
        dtype: str,
    ) -> Turbo1 | TurboM:
        """Create a TuRBO agent."""

        common_params = {
            "f": self.exact_f,
            "lb": self.f.lb,
            "ub": self.f.ub,
            "max_evals": num_samples,
            "batch_size": batch_size,
            "verbose": verbose,
            "use_ard": use_ard,
            "max_cholesky_size": max_cholesky_size,
            "n_training_steps": n_training_steps,
            "min_cuda": min_cuda,
            "device": device,
            "dtype": dtype,
        }

        if n_trust_regions == 1:
            n_init = num_init_samples
            return Turbo1(n_init=n_init, **common_params)

        return TurboM(
            n_init=num_init_samples // n_trust_regions,
            n_trust_regions=n_trust_regions,
            **common_params,
        )

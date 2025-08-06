from typing import Literal

import numpy as np
import torch
from botorch import fit_fully_bayesian_model_nuts
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley as BoTorchAckley
from botorch.test_functions import Griewank as BoTorchGriewank
from botorch.test_functions import Michalewicz as BoTorchMichalewicz
from botorch.test_functions import Rastrigin as BoTorchRastrigin
from botorch.test_functions import Rosenbrock as BoTorchRosenbrock
from botorch.test_functions import SyntheticTestFunction
from torch import Tensor
from torch.quasirandom import SobolEngine

from balsa.obj_func import ObjectiveFunction
from balsa.search_methods.base import BaseOptimisation

torch.set_default_dtype(torch.float64)  # Use double precision for better accuracy


class Schwefel(SyntheticTestFunction):
    _optimal_value = 0.0

    def __init__(
        self,
        dim=2,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
            dtype: The dtype that is used for the bounds of the function.
        """
        self.dim = dim
        self.continuous_inds = list(range(dim))
        if bounds is None:
            bounds = [(-500, 500) for _ in range(self.dim)]
        self._optimizers = [tuple(420.9687 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, dtype=dtype)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return 418.9829 * self.dim + torch.sum(
            X * torch.sin(torch.sqrt(torch.abs(X))), dim=-1
        )


class SaasBO(BaseOptimisation):
    """Bayesian Optimisation using BoTorch's SAAS model.

    Attributes:
        f: Objective function to optimize.
        dims: Number of dimensions.
    """

    def exact_f(self, x: torch.Tensor) -> float:
        """Evaluate the objective function at x, ensuring float output.

        Args:
            x: Input vector as torch tensor.

        Returns:
            float: Objective function value.
        """
        # Try to use BoTorch backend if available
        botorch_func = self._get_botorch_function(self.f, x.device, x.dtype)
        if botorch_func is not None:
            # Use BoTorch for evaluation but maintain original tracking behavior
            # Apply discretization in PyTorch
            turn_tensor = torch.tensor(self.f.turn, dtype=x.dtype, device=x.device)
            x_processed = torch.round(x / turn_tensor) * turn_tensor
            x_processed = x_processed.to(dtype=x.dtype, device=x.device)
            self.f.counter += int(x_processed.shape[0])  # Increment counter

            # Evaluate using BoTorch (already a tensor)
            y = botorch_func(x_processed)

            # Convert to numpy only for tracking
            x_np_for_tracking = x_processed.detach().cpu().numpy()
            y_float = y.detach().cpu().numpy().tolist()

            # track the evaluation
            for y_i, x_i in zip(y_float, x_np_for_tracking):
                self.f.tracker.track(y_i, x_i, saver=True)

            return y
        else:
            raise ValueError(
                f"No BoTorch function found for the objective function: {self.f.name}"
            )

    def run(
        self,
        num_acquisitions: int,
        num_init_samples: int = 200,
        num_samples_per_acquisition: int = 20,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        dtype: Literal["float32", "float64", "double"] = "float64",
        warmup_steps: int = 64,
        thinning: int = 16,
        num_mcmc_samples: int = 128,
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        """Run the SAASBO optimisation process.

        Args:
            num_acquisitions: Number of optimisation steps (batches).
            num_init_samples: Number of initial Sobol samples.
            batch_size: Number of candidates per batch.
            device: Device for computation ('cpu' or 'cuda').
            dtype: Torch dtype (default: torch.double).
            warmup_steps: NUTS warmup steps.
            thinning: NUTS thinning parameter.
            num_mcmc_samples: Number of MCMC samples.
            seed: Random seed for reproducibility.
            verbose: Whether to print progress.
        """
        seed = np.random.randint(0, 100000)

        # Convert string dtype to torch.dtype
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "double": torch.double,
        }
        torch_dtype = dtype_map[dtype]

        tkwargs = {"device": torch.device(device), "dtype": torch_dtype}
        dim = self.dims
        bounds = torch.stack(
            [
                torch.tensor(self.f.lb).to(**tkwargs),
                torch.tensor(self.f.ub).to(**tkwargs),
            ]
        ).to(**tkwargs)

        # Initial design
        # X = torch.rand(num_init_samples, dim, **tkwargs)
        X = (
            SobolEngine(dimension=dim, scramble=True, seed=seed)
            .draw(num_init_samples)
            .to(**tkwargs)
        )
        # Scale X to the bounds
        X = bounds[0] + (bounds[1] - bounds[0]) * X
        Y = self.exact_f(X).unsqueeze(-1)
        X = X.type(tkwargs["dtype"])
        Y = Y.type(tkwargs["dtype"])

        for _ in range(num_acquisitions):
            train_Y = -1 * Y  # Flip sign for minimisation
            gp = SaasFullyBayesianSingleTaskGP(
                train_X=X,
                train_Y=train_Y,
                train_Yvar=torch.full_like(train_Y, 1e-6),
                input_transform=Normalize(d=dim, bounds=bounds),
                outcome_transform=Standardize(m=1),
            )
            fit_fully_bayesian_model_nuts(
                gp,
                warmup_steps=warmup_steps,
                num_samples=num_mcmc_samples,
                thinning=thinning,
                disable_progbar=not verbose,
            )
            EI = qLogExpectedImprovement(model=gp, best_f=train_Y.max())
            candidates, _ = optimize_acqf(
                EI,
                bounds=bounds,
                q=num_samples_per_acquisition,
                num_restarts=10,
                raw_samples=1024,
            )
            Y_next = self.exact_f(candidates).unsqueeze(-1)

            X = torch.cat((X, candidates))
            Y = torch.cat((Y, Y_next))

        return None

    def _get_botorch_function(
        self, obj_func: ObjectiveFunction, device: torch.device, dtype: torch.dtype
    ):
        """Create a BoTorch synthetic function based on the objective function name.

        This allows us to use BoTorch's optimized implementations while maintaining
        compatibility with the existing ObjectiveFunction interface.

        Args:
            obj_func: The objective function instance
            device: Device to place the bounds tensor on
            dtype: Data type for the bounds tensor

        Returns:
            BoTorch synthetic function instance with bounds copied from ObjectiveFunction
        """
        name = obj_func.name.lower()
        dim = obj_func.dims

        # Create bounds on the correct device and dtype
        bounds = torch.column_stack(
            [
                torch.tensor(obj_func.lb, device=device, dtype=dtype),
                torch.tensor(obj_func.ub, device=device, dtype=dtype),
            ]
        )

        # Create the BoTorch function
        botorch_func = None
        match name:
            case "ackley":
                botorch_func = BoTorchAckley(dim=dim, bounds=bounds, dtype=dtype)
            case "rastrigin":
                botorch_func = BoTorchRastrigin(dim=dim, bounds=bounds, dtype=dtype)
            case "griewank":
                botorch_func = BoTorchGriewank(dim=dim, bounds=bounds, dtype=dtype)
            case "rosenbrock":
                botorch_func = BoTorchRosenbrock(dim=dim, bounds=bounds, dtype=dtype)
            case "michalewicz":
                botorch_func = BoTorchMichalewicz(dim=dim, bounds=bounds, dtype=dtype)
            case "schwefel":
                botorch_func = Schwefel(dim=dim, bounds=bounds, dtype=dtype)
            case _:
                raise ValueError(f"Unknown objective function: {name}")
        return botorch_func

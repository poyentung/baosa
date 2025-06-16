from typing import Literal
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch import fit_fully_bayesian_model_nuts
from botorch.test_functions import (
    Ackley as BoTorchAckley,
    Rastrigin as BoTorchRastrigin,
    Griewank as BoTorchGriewank,
    Rosenbrock as BoTorchRosenbrock,
    Michalewicz as BoTorchMichalewicz,
)

from balsa.obj_func import ObjectiveFunction
from balsa.search_methods.base import BaseOptimisation


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
        botorch_func = self._get_botorch_function(self.f)
        if botorch_func is not None:
            # Use BoTorch for evaluation but maintain original tracking behavior
            # Apply discretization in PyTorch
            turn_tensor = torch.tensor(self.f.turn, dtype=x.dtype, device=x.device)
            x_processed = torch.round(x / turn_tensor) * turn_tensor
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
        device: Literal["cpu", "cuda"] = "cpu",
        dtype: torch.dtype = torch.double,
        warmup_steps: int = 256,
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
        torch.manual_seed(seed)
        np.random.seed(seed)

        tkwargs = {"device": torch.device(device), "dtype": dtype}
        dim = self.dims
        bounds = torch.stack(
            [torch.tensor(self.f.lb, **tkwargs), torch.tensor(self.f.ub, **tkwargs)]
        )

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

        for _ in range(num_acquisitions):
            train_Y = -1 * Y  # Flip sign for minimisation
            gp = SaasFullyBayesianSingleTaskGP(
                train_X=X,
                train_Y=train_Y,
                train_Yvar=torch.full_like(train_Y, 1e-6),
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

    def _get_botorch_function(self, obj_func: ObjectiveFunction):
        """Create a BoTorch synthetic function based on the objective function name.

        This allows us to use BoTorch's optimized implementations while maintaining
        compatibility with the existing ObjectiveFunction interface.

        Args:
            obj_func: The objective function instance

        Returns:
            BoTorch synthetic function instance with bounds copied from ObjectiveFunction
        """
        name = obj_func.name.lower()
        dim = obj_func.dims

        bounds = [
            (float(low), float(high)) for (low, high) in zip(obj_func.lb, obj_func.ub)
        ]

        # Create the BoTorch function
        botorch_func = None
        match name:
            case "ackley":
                botorch_func = BoTorchAckley(dim=dim, bounds=bounds)
            case "rastrigin":
                botorch_func = BoTorchRastrigin(dim=dim, bounds=bounds)
            case "griewank":
                botorch_func = BoTorchGriewank(dim=dim, bounds=bounds)
            case "rosenbrock":
                botorch_func = BoTorchRosenbrock(dim=dim, bounds=bounds)
            case "michalewicz":
                botorch_func = BoTorchMichalewicz(dim=dim, bounds=bounds)
            case _:
                raise ValueError(f"Unknown objective function: {name}")
        return botorch_func

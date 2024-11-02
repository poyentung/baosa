import random
from typing import Any

import cma
import nevergrad as ng
from numpy.typing import NDArray
import numpy as np
from scipy.optimize import dual_annealing, differential_evolution

from balsa.search_methods.base import BaseOptimisation


class DualAnnealing(BaseOptimisation):
    """Dual Annealing optimization algorithm."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def single_rollout(
        self,
        X: NDArray,
        x_current: NDArray,
        rollout_round: int,
        num_samples_per_acquisition: int = 20,
        method_args: dict[str, Any] = {"initial_temp": 0.05},
    ) -> NDArray:
        """Perform a single rollout of the Dual Annealing algorithm."""
        if self.mode == "fast":
            ret = dual_annealing(
                self.predict,
                bounds=self.bounds,
                x0=x_current,
                maxfun=rollout_round,
                **method_args,
            )
        elif self.mode == "origin":
            ret = dual_annealing(self.predict, bounds=self.bounds, x0=x_current)

        self.all_proposed.append(np.round(ret.x, int(-np.log10(self.f.turn))))
        return self.get_top_X(X, num_samples_per_acquisition)


class DifferentialEvolution(BaseOptimisation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def single_rollout(
        self,
        X: NDArray,
        x_current: NDArray,
        rollout_round: int,
        num_samples_per_acquisition: int = 16,
        method_args: dict[str, Any] = {},
    ) -> NDArray:
        """Perform a single rollout of the Differential Evolution algorithm."""
        if self.mode == "fast":
            popsize = int(max(100 / self.f.dims, 1))
            ret = differential_evolution(
                self.predict,
                bounds=self.bounds,
                x0=x_current,
                maxiter=1,
                popsize=popsize,
                **method_args,
            )
        elif self.mode == "origin":
            ret = differential_evolution(self.predict, bounds=self.bounds, x0=x_current)
        self.all_proposed.append(np.round(ret.x, int(-np.log10(self.f.turn))))
        return self.get_top_X(X, num_samples_per_acquisition)


class CMAES(BaseOptimisation):
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES) optimisation algorithm."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def single_rollout(
        self,
        X: NDArray,
        x_current: NDArray,
        rollout_round: int,
        num_samples_per_acquisition: int = 20,
        method_args: dict[str, Any] = {},
    ) -> NDArray:
        """Perform a single rollout of the CMA-ES algorithm."""
        if self.mode == "fast":
            options = {
                "maxiter": int(rollout_round / 10),
                "bounds": [self.f.lb[0], self.f.ub[0]],
            }
            es = cma.fmin(self.predict, x_current, 0.5, options, **method_args)
        elif self.mode == "origin":
            options = {"bounds": [self.f.lb[0], self.f.ub[0]]}
            es = cma.fmin(
                self.predict,
                x_current,
                (self.f.ub[0] - self.f.lb[0]) / 4,
                options,
                **method_args,
            )
        return self.get_top_X(X, num_samples_per_acquisition)


class Shiwa(BaseOptimisation):
    """Shiwa optimisation algorithm."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def single_rollout(
        self,
        X: NDArray,
        x_current: NDArray,
        rollout_round: int,
        num_samples_per_acquisition: int = 20,
        method_args: dict[str, Any] = {},
    ) -> NDArray:
        param = ng.p.Array(init=x_current).set_bounds(self.f.lb, self.f.ub)
        optimizer = ng.optimization.optimizerlib.Shiwa(
            parametrization=param, budget=rollout_round, **method_args
        )
        optimizer.minimize(self.predict)
        return self.get_top_X(X, num_samples_per_acquisition)


class MCMC(BaseOptimisation):
    """Markov Chain Monte Carlo (MCMC) optimisation algorithm."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def choose(self, board):
        "Choose the best successor of node. (Choose a move in the game)"
        turn = self.turn
        aaa = np.arange(self.f.lb[0], self.f.ub[0] + self.f.turn, self.f.turn).round(5)
        index = np.random.randint(0, self.dims)
        tup = np.array(board)
        flip = random.randint(0, 5)

        if flip == 0:
            tup[index] += turn
        elif flip == 1:
            tup[index] -= turn
        elif flip == 2:
            for i in range(int(self.dims / 5)):
                index_2 = random.randint(0, len(tup) - 1)
                tup[index_2] = np.random.choice(aaa)
        elif flip == 3:
            for i in range(int(self.dims / 10)):
                index_2 = random.randint(0, len(tup) - 1)
                tup[index_2] = np.random.choice(aaa)
        elif flip == 4:
            tup[index] = np.random.choice(aaa)
        elif flip == 5:
            tup[index] = np.random.choice(aaa)

        tup[index] = round(tup[index], 5)
        ind1 = np.where(tup > self.f.ub[0])[0]

        if len(ind1) > 0:
            tup[ind1] = self.f.ub[0]
        ind1 = np.where(tup < self.f.lb[0])[0]
        if len(ind1) > 0:
            tup[ind1] = self.f.lb[0]
        value = self.model.predict(np.array(tup).reshape(1, -1, 1), verbose=False)
        value = np.array(value).reshape(1)
        return tup, value

    def single_rollout(
        self,
        X,
        x_current,
        rollout_round,
        num_samples_per_acquisition=20,
        method_args={},
    ):
        values = self.model.predict(
            np.array(x_current).reshape(1, -1, 1), verbose=False
        )
        cu_Y = np.array(values).reshape(-1)

        boards = []
        for i in range(rollout_round):
            board, temp_Y = self.choose(x_current)
            boards.append(board)
            if temp_Y > cu_Y * 1:
                x_current = np.array(board)
                cu_Y = np.array(temp_Y)

        new_x = self.data_process(X, boards)
        new_pred = self.model.predict(
            np.array(new_x).reshape(len(new_x), -1, 1), verbose=False
        )
        new_pred = np.array(new_pred).reshape(len(new_x))

        if len(new_x) >= num_samples_per_acquisition:
            ind = np.argsort(new_pred)[-num_samples_per_acquisition:]
            top_X = new_x[ind]
        else:
            aaa = np.arange(
                self.f.lb[0], self.f.ub[0] + self.f.turn, self.f.turn
            ).round(5)
            random_X = np.random.choice(
                aaa, size=(num_samples_per_acquisition - len(new_x), self.f.dims)
            )
            top_X = np.concatenate((new_x, random_X), axis=0)
        return top_X

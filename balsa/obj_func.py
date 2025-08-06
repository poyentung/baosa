from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, override

import numpy as np
from numpy.typing import NDArray
from balsa.utils import Tracker


@dataclass
class ObjectiveFunction(ABC):
    """Abstract base class for optimisation functions."""

    dims: int = field(default=3)
    name: str = field(init=False)
    turn: float = field(default=0.1)
    iters: Optional[int] = None
    func_args: Dict[str, Any] = field(default_factory=dict)

    lb: Optional[NDArray] = None
    ub: Optional[NDArray] = None
    counter: int = 0
    tracker: Tracker = field(init=False)

    def __post_init__(self):
        """Initialise the tracker after object creation."""
        self.tracker = Tracker(f"{self.name}-{self.dims}")

    @abstractmethod
    def _scaled(self, y: float) -> float:
        pass

    @abstractmethod
    def __call__(
        self, x: NDArray, saver: bool = True, return_scaled: bool = False
    ) -> float:
        """
        Abstract method to be implemented by subclasses.
        """
        pass


class Ackley(ObjectiveFunction):
    name = "ackley"

    def __post_init__(self):
        super().__post_init__()
        self.lb = -5 * np.ones(self.dims)
        self.ub = 5 * np.ones(self.dims)

    @override
    def _scaled(self, y: float) -> float:
        return 100 / (y + 0.01)

    @override
    def __call__(self, x: NDArray, saver: bool = True, return_scaled=False) -> float:
        x = x.reshape(self.dims)
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        y = float(
            -20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size))
            - np.exp(np.cos(2 * np.pi * x).sum() / x.size)
            + 20
            + np.e
        )
        self.tracker.track(y, x, saver)
        return y if not return_scaled else self._scaled(y)


class Rastrigin(ObjectiveFunction):
    name = "rastrigin"

    def __post_init__(self):
        super().__post_init__()
        self.lb = -5 * np.ones(self.dims)
        self.ub = 5 * np.ones(self.dims)

    @override
    def _scaled(self, y: float) -> float:
        return -1 * y

    @override
    def __call__(
        self, x: NDArray, A: float = 10.0, saver: bool = True, return_scaled=False
    ) -> float:
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        n = len(x)
        sum = np.sum(x**2 - A * np.cos(2 * np.pi * x))
        y = float(A * n + sum)

        self.tracker.track(y, x, saver)
        return y if not return_scaled else self._scaled(y)


class Rosenbrock(ObjectiveFunction):
    name = "rosenbrock"

    def __post_init__(self):
        super().__post_init__()
        self.lb = -5 * np.ones(self.dims)
        self.ub = 5 * np.ones(self.dims)

    @override
    def _scaled(self, y: float) -> float:
        return 100 / (y / (self.dims * 100) + 0.01)

    @override
    def __call__(self, x: NDArray, saver: bool = True, return_scaled=False) -> float:
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        y = float(np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0))

        self.tracker.track(y, x, saver)
        return y if not return_scaled else self._scaled(y)


class Griewank(ObjectiveFunction):
    name = "griewank"

    def __post_init__(self):
        super().__post_init__()
        self.lb = -600 * np.ones(self.dims)
        self.ub = 600 * np.ones(self.dims)

    @override
    def _scaled(self, y: float) -> float:
        return 10 / (y / (self.dims) + 0.001)

    @override
    def __call__(self, x: NDArray, saver: bool = True, return_scaled=False) -> float:
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        sum_term = np.sum(x**2)
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        y = float(1 + sum_term / 4000 - prod_term)

        self.tracker.track(y, x, saver)
        return y if not return_scaled else self._scaled(y)


class Michalewicz(ObjectiveFunction):
    name = "michalewicz"

    def __post_init__(self):
        super().__post_init__()
        self.lb = np.zeros(self.dims)
        self.ub = np.pi * np.ones(self.dims)

    @override
    def _scaled(self, y: float) -> float:
        return -1 * y

    @override
    def __call__(
        self, x: NDArray, m: int = 10, saver: bool = True, return_scaled=False
    ) -> float:
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        d = len(x)
        y = 0.0
        for i in range(d):
            y += float(np.sin(x[i]) * np.sin((i + 1) * x[i] ** 2 / np.pi) ** (2 * m))
        y *= -1.0

        self.tracker.track(y, x, saver)
        return y if not return_scaled else self._scaled(y)


class Schwefel(ObjectiveFunction):
    name = "schwefel"

    def __post_init__(self):
        super().__post_init__()
        self.lb = -500 * np.ones(self.dims)
        self.ub = 500 * np.ones(self.dims)

    @override
    def _scaled(self, y: float) -> float:
        if np.isclose(y, 0.0):
            return 10000
        return -1 * y / 100.0

    @override
    def __call__(self, x: NDArray, saver: bool = True, return_scaled=False) -> float:
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        dimension = len(x)
        sum_part = np.sum(-x * np.sin(np.sqrt(np.abs(x))))

        if np.all(np.array(x) == 421, axis=0):
            return 0

        y = float(418.9829 * dimension + sum_part)

        self.tracker.track(y, x, saver)
        return y if not return_scaled else self._scaled(y)


class Sphere(ObjectiveFunction):
    name = "sphere"

    def __post_init__(self):
        super().__post_init__()
        self.lb = -5 * np.ones(self.dims)
        self.ub = 5 * np.ones(self.dims)

    @override
    def _scaled(self, y: float) -> float:
        return y

    @override
    def __call__(self, x: NDArray, saver: bool = True, return_scaled=False) -> float:
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        sum = np.sum(x**2)
        y = float(sum)

        self.tracker.track(y, x, saver)
        return y if not return_scaled else self._scaled(y)


class Levy(ObjectiveFunction):
    name = "levy"

    def __post_init__(self):
        super().__post_init__()
        self.lb = -10 * np.ones(self.dims)
        self.ub = 10 * np.ones(self.dims)

    @override
    def _scaled(self, y: float) -> float:
        return y

    @override
    def __call__(self, x: NDArray, saver: bool = True, return_scaled=False) -> float:
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        if len(x) == 1:
            w = 1 + (x[0] - 1) / 4
            sum = np.sin(np.pi * w) ** 2
        else:
            w = 1 + (x - 1) / 4
            sum = (
                np.sin(np.pi * w[0]) ** 2
                + np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
                + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
            )
        y = float(sum)

        self.tracker.track(y, x, saver)
        return y if not return_scaled else self._scaled(y)

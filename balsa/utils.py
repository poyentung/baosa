import os
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Optional, Callable
import logging
import numpy as np
from numpy.typing import NDArray
logger = logging.getLogger(__name__)


class SearchMode(StrEnum):
    fast = auto()
    original = auto()


@dataclass
class Tracker:
    """A class for tracking optimisation results."""

    foldername: str
    counter: int = 0
    results: list[float] = field(default_factory=list)
    x_values: list[Optional[NDArray]] = field(default_factory=list)
    current_best: float = float("inf")
    current_best_x: Optional[NDArray] = None

    def __post_init__(self):
        """Create the folder for saving results after initialization."""
        try:
            os.makedirs(self.foldername, exist_ok=True)
            logger.info(f"Successfully created the directory {self.foldername}")
        except OSError:
            logger.error(f"Creation of the directory {self.foldername} failed")

    def dump_trace(self):
        """Save the current results to a file."""
        np.save(
            os.path.join(self.foldername, "result.npy"),
            np.array(self.results),
            allow_pickle=True,
        )

    def track(self, result: float, x: Optional[NDArray] = None, saver: bool = False):
        """
        Track the optimization progress.

        Args:
            result: The current optimization result.
            x: The current input values (optional).
            saver: Whether to save the results immediately.
        """
        self.counter += 1
        if result < self.current_best:
            self.current_best = result
            self.current_best_x = x

        # Create messages first to calculate max length
        messages = [
            f"total number of samples: {len(self.results) + 1}",
            f"current best f(x): {self.current_best}",
        ]
        if self.current_best_x is not None:
            messages.append(
                f"current best x: {np.around(self.current_best_x, decimals=4)}"
            )

        # Calculate separator length based on longest message
        max_length = max(len(msg) for msg in messages)
        separator = "=" * max_length

        # Log messages with consistent separator
        logger.info(separator)
        for msg in messages:
            logger.info(msg)
        logger.info(separator)
        logger.info("")  # Empty line for readability

        self.results.append(self.current_best)
        self.x_values.append(x)

        if saver or self.counter % 20 == 0 or round(self.current_best, 5) == 0:
            self.dump_trace()


SyntheticFunction = Callable[[NDArray, Optional[bool], Optional[bool]], float]


def sampling_points(f: SyntheticFunction, n_samples: int = 200, return_scaled=False):
    input_X = np.concatenate(
        [np.random.uniform(lb, ub, (n_samples, 1)) for lb, ub in zip(f.lb, f.ub)],
        axis=1,
    ).round(5)
    input_y = [f(x, return_scaled=return_scaled) for x in input_X]

    _log_initial_data_points(n_samples)

    return np.array(input_X), np.array(input_y)


def _log_initial_data_points(n_samples: int):
    message = (
        f"{n_samples} initial data points collection completed, optimization started!"
    )
    separator = "=" * len(message)
    logger.info("")
    logger.info(separator)
    logger.info(message)
    logger.info(separator)
    logger.info("")

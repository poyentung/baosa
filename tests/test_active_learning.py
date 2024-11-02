"""Test cases for active learning optimisation process."""

import numpy as np
import pytest

from balsa.active_learning import OptimisationConfig, ActiveLearningPipeline
from balsa.utils import SearchMode


@pytest.fixture
def base_config():
    """Base configuration fixture for tests."""
    return {
        "dims": 3,
        "num_acquisitions": 5,
        "num_samples_per_acquisition": 10,
        "num_init_samples": 20,
        "obj_func_name": "ackley",
        "surrogate": "ackley_surrogate",
        "mode": SearchMode.fast,
    }


def test_lamcts_optimisation(base_config):
    """Test LaMCTS optimisation process."""
    config = OptimisationConfig(
        **base_config,
        search_method="lamcts",
    )
    active_learning_pipeline = ActiveLearningPipeline(config)
    active_learning_pipeline.run()
    assert True


def test_turbo_optimisation(base_config):
    """Test TuRBO optimisation process."""
    config = OptimisationConfig(**base_config, search_method="turbo")
    optimiser = ActiveLearningPipeline(config)
    optimiser.run()
    assert True


def test_bo_optimisation(base_config):
    """Test Bayesian Optimisation process."""
    config = OptimisationConfig(**base_config, search_method="bo")
    optimiser = ActiveLearningPipeline(config)
    optimiser.run()
    assert True


@pytest.mark.slow
@pytest.mark.parametrize("search_method", ["voo", "doo", "soo", "da"])
def test_dual_annealing_optimisation(base_config, search_method):
    """Test Dual Annealing optimisation process."""
    config = OptimisationConfig(
        **base_config,
        search_method=search_method,
    )

    optimiser = ActiveLearningPipeline(config)
    optimiser.run()

    # Verify optimisation results
    assert hasattr(optimiser, "input_X")
    assert hasattr(optimiser, "input_scaled_y")
    assert len(optimiser.input_X) > optimiser.config.num_init_samples


@pytest.mark.slow
@pytest.mark.parametrize("search_method", ["voo", "doo", "soo", "da"])
def test_optimisation_bounds(base_config, search_method):
    """Test if optimisation respects bounds for different search methods."""
    config = OptimisationConfig(**base_config, search_method=search_method)

    active_learning_pipeline = ActiveLearningPipeline(config)
    active_learning_pipeline.run()

    # Get the bounds from the objective function
    lb = active_learning_pipeline.obj_func.lb
    ub = active_learning_pipeline.obj_func.ub

    # For Dual Annealing, check input_X
    assert np.all(active_learning_pipeline.input_X >= lb)
    assert np.all(active_learning_pipeline.input_X <= ub)

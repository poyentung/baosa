"""Unit tests for various search and optimisation methods.

This module contains tests for different optimisation algorithms including
BO, LaMCTS, TuRBO, and others, testing their functionality with various
objective functions and dimensions.
"""

import numpy as np
from numpy.typing import NDArray
import pytest

from balsa.obj_func import Ackley, Rastrigin
from balsa.search_methods import (
    BO,
    LaMCTS,
    TuRBO,
    DualAnnealing,
    CMAES,
    Shiwa,
    MCMC,
    VOO,
    SOO,
    DOO,
)
from balsa.surrogate import AckleySurrogate
from balsa.utils import sampling_points


# Common fixtures
@pytest.fixture
def ackley_3d():
    """Creates a 3D Ackley function instance."""
    return Ackley(dims=3)


@pytest.fixture
def ackley_surrogate(ackley_3d):
    """Creates an Ackley surrogate model."""
    return AckleySurrogate(input_dimension=ackley_3d.dims)


@pytest.fixture
def optimisation_setup(ackley_3d, ackley_surrogate):
    """Sets up common optimisation parameters and training data."""
    n_samples = 20
    input_X, input_y = sampling_points(
        ackley_3d, n_samples=n_samples, return_scaled=True
    )
    surrogate = ackley_surrogate.train_and_evaluate(input_X, input_y)
    return input_X, input_y, surrogate


# Optimizer fixtures
@pytest.fixture
def bo_optimiser(ackley_3d):
    """Creates a Bayesian Optimization instance."""
    return BO(f=ackley_3d)


@pytest.fixture
def lamcts_optimiser(ackley_3d):
    """Creates a LaMCTS optimiser instance."""
    return LaMCTS(f=ackley_3d, dims=ackley_3d.dims, model=None, name=ackley_3d.name)


@pytest.fixture
def turbo1_optimiser(ackley_3d):
    """Creates a TuRBO-1 optimiser instance."""
    return TuRBO(f=ackley_3d)


@pytest.fixture
def turbo5_optimiser(ackley_3d):
    """Creates a TuRBO-M optimiser instance."""
    return TuRBO(f=ackley_3d)


@pytest.fixture
def voo_optimiser(ackley_3d):
    """Creates a VOO optimiser instance."""
    return VOO(f=ackley_3d, dims=ackley_3d.dims, model=None, name=ackley_3d.name)


@pytest.fixture
def soo_optimiser(ackley_3d):
    """Creates a SOO optimiser instance."""
    return SOO(f=ackley_3d, dims=ackley_3d.dims, model=None, name=ackley_3d.name)


@pytest.fixture
def doo_optimiser(ackley_3d):
    """Creates a DOO optimiser instance."""
    return DOO(f=ackley_3d, dims=ackley_3d.dims, model=None, name=ackley_3d.name)


# Test classes
class TestBayesianOptimization:
    """Tests for Bayesian Optimization."""

    @pytest.mark.parametrize("dims", [2, 3, 5])
    def test_dimensions(self, dims):
        """Tests BO with different dimensions."""
        f = Ackley(dims=dims)
        bo = BO(f=f)
        bo.run(num_samples=5, num_init_samples=3)

        for x in bo.all_proposed:
            assert x.shape == (dims,)
            assert np.all((f.lb <= x) & (x <= f.ub))

    @pytest.mark.parametrize("obj_func", [Ackley, Rastrigin])
    def test_different_functions(self, obj_func):
        """Tests BO with different objective functions."""
        f = obj_func(dims=3)
        bo = BO(f=f)
        bo.run(num_samples=5, num_init_samples=3)
        assert all(isinstance(x, NDArray) for x in bo.all_proposed)

    def test_optimisation(self, bo_optimiser):
        """Tests basic BO optimisation functionality."""
        bo_optimiser.run(num_samples=10, num_init_samples=5)

        assert all(isinstance(x, NDArray) for x in bo_optimiser.all_proposed)
        for x in bo_optimiser.all_proposed:
            assert np.all((bo_optimiser.f.lb <= x) & (x <= bo_optimiser.f.ub))

    def test_min_max_conversion(self, bo_optimiser):
        """Tests the min-max conversion functionality."""
        test_values = [1.0, 2.0, 5.0, 10.0]
        for value in test_values:
            converted = bo_optimiser.min_max_conversion(value)
            assert converted == 1.0 / value
            assert converted > 0


class TestLaMCTS:
    """Tests for LaMCTS optimisation."""

    @pytest.mark.parametrize("dims", [2, 3, 5])
    def test_dimensions(self, dims):
        """Tests LaMCTS with different dimensions."""
        f = Ackley(dims=dims)
        optimiser = LaMCTS(f=f, dims=f.dims, model=None, name=f.name)
        optimiser.run(num_samples=5, num_init_samples=3)

        assert f.dims == dims
        assert optimiser.dims == dims

    @pytest.mark.parametrize("obj_func", [Ackley, Rastrigin])
    def test_different_functions(self, obj_func):
        """Tests LaMCTS with different objective functions."""
        f = obj_func(dims=3)
        optimiser = LaMCTS(f=f, dims=f.dims, model=None, name=f.name)
        optimiser.run(num_samples=5, num_init_samples=3)

        assert hasattr(optimiser, "f")
        assert optimiser.dims == 3

    def test_optimisation(self, lamcts_optimiser):
        """Tests basic LaMCTS optimisation functionality."""
        lamcts_optimiser.run(num_samples=10, num_init_samples=5)

        assert lamcts_optimiser.dims == 3
        assert all(hasattr(lamcts_optimiser, attr) for attr in ["f"])
        assert all(hasattr(lamcts_optimiser.f, attr) for attr in ["lb", "ub"])

    def test_exact_f(self, lamcts_optimiser):
        """Tests the exact_f function of LaMCTS."""
        test_point = np.zeros(lamcts_optimiser.dims)
        result = lamcts_optimiser.exact_f(test_point)
        assert isinstance(result, float)


class TestTuRBO:
    """Tests for TuRBO optimisation."""

    @pytest.mark.parametrize("dims", [2, 3, 5])
    def test_dimensions(self, dims):
        """Tests TuRBO with different dimensions."""
        f = Ackley(dims=dims)
        optimiser = TuRBO(f=f, dims=f.dims, model=None, name=f.name)
        optimiser.run(num_samples=20, num_init_samples=10)

        assert f.dims == dims
        for x in optimiser.all_proposed:
            assert x.shape == (dims,)
            assert np.all((f.lb <= x) & (x <= f.ub))

    @pytest.mark.parametrize("obj_func", [Ackley, Rastrigin])
    def test_different_functions(self, obj_func):
        """Tests TuRBO with different objective functions."""
        f = obj_func(dims=3)
        optimiser = TuRBO(f=f)
        optimiser.run(num_samples=20, num_init_samples=10)

        assert hasattr(optimiser, "f")
        assert all(isinstance(x, NDArray) for x in optimiser.all_proposed)

    def test_turbo1_optimisation(self, turbo1_optimiser):
        """Tests TuRBO-1 optimisation functionality."""
        turbo1_optimiser.run(num_samples=20, num_init_samples=10, n_trust_regions=1)

        assert all(isinstance(x, NDArray) for x in turbo1_optimiser.all_proposed)
        for x in turbo1_optimiser.all_proposed:
            assert np.all((turbo1_optimiser.f.lb <= x) & (x <= turbo1_optimiser.f.ub))

    def test_turbo5_optimisation(self, turbo5_optimiser):
        """Tests TuRBO-M optimisation with 5 trust regions."""
        turbo5_optimiser.run(num_samples=20, num_init_samples=10, n_trust_regions=5)

        assert all(isinstance(x, NDArray) for x in turbo5_optimiser.all_proposed)
        for x in turbo5_optimiser.all_proposed:
            assert np.all((turbo5_optimiser.f.lb <= x) & (x <= turbo5_optimiser.f.ub))

    def test_exact_f(self, turbo1_optimiser):
        """Tests the exact_f function of TuRBO."""
        test_point = np.zeros(turbo1_optimiser.f.dims)
        result = turbo1_optimiser.exact_f(test_point)
        assert isinstance(result, float)


class TestOptimizers:
    """Tests for various optimisation algorithms."""

    @pytest.mark.parametrize(
        "optimiser_class,mode,expected_samples",
        [
            (DualAnnealing, "fast", 20),
            (CMAES, "fast", 20),
            (Shiwa, None, 20),
            (MCMC, None, 20),
        ],
    )
    def test_basic(
        self, ackley_3d, optimisation_setup, optimiser_class, mode, expected_samples
    ):
        """Tests basic optimisation functionality."""
        input_X, input_y, surrogate = optimisation_setup
        optimiser = optimiser_class(
            f=ackley_3d, dims=ackley_3d.dims, model=surrogate, name=ackley_3d.name
        )

        if mode:
            optimiser.mode = mode

        result = optimiser.rollout(input_X, input_y, rollout_round=50)

        assert isinstance(result, NDArray)
        assert result.shape == (expected_samples, ackley_3d.dims)
        assert np.all((ackley_3d.lb <= result) & (result <= ackley_3d.ub))

    @pytest.mark.parametrize(
        "optimiser_class,mode,expected_samples",
        [
            (DualAnnealing, "fast", 20),
            (CMAES, "fast", 20),
            (Shiwa, None, 20),
            (MCMC, None, 20),
        ],
    )
    @pytest.mark.parametrize("obj_func", [Ackley, Rastrigin])
    def test_different_functions(
        self, obj_func, ackley_surrogate, optimiser_class, mode, expected_samples
    ):
        """Tests optimisers with different objective functions."""
        f = obj_func(dims=3)
        input_X, input_y = sampling_points(f, n_samples=50, return_scaled=True)
        surrogate = ackley_surrogate.train_and_evaluate(input_X, input_y)

        optimiser = optimiser_class(f=f, dims=f.dims, model=surrogate, name=f.name)
        if mode:
            optimiser.mode = mode

        result = optimiser.rollout(input_X, input_y, rollout_round=expected_samples)

        assert result.shape == (expected_samples, f.dims)
        assert np.all((f.lb <= result) & (result <= f.ub))


class TestVOO:
    """Tests for VOO optimisation."""

    def test_optimisation(self, voo_optimiser, optimisation_setup):
        """Tests basic VOO optimisation functionality."""
        input_X, input_y, surrogate = optimisation_setup
        voo_optimiser.model = surrogate

        result = voo_optimiser.rollout(
            input_X, input_y, rollout_round=50, method_args={"explr_p": 0.001}
        )

        assert isinstance(result, NDArray)
        assert result.shape == (20, voo_optimiser.dims)
        assert np.all((voo_optimiser.f.lb <= result) & (result <= voo_optimiser.f.ub))

    def test_different_functions(self, ackley_surrogate):
        """Tests VOO with different objective functions."""
        f = Rastrigin(dims=3)
        input_X, input_y = sampling_points(f, n_samples=50, return_scaled=True)
        surrogate = ackley_surrogate.train_and_evaluate(input_X, input_y)

        optimiser = VOO(f=f, dims=f.dims, model=surrogate, name=f.name)
        result = optimiser.rollout(
            input_X, input_y, rollout_round=20, method_args={"explr_p": 0.001}
        )

        assert result.shape == (20, f.dims)
        assert np.all((f.lb <= result) & (result <= f.ub))


class TestSOO:
    """Tests for SOO optimisation."""

    def test_optimisation(self, soo_optimiser, optimisation_setup):
        """Tests basic SOO optimisation functionality."""
        input_X, input_y, surrogate = optimisation_setup
        soo_optimiser.model = surrogate

        result = soo_optimiser.rollout(
            input_X, input_y, rollout_round=20, method_args={}
        )

        assert isinstance(result, NDArray)
        assert result.shape == (20, soo_optimiser.dims)
        assert np.all((soo_optimiser.f.lb <= result) & (result <= soo_optimiser.f.ub))

    def test_different_functions(self, ackley_surrogate):
        """Tests SOO with different objective functions."""
        f = Rastrigin(dims=3)
        input_X, input_y = sampling_points(f, n_samples=50, return_scaled=True)
        surrogate = ackley_surrogate.train_and_evaluate(input_X, input_y)

        optimiser = SOO(f=f, dims=f.dims, model=surrogate, name=f.name)
        result = optimiser.rollout(input_X, input_y, rollout_round=20, method_args={})

        assert result.shape == (20, f.dims)
        assert np.all((f.lb <= result) & (result <= f.ub))


class TestDOO:
    """Tests for DOO optimisation."""

    def test_optimisation(self, doo_optimiser, optimisation_setup):
        """Tests basic DOO optimisation functionality."""
        input_X, input_y, surrogate = optimisation_setup
        doo_optimiser.model = surrogate

        result = doo_optimiser.rollout(
            input_X, input_y, rollout_round=20, method_args={"explr_p": 0.01}
        )

        assert isinstance(result, NDArray)
        assert result.shape == (20, doo_optimiser.dims)
        assert np.all((doo_optimiser.f.lb <= result) & (result <= doo_optimiser.f.ub))

    def test_different_functions(self, ackley_surrogate):
        """Tests DOO with different objective functions."""
        f = Rastrigin(dims=3)
        input_X, input_y = sampling_points(f, n_samples=50, return_scaled=True)
        surrogate = ackley_surrogate.train_and_evaluate(input_X, input_y)

        optimiser = DOO(f=f, dims=f.dims, model=surrogate, name=f.name)
        result = optimiser.rollout(
            input_X, input_y, rollout_round=20, method_args={"explr_p": 0.01}
        )

        assert result.shape == (20, f.dims)
        assert np.all((f.lb <= result) & (result <= f.ub))

import numpy as np
import pytest

from balsa.obj_func import (
    Ackley,
    Rastrigin,
    Rosenbrock,
    Griewank,
    Michalewicz,
    Schwefel,
)


class TestAckley:
    @pytest.mark.parametrize(
        ("dims", "input_x", "expected"),
        [
            (2, np.array([0, 0]), pytest.approx(0.0, abs=1e-4)),
            (3, np.array([1, 1, 1]), pytest.approx(3.6253849, abs=1e-4)),
            (2, np.array([-5, -5]), pytest.approx(12.64241, abs=1e-4)),
            (3, np.array([5, 5, 5]), pytest.approx(12.6424, abs=1e-4)),
        ],
    )
    def test_ackley(self, dims, input_x, expected):
        func = Ackley(dims=dims)
        result = func(input_x, saver=False)
        assert result == pytest.approx(expected, abs=1e-4)


class TestRastrigin:
    @pytest.mark.parametrize(
        ("dims", "input_x", "expected"),
        [
            (2, np.array([0, 0]), pytest.approx(0.0, abs=1e-4)),
            (3, np.array([1, 1, 1]), pytest.approx(3.0, abs=1e-4)),
            (2, np.array([-5, -5]), pytest.approx(50.0, abs=1e-4)),
            (3, np.array([5, 5, 5]), pytest.approx(75.0, abs=1e-4)),
        ],
    )
    def test_rastrigin(self, dims, input_x, expected):
        func = Rastrigin(dims=dims)
        result = func(input_x, saver=False)
        assert result == pytest.approx(expected, abs=1e-4)


class TestRosenbrock:
    @pytest.mark.parametrize(
        ("dims", "input_x", "expected"),
        [
            (2, np.array([1, 1]), pytest.approx(0.0, abs=1e-4)),
            (3, np.array([1, 2, 3]), pytest.approx(201.0, abs=1e-4)),
            (2, np.array([-5, -5]), pytest.approx(90036.0, abs=1e-4)),
            (3, np.array([5, 5, 5]), pytest.approx(80032.0, abs=1e-4)),
        ],
    )
    def test_rosenbrock(self, dims, input_x, expected):
        func = Rosenbrock(dims=dims)
        result = func(input_x, saver=False)
        assert result == pytest.approx(expected, abs=1e-4)


class TestGriewank:
    @pytest.mark.parametrize(
        ("dims", "input_x", "expected"),
        [
            (2, np.array([0, 0]), pytest.approx(0.0, abs=1e-4)),
            (3, np.array([1, 1, 1]), pytest.approx(0.65656, abs=1e-4)),
            (2, np.array([-600, -600]), pytest.approx(180.01205, abs=1e-4)),
            (3, np.array([600, 600, 600]), pytest.approx(270.3368, abs=1e-4)),
        ],
    )
    def test_griewank(self, dims, input_x, expected):
        func = Griewank(dims=dims)
        result = func(input_x, saver=False)
        assert result == pytest.approx(expected, abs=1e-4)


class TestMichalewicz:
    @pytest.mark.parametrize(
        ("dims", "input_x", "expected"),
        [
            (2, np.array([2.20, 1.57]), pytest.approx(-1.7665, abs=1e-4)),
            (3, np.array([2.20, 1.57, 1.28]), pytest.approx(-2.71245, abs=1e-4)),
            (2, np.array([0, 0]), pytest.approx(0.0, abs=1e-4)),
            (3, np.array([np.pi, np.pi, np.pi]), pytest.approx(0.0, abs=1e-4)),
        ],
    )
    def test_michalewicz(self, dims, input_x, expected):
        func = Michalewicz(dims=dims)
        result = func(input_x, saver=False)
        assert result == pytest.approx(expected, abs=1e-4)


class TestSchwefel:
    @pytest.mark.parametrize(
        ("dims", "input_x", "expected"),
        [
            (2, np.array([420.9687, 420.9687]), pytest.approx(0, abs=1e-4)),
            (3, np.array([420.9687, 420.9687, 420.9687]), pytest.approx(0, abs=1e-4)),
            (2, np.array([-500, -500]), pytest.approx(476.78748, abs=1e-4)),
            (3, np.array([500, 500, 500]), pytest.approx(1798.7161, abs=1e-4)),
        ],
    )
    def test_schwefel(self, dims, input_x, expected):
        func = Schwefel(dims=dims)
        result = func(input_x, saver=False)
        assert result == pytest.approx(expected, abs=1e-4)

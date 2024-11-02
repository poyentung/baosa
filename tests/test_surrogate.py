from typing import Type

from numpy.typing import NDArray
import numpy as np
import pytest
import tensorflow as tf

from balsa.surrogate import (
    DivideLayer,
    Surrogate,
    AckleySurrogate,
    RastriginSurrogate,
    RosenbrockSurrogate,
    GriewankSurrogate,
    SchwefelSurrogate,
    MichalewiczSurrogate,
    DefaultSurrogate,
)


@pytest.fixture
def sample_data() -> tuple[NDArray, NDArray]:
    """Generate sample data for testing."""
    np.random.seed(42)
    input_dim = 10
    num_samples = 100
    X = np.random.rand(num_samples, input_dim)
    y = np.random.rand(num_samples, 1)
    return X, y


@pytest.mark.parametrize(
    "SurrogateClass",
    [
        AckleySurrogate,
        RastriginSurrogate,
        RosenbrockSurrogate,
        GriewankSurrogate,
        SchwefelSurrogate,
        MichalewiczSurrogate,
        DefaultSurrogate,
    ],
)
def test_surrogate_model(
        SurrogateClass: Type[Surrogate],
        sample_data: tuple[NDArray, NDArray],
) -> None:
    """Test each surrogate model class."""
    X, y = sample_data
    input_dimension = X.shape[1]

    surrogate = SurrogateClass(
        input_dimension=input_dimension,
        max_epochs=1,
        batch_size=32,
        validation_split=0.2,
    )

    model = surrogate.train_and_evaluate(X, y)

    assert model is not None
    assert isinstance(model, tf.keras.Model)

    # Test prediction
    X_test: NDArray = np.random.rand(10, input_dimension)
    X_test_reshaped: NDArray = X_test.reshape(len(X_test), input_dimension, 1)
    predictions: NDArray = model.predict(X_test_reshaped, verbose=False)

    assert predictions.shape == (10, 1)
    assert np.all(np.isfinite(predictions))


def test_dividelayer() -> None:
    """Test the DivideLayer class."""
    from balsa.surrogate import DivideLayer

    input_shape = (5, 1)
    const = 10.0
    layer = DivideLayer(input_shape, const)

    inputs: tf.Tensor = tf.random.uniform((3, 5, 1))
    outputs: tf.Tensor = layer(inputs)

    assert outputs.shape == inputs.shape
    assert np.allclose(outputs.numpy(), inputs.numpy() / const, rtol=1e-5)


@pytest.mark.parametrize(
    "SurrogateClass, expected_loss",
    [
        (AckleySurrogate, "mean_squared_error"),
        (RastriginSurrogate, "mean_absolute_percentage_error"),
        (RosenbrockSurrogate, "mean_squared_error"),
        (GriewankSurrogate, "mean_squared_error"),
        (SchwefelSurrogate, "mean_absolute_percentage_error"),
        (MichalewiczSurrogate, "mean_squared_error"),
        (DefaultSurrogate, "mean_squared_error"),
    ],
)
def test_surrogate_loss_function(
        SurrogateClass: Type[Surrogate],
        expected_loss: str,
        sample_data: tuple[NDArray, NDArray],
) -> None:
    """Test the loss function for each surrogate model class."""
    X, _ = sample_data
    input_dimension = X.shape[1]

    surrogate = SurrogateClass(input_dimension=input_dimension)
    model = surrogate.build_model()

    assert model.loss == expected_loss


@pytest.mark.parametrize(
    "SurrogateClass, expected_layers",
    [
        (
                AckleySurrogate,
                (
                        tf.keras.layers.Conv1D,
                        tf.keras.layers.MaxPooling1D,
                        tf.keras.layers.Dropout,
                ),
        ),
        (
                RastriginSurrogate,
                (tf.keras.layers.Conv1D, tf.keras.layers.LayerNormalization),
        ),
        (
                RosenbrockSurrogate,
                (
                        tf.keras.layers.Conv1D,
                        tf.keras.layers.MaxPooling1D,
                        tf.keras.layers.Dropout,
                ),
        ),
        (
                GriewankSurrogate,
                (
                        tf.keras.layers.Conv1D,
                        tf.keras.layers.MaxPooling1D,
                        tf.keras.layers.Dropout,
                        DivideLayer,
                ),
        ),
        (
                SchwefelSurrogate,
                (tf.keras.layers.Conv1D, tf.keras.layers.MaxPooling1D, DivideLayer),
        ),
        (
                MichalewiczSurrogate,
                (tf.keras.layers.Conv1D, tf.keras.layers.MaxPooling1D, DivideLayer),
        ),
        (
                DefaultSurrogate,
                (
                        tf.keras.layers.Conv1D,
                        tf.keras.layers.MaxPooling1D,
                        tf.keras.layers.Dropout,
                ),
        ),
    ],
)
def test_surrogate_model_architecture(
        SurrogateClass: Type[Surrogate],
        expected_layers: tuple[Type[tf.keras.layers.Layer], ...],
        sample_data: tuple[NDArray, NDArray],
) -> None:
    """Test the architecture of each surrogate model class."""
    X, _ = sample_data
    input_dimension = X.shape[1]

    surrogate = SurrogateClass(input_dimension=input_dimension)
    model = surrogate.build_model()

    for layer_type in expected_layers:
        assert any(isinstance(layer, layer_type) for layer in model.layers)

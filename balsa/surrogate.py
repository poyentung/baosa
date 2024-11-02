from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    LayerNormalization,
)

logger = logging.getLogger(__name__)


@dataclass
class Surrogate(ABC):
    """Abstract base class for surrogate models.

    Attributes:
        input_dimension: The dimension of the input features.
        learning_rate: The learning rate for the optimizer.
        batch_size: The batch size for training.
        max_epochs: The maximum number of epochs for training.
        early_stopping_patience: The number of epochs with no improvement after which training will be stopped.
        validation_split: The proportion of the dataset to use for validation.
        random_seed: The random seed for reproducibility.
        loss_function: The loss function to use for training.
        model: The Keras model instance.
    """

    input_dimension: int
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 500
    early_stopping_patience: int = 30
    validation_split: float = 0.2
    random_seed: int = 42
    loss_function: str = "mean_squared_error"
    model: Optional[keras.Model] = None

    @abstractmethod
    def build_model(self) -> Sequential:
        """Build and return the Keras model."""
        pass

    def train_and_evaluate(
        self, features: NDArray, targets: NDArray
    ) -> keras.Model:
        """Train the model and evaluate its performance.

        Args:
            features: Input features for training.
            targets: Target values for training.

        Returns:
            The trained Keras model.
        """
        X_train, X_test, y_train, y_test = self._split_data(features, targets)
        X_train_reshaped, X_test_reshaped = self._reshape_data(X_train, X_test)

        self.model = self.build_model()
        self._train_model(X_train_reshaped, y_train, X_test_reshaped, y_test)
        self._evaluate_model(X_test_reshaped, y_test)

        return self.model

    def _split_data(
        self, features: NDArray, targets: NDArray
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Split the data into training and testing sets."""
        return train_test_split(
            features,
            targets,
            test_size=self.validation_split,
            random_state=self.random_seed,
        )

    def _reshape_data(
        self, X_train: NDArray, X_test: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Reshape the input data for the model."""
        return (
            X_train.reshape(len(X_train), self.input_dimension, 1),
            X_test.reshape(len(X_test), self.input_dimension, 1),
        )

    def _train_model(
        self,
        X_train: NDArray,
        y_train: NDArray,
        X_test: NDArray,
        y_test: NDArray,
    ) -> None:
        """Train the model with the given data."""
        callbacks = self._get_callbacks()

        self.model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.max_epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=False,
        )

        self.model = keras.models.load_model("best_model.keras")

    def _get_callbacks(self) -> list[keras.callbacks.Callback]:
        """Get the callbacks for model training."""
        return [
            ModelCheckpoint(
                "best_model.keras",
                monitor="val_loss",
                mode="min",
                verbose=False,
                save_best_only=True,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_patience,
                restore_best_weights=True,
            ),
        ]

    def _evaluate_model(self, X_test: NDArray, y_test: NDArray) -> None:
        """Evaluate the model and print performance metrics."""
        y_pred = self.model.predict(X_test, verbose=False)

        pearson_correlation, _ = stats.pearsonr(y_pred.flatten(), y_test.flatten())
        r_squared = np.round(pearson_correlation**2, 5)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mape = metrics.mean_absolute_percentage_error(y_test, y_pred)

        logger.info(f"Model performance: R² {r_squared}, MAE {mae}, MAPE {mape}")


@dataclass
class AckleySurrogate(Surrogate):
    def build_model(self) -> Sequential:
        if self.input_dimension <= 100:
            model = Sequential(
                [
                    Conv1D(
                        128,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        activation="elu",
                        input_shape=(self.input_dimension, 1),
                    ),
                    MaxPooling1D(pool_size=2, strides=1),
                    Dropout(0.2),
                    Conv1D(
                        64, kernel_size=3, strides=1, padding="same", activation="elu"
                    ),
                    MaxPooling1D(pool_size=2, strides=1),
                    Dropout(0.2),
                    Conv1D(
                        32, kernel_size=3, strides=1, padding="same", activation="elu"
                    ),
                    Conv1D(
                        16, kernel_size=3, strides=1, padding="same", activation="elu"
                    ),
                    Conv1D(
                        8, kernel_size=3, strides=1, padding="same", activation="elu"
                    ),
                    Flatten(),
                    Dense(128, activation="elu"),
                    Dense(64, activation="elu"),
                    Dense(1, activation="linear"),
                ]
            )
        else:
            model = Sequential(
                [
                    Conv1D(
                        128,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        activation="elu",
                        input_shape=(self.input_dimension, 1),
                    ),
                    MaxPooling1D(pool_size=2),
                    Dropout(0.2),
                    Conv1D(
                        64, kernel_size=3, strides=1, padding="same", activation="elu"
                    ),
                    MaxPooling1D(pool_size=2),
                    Dropout(0.2),
                    Conv1D(
                        32, kernel_size=3, strides=1, padding="same", activation="elu"
                    ),
                    MaxPooling1D(pool_size=2, strides=1),
                    Conv1D(
                        16, kernel_size=3, strides=1, padding="same", activation="elu"
                    ),
                    Conv1D(
                        8, kernel_size=3, strides=1, padding="same", activation="elu"
                    ),
                    Conv1D(
                        4, kernel_size=3, strides=1, padding="same", activation="elu"
                    ),
                    Flatten(),
                    Dense(64, activation="elu"),
                    Dense(1, activation="linear"),
                ]
            )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss_function
        )
        return model


@dataclass
class RastriginSurrogate(Surrogate):
    loss_function: str = "mean_absolute_percentage_error"

    def build_model(self) -> Sequential:
        model = Sequential(
            [
                Conv1D(
                    256,
                    kernel_size=5,
                    strides=1,
                    padding="same",
                    activation="elu",
                    input_shape=(self.input_dimension, 1),
                ),
                LayerNormalization(),
                Conv1D(128, kernel_size=5, strides=2, padding="same", activation="elu"),
                Conv1D(64, kernel_size=3, strides=2, padding="same", activation="elu"),
                Conv1D(32, kernel_size=3, strides=1, padding="same", activation="elu"),
                Conv1D(16, kernel_size=3, strides=1, padding="same", activation="elu"),
                Conv1D(8, kernel_size=3, strides=1, padding="same", activation="elu"),
                Flatten(),
                Dense(128, activation="elu"),
                Dense(64, activation="elu"),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss_function
        )
        return model


@dataclass
class RosenbrockSurrogate(Surrogate):
    def build_model(self) -> Sequential:
        model = Sequential(
            [
                Conv1D(
                    128,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    activation="elu",
                    input_shape=(self.input_dimension, 1),
                ),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Conv1D(64, kernel_size=3, strides=1, padding="same", activation="elu"),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Conv1D(32, kernel_size=3, strides=1, padding="same", activation="elu"),
                MaxPooling1D(pool_size=2, strides=1),
                Conv1D(16, kernel_size=3, strides=1, padding="same", activation="elu"),
                Conv1D(8, kernel_size=3, strides=1, padding="same", activation="elu"),
                Conv1D(4, kernel_size=3, strides=1, padding="same", activation="elu"),
                Flatten(),
                Dense(64, activation="elu"),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss_function
        )
        return model


@dataclass
class GriewankSurrogate(Surrogate):
    def build_model(self) -> Sequential:
        model = Sequential(
            [
                DivideLayer(input_shape=(self.input_dimension, 1), const=600),
                Conv1D(128, kernel_size=3, strides=1, padding="same", activation="elu"),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Conv1D(64, kernel_size=3, strides=1, padding="same", activation="elu"),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Conv1D(32, kernel_size=3, strides=1, padding="same", activation="elu"),
                MaxPooling1D(pool_size=2, strides=1),
                Conv1D(16, kernel_size=3, strides=1, padding="same", activation="elu"),
                Conv1D(8, kernel_size=3, strides=1, padding="same", activation="elu"),
                Conv1D(4, kernel_size=3, strides=1, padding="same", activation="elu"),
                Flatten(),
                Dense(64, activation="elu"),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error"
        )
        return model


@dataclass
class SchwefelSurrogate(Surrogate):
    loss_function: str = "mean_absolute_percentage_error"

    def build_model(self) -> Sequential:
        model = Sequential(
            [
                DivideLayer(input_shape=(self.input_dimension, 1), const=1000),
                Conv1D(256, kernel_size=5, padding="same", activation="elu"),
                Conv1D(128, kernel_size=5, padding="same", activation="elu"),
                MaxPooling1D(pool_size=2),
                Conv1D(64, kernel_size=5, padding="same", activation="elu"),
                Conv1D(32, kernel_size=5, padding="same", activation="elu"),
                MaxPooling1D(pool_size=2),
                Conv1D(16, kernel_size=5, padding="same", activation="elu"),
                Conv1D(8, kernel_size=5, padding="same", activation="elu"),
                Conv1D(4, kernel_size=5, padding="same", activation="elu"),
                Flatten(),
                Dense(128, activation="elu"),
                Dense(64, activation="elu"),
                Dense(32, activation="elu"),
                Dense(16, activation="elu"),
                Dense(8, activation="elu"),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss_function
        )
        return model


@dataclass
class MichalewiczSurrogate(Surrogate):
    def build_model(self) -> Sequential:
        model = Sequential(
            [
                DivideLayer(input_shape=(self.input_dimension, 1), const=np.pi),
                Conv1D(128, kernel_size=3, strides=1, padding="same", activation="elu"),
                MaxPooling1D(pool_size=2),
                Conv1D(64, kernel_size=3, strides=1, padding="same", activation="elu"),
                MaxPooling1D(pool_size=2),
                Conv1D(32, kernel_size=3, strides=1, padding="same", activation="elu"),
                MaxPooling1D(pool_size=2, strides=1),
                Conv1D(16, kernel_size=3, strides=1, padding="same", activation="elu"),
                Conv1D(8, kernel_size=3, strides=1, padding="same", activation="elu"),
                Flatten(),
                Dense(64, activation="elu"),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss_function
        )
        return model


@dataclass
class DefaultSurrogate(Surrogate):
    def build_model(self) -> Sequential:
        model = Sequential(
            [
                Conv1D(
                    128,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    activation="relu",
                    input_shape=(self.input_dimension, 1),
                ),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Conv1D(64, kernel_size=3, strides=1, padding="same", activation="relu"),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Conv1D(32, kernel_size=3, strides=1, padding="same", activation="relu"),
                MaxPooling1D(pool_size=2, strides=1),
                Conv1D(16, kernel_size=3, strides=1, padding="same", activation="relu"),
                Conv1D(8, kernel_size=3, strides=1, padding="same", activation="relu"),
                Conv1D(4, kernel_size=3, strides=1, padding="same", activation="relu"),
                Flatten(),
                Dense(64, activation="relu"),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss_function
        )
        return model


class DivideLayer(keras.layers.Layer):
    def __init__(self, input_shape: tuple[int, int], const: float, **kwargs):
        super().__init__(**kwargs)
        self.const = const
        self.w = self.add_weight(initializer="ones", shape=input_shape, trainable=False)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return keras.ops.divide(inputs, self.w) / self.const

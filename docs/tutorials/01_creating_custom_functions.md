# Adding Custom Functions to BALSA - Tutorial

This tutorial explains how to add your custom optimisation function to the BALSA benchmark suite.

## Table of Contents
- [1. Creating Your Custom Function](#1-creating-your-custom-function)
- [2. Registering Your Function](#2-registering-your-function)
- [3. Usage Example](#3-usage-example)
- [4. Optional: Custom Surrogate Model](#4-optional-custom-surrogate-model)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## 1. Creating Your Custom Function

Create a new file `my_custom_function.py` in the `balsa/vlab/` directory:

```python
from dataclasses import field
from typing import Any, override
import numpy as np

from balsa.obj_func import ObjectiveFunction

class MyCustomFunction(ObjectiveFunction):
    """Custom objective function for optimisation."""
    
    name: str = "my_custom"
    dims: int = 5  # Set your function's dimension
    turn: float = 0.1  # Parameter discretization step
    func_args: dict[str, Any] = field(
        default_factory=lambda: {
            "param1": None,
            "param2": None,
            "lb": [-5.0] * 5,  # Lower bounds
            "ub": [5.0] * 5,   # Upper bounds
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.dims > 0
        # Define parameter bounds
        self.lb = np.array(self.func_args["lb"])
        self.ub = np.array(self.func_args["ub"])
        
        # Initialize custom parameters
        self.param1 = self.func_args.get("param1")
        self.param2 = self.func_args.get("param2")

    @override
    def _scaled(self, y: float) -> float:
        return y  # Modify for minimization/maximization

    @override
    def __call__(self, x: np.ndarray, saver: bool = True, return_scaled=False) -> float:
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1

        # Your optimization function here
        y = float(np.sum(x**2))  # Example function
        
        self.tracker.track(y, x, saver)
        return y if not return_scaled else self._scaled(y)
```

## 2. Registering Your Function

Add your function to `balsa/active_learning.py`:

```python
@dataclass
class OptimisationRegistry:
    FUNCTIONS = {
        # Existing functions
        "my_custom": MyCustomFunction,  # Add your function
    }

    @classmethod
    def register_special_functions(cls, func_name: str) -> None:
        if func_name == "my_custom":
            from .vlab.my_custom_function import MyCustomFunction
            cls.FUNCTIONS["my_custom"] = MyCustomFunction
```

## 3. Usage Example

Here's how to use your custom function:

```python
from balsa.active_learning import OptimisationConfig, ActiveLearningPipeline

# Configuration
config = OptimisationConfig(
    dims=5,                          # Input dimensions
    search_method="turbo",           # Optimization algorithm
    obj_func_name="my_custom",       # Your function name
    num_acquisitions=50,             # Optimization iterations
    num_samples_per_acquisition=1,   # Samples per iteration
    surrogate="default_surrogate",   # Surrogate model
    num_init_samples=30,             # Initial samples
    func_args={                      # Custom parameters
        "param1": "value1",
        "param2": "value2"
    }
)

# Run optimization
pipeline = ActiveLearningPipeline(config)
pipeline.run()
```

## 4. Optional: Custom Surrogate Model

If needed, create a custom surrogate model in `balsa/surrogate.py`:

```python
@dataclass
class MyCustomSurrogate(Surrogate):
    def build_model(self) -> Sequential:
        model = Sequential([
            Conv1D(128, kernel_size=3, padding="same", activation="elu",
                  input_shape=(self.input_dimension, 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, padding="same", activation="elu"),
            Flatten(),
            Dense(64, activation="elu"),
            Dense(1, activation="linear"),
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=self.loss_function
        )
        return model
```

Register it in `OptimisationRegistry`:

```python
SURROGATES = {
    "my_custom": MyCustomSurrogate,
    # ... existing surrogates ...
}
```

## Best Practices

1. **Function Implementation**
   - Inherit from `ObjectiveFunction`
   - Define appropriate bounds
   - Implement proper scaling
   - Use type hints
   - Add comprehensive docstrings

2. **Parameter Handling**
   - Validate inputs in `__post_init__`
   - Use appropriate data types
   - Handle edge cases
   - Document parameter requirements

3. **Tracking and Logging**
   - Use `tracker.track()` for progress
   - Log important events
   - Save intermediate results
   - Monitor performance

---

For more examples and detailed documentation, visit the [BALSA documentation](https://github.com/poyentung/balsa).
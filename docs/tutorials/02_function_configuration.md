# Configuring and Running Custom Functions in BALSA

This tutorial explains how to create configuration files and run scripts for your custom functions in BALSA.

## Table of Contents
- [1. Creating Configuration File](#1-creating-configuration-file)
- [2. Creating Run Script](#2-creating-run-script)
- [3. Running Your Function](#3-running-your-function)
- [4. Example Implementation](#4-example-implementation)

## 1. Creating Configuration File

Create a YAML configuration file `scripts/conf/run_custom.yaml`:

```yaml
# Basic configuration
dims: 5                           # Number of dimensions
search_method: turbo              # Choice of optimisation algorithm
obj_func_name: my_custom         # Your custom function name
num_acquisitions: 100            # Number of optimisation iterations
num_samples_per_acquisition: 1    # Samples per iteration
surrogate: default               # Surrogate model type
num_init_samples: 20             # Initial random samples
mode: fast                       # Optimisation mode

# Custom function parameters
func_args:
  param1: value1                 # Your custom parameters
  param2: value2
  lb: [-5.0, -5.0, -5.0, -5.0, -5.0]  # Lower bounds for each dimension
  ub: [5.0, 5.0, 5.0, 5.0, 5.0]       # Upper bounds for each dimension

# Optimisation algorithm parameters
search_method_args:
  turbo:
    n_trust_regions: 5           # Number of trust regions
    n_repeat: 1                  # Number of repeats
    batch_size: 1                # Batch size
    verbose: True                # Verbosity
    use_ard: True                # Automatic Relevance Determination
    max_cholesky_size: 2000      # Cholesky decomposition limit
    n_training_steps: 50         # Training steps
    min_cuda: 1024               # Minimum size for CUDA
    device: cpu                  # Device type
    dtype: float32               # Data type

  lamcts:
    Cp: 1                        # Exploration constant
    leaf_size: 10                # Tree leaf size
    kernel_type: linear          # Kernel type
    gamma_type: auto             # Gamma parameter

  da:
    initial_temp: 0.05           # Initial temperature

  doo:
    explr_p: 0.01               # Exploration parameter

  voo:
    explr_p: 0.001              # Exploration parameter
    sampling_mode: centered_uniform
    switch_counter: 100
```

## 2. Creating Run Script

Create a Python script `scripts/run_custom.py`:

```python
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

import warnings
import hydra
from omegaconf import DictConfig
from balsa.active_learning import ActiveLearningPipeline, OptimisationConfig

# Filter warnings if needed
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Do not pass an `input_shape`/`input_dim` argument to a layer.*",
)

@hydra.main(version_base="1.3", config_path="conf", config_name="run_custom.yaml")
def main(config: DictConfig):
    """Main function to run optimisation.
    
    Args:
        config: Configuration from Hydra
    """
    # Create optimisation configuration
    optim_config = OptimisationConfig(**config)
    
    # Initialise and run pipeline
    pipeline = ActiveLearningPipeline(optim_config)
    pipeline.run()

if __name__ == "__main__":
    main()
```

## 3. Running Your Function

Execute your optimisation from the command line:

```bash
# Run with default configuration
python scripts/run_custom.py

# Override configuration parameters
python scripts/run_custom.py dims=10 num_acquisitions=200

# Use a different search method
python scripts/run_custom.py search_method=lamcts
```

## 4. Example Implementation

Here's a complete example with a simple quadratic function:

```yaml
# scripts/conf/run_quadratic.yaml
dims: 3
search_method: turbo
obj_func_name: quadratic
num_acquisitions: 50
num_samples_per_acquisition: 1
surrogate: default
num_init_samples: 20
mode: fast

func_args:
  scale_factor: 2.0
  offset: 1.0
  lb: [-10.0, -10.0, -10.0]
  ub: [10.0, 10.0, 10.0]

search_method_args:
  turbo:
    n_trust_regions: 3
    batch_size: 1
    verbose: True
```

```python
# balsa/vlab/quadratic.py
from dataclasses import field
from typing import Any, override
import numpy as np

from balsa.obj_func import ObjectiveFunction

class QuadraticFunction(ObjectiveFunction):
    name: str = "quadratic"
    dims: int = 3
    turn: float = 0.1
    func_args: dict[str, Any] = field(
        default_factory=lambda: {
            "scale_factor": 1.0,
            "offset": 0.0,
            "lb": None,
            "ub": None
        }
    )

    def __post_init__(self) -> None:
        self.scale = self.func_args.get("scale_factor", 1.0)
        self.offset = self.func_args.get("offset", 0.0)
        self.lb = np.array(self.func_args.get("lb", [-10.0] * self.dims))
        self.ub = np.array(self.func_args.get("ub", [10.0] * self.dims))

    @override
    def _scaled(self, y: float) -> float:
        return -y  # Convert minimization to maximization

    @override
    def __call__(self, x: np.ndarray, saver: bool = True, return_scaled=False) -> float:
        y = float(self.scale * np.sum(x**2) + self.offset)
        self.tracker.track(y, x, saver)
        return y if not return_scaled else self._scaled(y)
```

## Best Practices

1. **Configuration Organization**
   - Group related parameters
   - Use meaningful parameter names
   - Document parameter purposes
   - Set reasonable defaults

2. **Run Script Structure**
   - Handle paths properly
   - Add error handling
   - Include logging
   - Support parameter overrides

3. **Directory Structure**
```
balsa/
├── scripts/
│   ├── conf/
│   │   ├── run_custom.yaml
│   │   └── run_quadratic.yaml
│   ├── run_custom.py
│   └── run_quadratic.py
├── balsa/
│   └── vlab/
│       └── quadratic.py
└── README.md
```

## Troubleshooting

**Configuration Issues**
   - Verify YAML syntax
   - Check parameter types
   - Ensure paths are correct
   - Validate bounds

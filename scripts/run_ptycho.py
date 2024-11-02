import os
import sys

# Add the project root directory to Python path BEFORE imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
import hydra
from omegaconf import DictConfig
from balsa.active_learning import ActiveLearningPipeline, OptimisationConfig

# Filter out the specific UserWarning from Keras
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Do not pass an `input_shape`/`input_dim` argument to a layer.*",
)

@hydra.main(version_base="1.3", config_path="conf", config_name="run_ptycho.yaml")
def main(config: DictConfig):
    optim_config = OptimisationConfig(**config)
    active_learning_loop = ActiveLearningPipeline(optim_config)
    active_learning_loop.run()

if __name__ == "__main__":
    main()

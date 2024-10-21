from .base import BaseOptimization
from lamcts import MCTS

class LaMCTS(BaseOptimization):
    def __init__(self, **args):
        super().__init__(**args)

    def exact_f(self, x):
        try:
            return self.f(x)[0]
        except:
            return self.f(x)
    
    def run(self, 
            num_samples,
            num_init_samples:int=200,
            Cp:float= 1,
            leaf_size:int=10,
            kernel_type:str="linear",
            gamma_type:str= "auto"):
        
        agent = MCTS(
                lb = self.f.lb,              # the lower bound of each problem dimensions
                ub = self.f.ub,              # the upper bound of each problem dimensions
                dims = self.dims,            # the problem dimensions
                ninits = num_init_samples,   # the number of random samples used in initializations 
                func = self.exact_f,         # function object to be optimized
                Cp = Cp,                     # Cp for MCTS
                leaf_size = leaf_size,       # Tree leaf size
                kernel_type = kernel_type,   # SVM configruation
                gamma_type = gamma_type      # SVM configruation
            )

        agent.search(iterations = num_samples)
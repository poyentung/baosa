import numpy as np
from bayes_opt import BayesianOptimization
from .base import BaseOptimization

class BO(BaseOptimization):
    def __init__(self, **args):
        super().__init__(**args)

    def interface(self,
                  semiangle_cutoff, 
                  energy, 
                  num_iter, 
                  step_size, 
                  num_slices, 
                  slice_thicknesses, 
                  defocus, C12, phi12, C30, C21, phi21, C23, phi23):
        x = np.array([semiangle_cutoff, 
                      energy, 
                      num_iter, 
                      step_size, 
                      num_slices, 
                      slice_thicknesses, 
                      defocus, C12, phi12, C30, C21, phi21, C23, phi23])
        return self.exact_f(x)

    def exact_f(self, x):
        try:
            return self.min_max_conversion(self.f(x)[0])
        except:
            return self.min_max_conversion(self.f(x))
        
    def min_max_conversion(self, y):
        return 1/y
    
    def run(self, 
            num_samples,
            num_init_samples:int=200
            ):
        
        optimizer = BayesianOptimization(
            f=self.interface,
            pbounds={p:(lb ,ub) for p , lb ,ub in zip(self.f.param_names, self.f.lb, self.f.ub)},
            random_state=1,
        )

        optimizer.maximize(init_points=num_init_samples, n_iter=num_samples)
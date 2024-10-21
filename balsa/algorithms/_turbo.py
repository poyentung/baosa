from .base import BaseOptimization
from turbo.turbo_1 import Turbo1
from turbo.turbo_m import TurboM

class TuRBO(BaseOptimization):
    def __init__(self, **args):
        super().__init__(**args)

    def exact_f(self, x):
        try:
            return self.f(x)[0]
        except:
            return self.f(x)
    
    def run(self, 
            num_samples,                  # Maximum number of evaluations
            num_init_samples:int=200,     # Number of initial bounds from an Latin hypercube design
            n_trust_regions:int=5,        # Number trust regions: TuRBO-1 will be used if set to 1, otherwise TuRBOM will be used
            n_repeat:int=1,               # Number repeat time for the same condition
            batch_size:int=1,             # How large batch size TuRBO uses
            verbose=True,                 # Print information from each batch
            use_ard=True,                 # Set to true if you want to use ARD for the GP kernel 
            max_cholesky_size:int=2000,   # When we switch from Cholesky to Lanczos
            n_training_steps:int=50,      # Number of steps of ADAM to learn the hypers
            min_cuda:int=1024,            # Run on the CPU for small datasets
            device='cpu',                 # "cpu" or "cuda"
            dtype:str='float32'           # float64 or float32
            ):
        
        for i in range(n_repeat):
            if n_trust_regions == 1:
                agent = Turbo1(
                            f=self.exact_f,  
                            lb=self.f.lb,  
                            ub=self.f.ub,  
                            n_init=num_init_samples,  
                            max_evals=num_samples,  
                            batch_size=batch_size,  
                            verbose=verbose,
                            use_ard=use_ard,
                            max_cholesky_size=max_cholesky_size,
                            n_training_steps=n_training_steps,
                            min_cuda=min_cuda,
                            device=device,
                            dtype=dtype,
                        )
            else:
                agent = TurboM(
                            f=self.exact_f,  
                            lb=self.f.lb,  
                            ub=self.f.ub,  
                            n_init=num_init_samples // n_trust_regions,
                            max_evals=num_samples,
                            n_trust_regions=n_trust_regions, 
                            batch_size=batch_size,  
                            verbose=verbose,
                            use_ard=use_ard,
                            max_cholesky_size=max_cholesky_size,
                            n_training_steps=n_training_steps,
                            min_cuda=min_cuda,
                            device=device,
                            dtype=dtype,
                        )

            agent.optimize()
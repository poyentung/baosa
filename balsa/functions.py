import os
import numpy as np


class Function:
    def __init__(self, dims=3, turn=0.1, name='none', iters=None, func_args={}):
        self.dims    = dims
        self.name    = name
        self.lb      = None
        self.ub      = None
        self.counter = 0
        self.tracker = tracker(name+str(dims))
        self.iters   = iters
        self.turn    = turn
        self.func_args= func_args

    def __call__(self, x, saver=True):
        return NotImplementedError

class Ackley(Function):
    def __init__(self, **args):
        super().__init__(**args)
        self.lb      = -5 * np.ones(self.dims)
        self.ub      =  5 * np.ones(self.dims)

    def __call__(self, x, saver=True):
        x = x.reshape(self.dims)
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        result = (-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e)
        self.tracker.track(result, x, saver)
        
        return result, 100/(result+0.01)
    

class Rastrigin(Function):
    def __init__(self, **args):
        super().__init__(**args)
        self.lb      = -5 * np.ones(self.dims)
        self.ub      =  5 * np.ones(self.dims)

    def __call__(self, x, A=10, saver=True):
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        n = len(x)
        sum = np.sum(x**2 - A * np.cos(2 * np.pi * x))
        result = A*n + sum

        self.tracker.track(result, x, saver)
        return result, -result    


class Rosenbrock(Function):
    def __init__(self, **args):
        super().__init__(**args)
        self.lb      = -5 * np.ones(self.dims)
        self.ub      =  5 * np.ones(self.dims)

    def __call__(self, x, saver=True):
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        result = np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
        
        self.tracker.track(result, x, saver)
        return result, 100/(result/(self.dims*100)+0.01)    


class Griewank(Function):
    def __init__(self,**args):
        super().__init__(**args)
        self.lb      = -600 * np.ones(self.dims)
        self.ub      =  600 * np.ones(self.dims)

    def __call__(self, x, saver=True):
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        sum_term = np.sum(x ** 2)
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        result = 1 + sum_term / 4000 - prod_term

        self.tracker.track(result, x, saver)
        return result, 10/(result/(self.dims)+0.001)


class Michalewicz(Function):
    def __init__(self, **args):
        super().__init__(d**args)
        self.lb      = np.zeros(self.dims)
        self.ub      = np.pi * np.ones(self.dims)

    def __call__(self, x, m=10, saver=True):
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        d = len(x)
        result = 0
        for i in range(d):
            result += np.sin(x[i]) * np.sin((i + 1) * x[i]**2 / np.pi)**(2 * m)

        self.tracker.track(-result, x, saver)
        return -result, result


class Schwefel(Function):
    def __init__(self, **args):
        super().__init__(**args)
        self.lb      = -500 * np.ones(self.dims)
        self.ub      =  500 * np.ones(self.dims)

    def __call__(self, x, saver=True):
        x = np.array(x / self.turn).round(0) * self.turn
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        dimension = len(x)
        sum_part = np.sum(-x * np.sin(np.sqrt(np.abs(x))))
        if np.all(np.array(x) == 421, axis = 0):
            return 0, 10000
        result = 418.9829 * dimension + sum_part

        self.tracker.track(result, x, saver)
        return result, -result/100
    

class tracker:
    def __init__(self, foldername):
        self.counter   = 0
        self.results   = []
        self.x         = []
        self.curt_best = float("inf")
        self.curt_best_x = None
        self.foldername = foldername
        try:
            os.mkdir(foldername)
        except OSError:
            print ("Creation of the directory %s failed" % foldername)
        else:
            print ("Successfully created the directory %s " % foldername)
        
    def dump_trace(self):
        np.save(self.foldername +'/result.npy',np.array(self.results),allow_pickle=True)
            
    def track(self, result, x = None, saver = False):
        self.counter += 1
        if result < self.curt_best:
            self.curt_best = result
            self.curt_best_x = x
        print("")
        print("="*10)
        print("#samples:", self.counter, "total samples:", len(self.results)+1)
        print("="*10)
        print("current best f(x):", self.curt_best)
        print("current best x:", np.around(self.curt_best_x, decimals=4))
        self.results.append(self.curt_best)
        self.x.append(x)
        if saver == True:
            self.dump_trace()
        if self.counter % 20 == 0:
            self.dump_trace()
        if round(self.curt_best,5) == 0:
            self.dump_trace()
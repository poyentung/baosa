import random
import cma
import numpy as np
import nevergrad as ng
from scipy.optimize import dual_annealing, differential_evolution
from .base import BaseOptimization


class DualAnnealing(BaseOptimization):
    def __init__(self, **args):
        super().__init__(**args)
    
    def single_rollout(self, X, x_current, rollout_round, top_n:int=16, top_n2:int=4, method_args=dict(initial_temp=0.05)):
        if self.mode == 'fast':
            ret = dual_annealing(self.predict, bounds=self.bounds, x0=x_current, maxfun=rollout_round, **method_args)
        elif self.mode == 'origin':
            ret = dual_annealing(self.predict, bounds=self.bounds, x0=x_current)
        self.all_proposed.append(np.round(ret.x,int(-np.log10(self.f.turn))))
        return self.get_top_X(X,top_n, top_n2) 



class DifferentialEvolution(BaseOptimization):
    def __init__(self, **args):
        super().__init__(**args)
    
    def single_rollout(self, X, x_current, rollout_round, top_n:int=16, top_n2:int=4, method_args={}):
        if self.mode == 'fast':
            popsize = int(max(100 / self.f.dims, 1))
            ret = differential_evolution(self.predict, bounds=self.bounds, x0=x_current, maxiter=1, popsize=popsize, **method_args)
        elif self.mode == 'origin':
            ret = differential_evolution(self.predict, bounds=self.bounds, x0=x_current)
        self.all_proposed.append(np.round(ret.x, int(-np.log10(self.f.turn))))
        return self.get_top_X(X, top_n, top_n2)



class CMAES(BaseOptimization):
    def __init__(self, **args):
        super().__init__(**args)
        
    def single_rollout(self,X, x_current, rollout_round, top_n:int=16, top_n2:int=4, method_args={}):
        if self.mode == 'fast':
            options = {'maxiter':int(rollout_round/10),'bounds':[self.f.lb[0], self.f.ub[0]]}
            es =cma.fmin(self.predict, x_current, 0.5, options, **method_args)
        elif self.mode == 'origin':
            options = {'bounds':[self.f.lb[0], self.f.ub[0]]}
            es =cma.fmin(self.predict, x_current, (self.f.ub[0]-self.f.lb[0])/4, options, **method_args)
        return self.get_top_X(X, top_n, top_n2)



class Shiwa(BaseOptimization):
    def __init__(self, **args):
        super().__init__(**args)
    
    def single_rollout(self,X, x_current, rollout_round, top_n=16, top_n2=4, method_args={}):
        param = ng.p.Array(init=x_current).set_bounds(self.f.lb, self.f.ub)
        optimizer = ng.optimization.optimizerlib.Shiwa(parametrization=param, budget=rollout_round, **method_args)
        recommendation = optimizer.minimize(self.predict)
        return self.get_top_X(X, top_n, top_n2)



class MCMC(BaseOptimization):
    def __init__(self, **args):
        super().__init__(**args)
        
    def choose(self, board):
        "Choose the best successor of node. (Choose a move in the game)"
        turn = self.turn
        aaa = np.arange(self.f.lb[0], self.f.ub[0] + self.f.turn, self.f.turn).round(5)
        index = np.random.randint(0, self.dims)
        tup = np.array(board)
        flip = random.randint(0,5)
        if   flip ==0:
          tup[index] += turn
        elif flip ==1:
            tup[index] -= turn
        elif flip ==2:
          for i in range(int(self.dims/5)):
            index_2 = random.randint(0, len(tup)-1)
            tup[index_2] = np.random.choice(aaa)
        elif flip ==3:
          for i in range(int(self.dims/10)):
            index_2 = random.randint(0, len(tup)-1)
            tup[index_2] = np.random.choice(aaa)
        elif flip ==4:
            tup[index] = np.random.choice(aaa)
        elif flip ==5:
            tup[index] = np.random.choice(aaa)
        tup[index] = round(tup[index],5)
        ind1 = np.where(tup>self.f.ub[0])[0]
        if len(ind1) > 0:
            tup[ind1] = self.f.ub[0]
        ind1 = np.where(tup<self.f.lb[0])[0]
        if len(ind1) > 0:
            tup[ind1] = self.f.lb[0]
        value = self.model.predict(np.array(tup).reshape(1,-1,1))
        value = np.array(value).reshape(1)
        return tup, value
    
    def single_rollout(self,X, initial_X, rollout_round, top_n=20, method_args={}):
        values = self.model.predict(np.array(initial_X).reshape(1,-1,1))
        x_current=np.array(initial_X)
        cu_Y=np.array(values).reshape(-1)

        boards=[]
        for i in range(rollout_round): 
            board,temp_Y=self.choose(x_current)
            boards.append(board)
            if temp_Y>cu_Y*1:
                  x_current = np.array(board)
                  cu_Y=np.array(temp_Y)

        new_x = self.data_process(X,boards)
        new_pred = self.model.predict(np.array(new_x).reshape(len(new_x),-1,1))
        new_pred = np.array(new_pred).reshape(len(new_x))
        
        if len(new_x)>=top_n:
            ind = np.argsort(new_pred)[-top_n:]
            top_X =  new_x[ind]
        else:
            aaa = np.arange(self.f.lb[0], self.f.ub[0] + self.f.turn, self.f.turn).round(5)
            random_X = np.random.choice(aaa,size=(top_n - len(new_x), self.f.dims))
            top_X=np.concatenate((new_x, random_X),axis=0)
        return top_X 
    
import numpy as np

class BaseOptimization:
    def __init__(self, f=None, dims=20, model=None, name=None, dfo_method=None):
        self.f = f
        self.turn = f.turn
        self.dims = dims
        self.model = model
        self.name = name
        self.mode = None
        self.all_proposed=[]
        self.dfo_method = dfo_method
        self.bounds = [(float(self.f.lb[idx]), float(self.f.ub[idx])) for idx in range(0, len(self.f.lb))]

    def data_process(self, X, boards):
        new_x = []
        boards = np.array(boards)
        boards = np.unique(boards, axis=0)
        for i in boards:
          temp_x = np.array(i)
          same = np.all(temp_x==X, axis=1)
          has_true = any(same)
          if has_true == False:
            new_x.append(temp_x)
        new_x= np.array(new_x)
        print(f'unique number of boards: {len(new_x)}')
        return new_x
    
    def predict(self,x):
        x = np.round(x, int(-np.log10(self.f.turn)))
        self.all_proposed.append(x)
        try:
           pred = self.model.predict(np.array(x).reshape(len(x),self.f.dims,1))
           pred = np.array(pred).reshape(len(x))
        except:
           pred = self.model.predict(np.array(x).reshape(1,self.f.dims,1))
           pred = np.array(pred).reshape(1)
        if self.name == 'ackley':
            pred_fun=100/pred-0.01
        elif self.name == 'rastrigin':
            pred_fun = -1 * pred
        elif self.name == 'rosenbrock':
            pred_fun=(100/pred-0.01)*self.f.dims*100
        elif self.name == 'levy':
            pred_fun=(100/pred-0.01)*self.f.dims
        elif self.name == 'schwefel':
            pred_fun=(100/pred-0.01)*self.f.dims
        elif self.name == 'michalewicz':
            pred_fun = 100/pred
        elif self.name == 'griewank':
            pred_fun=(100/pred-0.01)*self.f.dims*0.1
        else:
            pred_fun=100/pred
        return float(pred_fun)
        
    def get_top_X(self, X, top_n, top_n2):
        new_x = self.data_process(X, self.all_proposed)
        new_pred = self.model.predict(np.array(new_x).reshape(len(new_x),-1,1))
        new_pred = np.array(new_pred).reshape(len(new_x))
        
        try:
            ind = np.argsort(new_pred)
            new_x2 = new_x[ind[:-top_n]]
            ind2=np.random.choice(len(new_x2),size=top_n2,replace=False)
            top_X=np.concatenate((new_x[ind[-top_n:]], new_x2[ind2]),axis=0)
        except:
            dummy = np.arange(self.f.lb[0], self.f.ub[0] + self.f.turn, self.f.turn).round(5)
            random_X = np.random.choice(dummy, size=(top_n - len(new_x), self.f.dims))
            top_X = np.concatenate((new_x, random_X),axis=0)
        return top_X

    def single_rollout(self):
        return NotImplementedError
    
    def rollout(self, X, y, rollout_round, method_args={}):
        if self.name == 'rastrigin' or self.name == 'ackley':
            index_max = np.argmax(y)
            initial_X = X[index_max,:]
            top_X = self.single_rollout(X, initial_X, rollout_round, method_args=method_args)
            
        else:
            #### unique initial points
            ind = np.argsort(y)
            x_current_top = X[ind[-3:]]
            x_current_top = np.unique(x_current_top, axis = 0)
            i = 0
            while len(x_current_top) < 3:
                x_current_top=np.concatenate((x_current_top.reshape(-1,self.f.dims),X[ind[i-4]].reshape(-1,self.f.dims)), axis = 0)
                i-=1
                x_current_top = np.unique(x_current_top, axis = 0)
            
            X_top=[]
            for i in range(3):
              initial_X = x_current_top[i]
              top_X = self.single_rollout(X, initial_X, rollout_round, top_n = 6, top_n2 = 1)
              X_top.append(top_X)
            
            top_X = np.vstack(X_top)
            top_X = top_X[-20:]
        return top_X
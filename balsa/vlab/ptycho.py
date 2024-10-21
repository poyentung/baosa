import sys
sys.path.append('../')

import py4DSTEM
import numpy as np

from vlab_bench.functions import Function

class ElectronPtychography(Function):
    def __init__(self, 
                 dims:int=14, 
                 turn:float=0.1, 
                 name:str='none', 
                 iters=None, 
                 func_args=dict(file_dir='data/MoS2_10layer_80kV_cutoff20_defocus130_nyquist60_abr_noise10000_v2.h5',
                                param_names=[],
                                lb=[],
                                ub=[],
                                )
                 ):
        
        super().__init__(dims, turn, name, iters)

        self.func_args = func_args
        self.dataset = self.read(func_args['file_dir'])
        self.param_names = func_args['param_names']
        self.lb = np.array(func_args['lb'])
        self.ub = np.array(func_args['ub'])

        # self.param_names = ['semiangle_cutoff', 'energy', 'num_iter', 'step_size', 'num_slices', 'slice_thicknesses',
        #                     'defocus', 'C12', 'phi12', 'C30', 'C21', 'phi21', 'C23', 'phi23']
        # self.lb = np.array([ 1,    1e3,   1,   0.01,    1,    0.1,   200,     0,          0,    5e4,    0,          0,    0,          0])
        # self.ub = np.array([30,  200e3,  20,   1.00,   50,   10.0,  -200,   100,  2*math.pi,   -5e4,  100,  2*math.pi,  100,  2*math.pi])
        

    def __call__(self, x, saver=True):
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        self.params = {self.param_names[i]:x[i] for i in range(self.dims)}
        print(self.params)

        ptycho = py4DSTEM.process.phase.MultislicePtychography(
                    datacube=self.dataset,
                    num_slices=int(self.params['num_slices']),
                    slice_thicknesses=float(self.params['slice_thicknesses']),
                    verbose=True,
                    energy=float(self.params['energy']),
                    semiangle_cutoff=float(self.params['semiangle_cutoff']),
                    device='cpu',
                    object_type='potential',
                    object_padding_px=(18,18),
                    polar_parameters={aberr:self.params[aberr] for aberr in self.param_names[-8:]}
        ).preprocess(
            plot_center_of_mass = False,
            plot_rotation=False,
        )

        ptycho = ptycho.reconstruct(
            reset=True,
            store_iterations=True,
            num_iter = int(self.params['num_iter']),
            step_size = float(self.params['step_size']),
        )

        self.tracker.track(ptycho.error, x, saver)
        return ptycho.error
    
    def read(self, file_dir):
        dataset = py4DSTEM.read(file_dir)
        dataset.calibration = py4DSTEM.data.calibration.Calibration()
        dataset.calibration["R_pixel_size"] = 0.3118
        dataset.calibration["Q_pixel_size"] = 0.0628
        dataset.calibration["R_pixel_units"] = "A"
        dataset.calibration["Q_pixel_units"] = "A^-1"
        dataset.calibration["QR_flip"] = False
        print(f'Dataset size: {dataset.shape}')
        return dataset
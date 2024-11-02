import sys
from dataclasses import field
from pathlib import Path
from typing import Any, override

sys.path.append("../")

import py4DSTEM
import numpy as np

from balsa.obj_func import ObjectiveFunction

DATA_PATH = Path(
    "data/MoS2_10layer_80kV_cutoff20_defocus130_nyquist60_abr_noise10000_v2.h5"
)

# Module-level constants
OBJECT_PADDING = (18, 18)
PIXEL_SIZE_R = 0.3118
PIXEL_SIZE_Q = 0.0628


class ElectronPtychography(ObjectiveFunction):
    name: str = "ptycho"
    dims: int = 14
    turn: float = 0.1
    func_args: dict[str, Any] = field(
        default_factory=lambda: {
            "file_dir": DATA_PATH,
            "param_names": [],
            "lb": [],
            "ub": [],
        }
    )

    def __post_init__(self) -> None:
        self.dataset = self.read(self.func_args["file_dir"])
        self.param_names = self.func_args["param_names"]
        self.lb = np.array(self.func_args["lb"])
        self.ub = np.array(self.func_args["ub"])

    @override
    def _scaled(self, y: float) -> float:
        return y

    @override
    def __call__(self, x: np.ndarray, saver: bool = True, return_scaled=False) -> float:
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        self.params = {self.param_names[i]: x[i] for i in range(self.dims)}
        print(self.params)

        ptycho = py4DSTEM.process.phase.MultislicePtychography(
            datacube=self.dataset,
            num_slices=int(self.params["num_slices"]),
            slice_thicknesses=float(self.params["slice_thicknesses"]),
            verbose=True,
            energy=float(self.params["energy"]),
            semiangle_cutoff=float(self.params["semiangle_cutoff"]),
            device="cpu",
            object_type="potential",
            object_padding_px=OBJECT_PADDING,
            polar_parameters={
                aberr: self.params[aberr] for aberr in self.param_names[-8:]
            },
        ).preprocess(
            plot_center_of_mass=False,
            plot_rotation=False,
        )

        ptycho = ptycho.reconstruct(
            reset=True,
            store_iterations=True,
            num_iter=int(self.params["num_iter"]),
            step_size=float(self.params["step_size"]),
        )

        self.tracker.track(ptycho.error, x, saver)
        return ptycho.error

    def read(self, file_dir):
        dataset = py4DSTEM.read(file_dir)
        dataset.calibration = py4DSTEM.data.calibration.Calibration()
        dataset.calibration["R_pixel_size"] = PIXEL_SIZE_R
        dataset.calibration["Q_pixel_size"] = PIXEL_SIZE_Q
        dataset.calibration["R_pixel_units"] = "A"
        dataset.calibration["Q_pixel_units"] = "A^-1"
        dataset.calibration["QR_flip"] = False
        print(f"Dataset size: {dataset.shape}")
        return dataset

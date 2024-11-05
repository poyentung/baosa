import sys
from dataclasses import field
from pathlib import Path
from typing import Any, override

sys.path.append("../")

import pyrosetta as pr
from pyrosetta import *
from pyrosetta.rosetta.protocols.rosetta_scripts import *
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
import re
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.shared.utils import copy_dict
# initialize pyrosetta
init('-corrections::beta_nov16 -detect_disulf false', extra_options="-run:preserve_header true")
import numpy as np

from balsa.obj_func import ObjectiveFunction

def score_interface(pdb_file):
    # load pose
    pose = pr.pose_from_pdb(pdb_file)

    # analyze interface statistics
    iam = InterfaceAnalyzerMover()
    iam.set_interface("A_B")
    scorefxn = pr.get_fa_scorefxn()
    iam.set_scorefunction(scorefxn)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    #iam.set_compute_interface_delta_hbond_unsat(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    # retrieve statistics
    interfacescore = iam.get_all_data()
    interface_sc = interfacescore.sc_value # shape complementarity
    interface_nres = iam.get_num_interface_residues() # number of interface residues
    interface_interface_hbonds = interfacescore.interface_hbonds # number of interface H-bonds
    interface_dG = iam.get_interface_dG() # interface dG
    interface_dSASA = iam.get_interface_delta_sasa() # interface dSASA (interface surface area)
    interface_packstat = iam.get_interface_packstat() # interface pack stat score
    interface_dG_SASA_ratio = interfacescore.dG_dSASA_ratio * 100
    interface_scores = {
    'interface_sc': interface_sc,
    'interface_packstat': interface_packstat,
    'interface_dG': interface_dG,
    'interface_dSASA': interface_dSASA,
    'interface_dG_SASA_ratio': interface_dG_SASA_ratio,
    'interface_nres': interface_nres,
    'interface_interface_hbonds': interface_interface_hbonds,
    }
    return interface_scores

def add_cyclic_offset(self, bug_fix=True):
    '''add cyclic offset to connect N and C term'''
    def cyclic_offset(L):
        i = np.arange(L)
        ij = np.stack([i,i+L],-1)
        offset = i[:,None] - i[None,:]
        c_offset = np.abs(ij[:,None,:,None] - ij[None,:,None,:]).min((2,3))
        if bug_fix:
            a = c_offset < np.abs(offset)
            c_offset[a] = -c_offset[a]
        return c_offset * np.sign(offset)
    idx = self._inputs["residue_index"]
    offset = np.array(idx[:,None] - idx[None,:])

    if self.protocol == "binder":
        c_offset = cyclic_offset(self._binder_len)
        offset[self._target_len:,self._target_len:] = c_offset
    self._inputs["offset"] = offset

def int2aa(seq):
    aacode = {
        "0": "A",
        "1": "R",
        "2": "N",
        "3": "D",
        "4": "C",
        "5": "Q",
        "6": "E",
        "7": "G",
        "8": "H",
        "9": "I",
        "10": "L",
        "11": "K",
        "12": "M",
        "13": "F",
        "14": "P",
        "15": "S",
        "16": "T",
        "17": "W",
        "18": "Y",
        "19": "V"
    }
    aa = [aacode[str(int(i))] for i in seq]
    return "".join(aa)

def set_model(binder, pdb, target_hotspot, data_dir):
    target_chain = "A" # Chain id of protein
    if target_hotspot == "": target_hotspot = None
    target_flexible = False # allow backbone of target structure to be flexible

    binder_len = None # length of binder to hallucination
    binder_seq = binder
    binder_seq = re.sub("[^A-Z]", "", binder_seq.upper())
    if len(binder_seq) > 0:
        binder_len = len(binder_seq)
    else:
        binder_seq = None # if defined, will initialize design with this sequence
    # model config
    use_multimer = True # use alphafold-multimer for design
    num_recycles = 6 # ["0", "1", "3", "6"]
    num_models = "1" # ["1", "2", "3", "4", "5", "all"]
    num_models = 5 if num_models == "all" else int(num_models) # number of trained models to use during optimization

    x = {
        "pdb_filename":pdb,
        "chain":target_chain,
        "binder_len":binder_len,
        "hotspot":target_hotspot,
        "use_multimer":use_multimer,
        "rm_target_seq":target_flexible
        }

    if "x_prev" not in dir() or x != x_prev:
        clear_mem()
        model = mk_afdesign_model(
        protocol="binder",
        use_multimer=x["use_multimer"],
        num_recycles=num_recycles,
        recycle_mode="sample",
        data_dir=data_dir) # Here is the dir of alphafold params

        model.prep_inputs(
        **x,
        ignore_missing=False
        )
        x_prev = copy_dict(x)
        binder_len = model._binder_len
        add_cyclic_offset(model, bug_fix=True)
        model.restart(seq=binder_seq)
    return model

class CyclicPeptide(ObjectiveFunction):
    name: str = "peptide"
    dims: int = 0
    turn: float = 1
    func_args: dict[str, Any] = field(
        default_factory=lambda: {
            "file_dir": None,
            "target_hotspot": None,
            "alphafold_params": None
        }
    )

    def __post_init__(self) -> None:
        assert self.dims > 0
        self.pdb_path = self.func_args["file_dir"]
        self.target_hotspot = self.func_args["target_hotspot"]
        self.data_dir = self.func_args["alphafold_params"]
        self.lb = np.array([0] * self.dims)
        self.ub = np.array([19] * self.dims)

    @override
    def _scaled(self, y: float) -> float:
        return y

    @override
    def __call__(self, x: np.ndarray, saver: bool = True, return_scaled=False) -> float:
        binder_seq = int2aa(x)
        model = set_model(binder_seq, self.pdb_path)
        model.predict()
        model.save_pdb(f"{self.name}-{self.dims}/{binder_seq}.pdb")
        values = score_interface(f"{self.name}-{self.dims}/{binder_seq}.pdb")
        return float(values['interface_sc']) * float(values['interface_dSASA']) / 100

import warnings

import numpy as np
from matplotlib import pyplot as plt
from nilearn import plotting

from gsp_neuro import data_loading as dload
from gsp_neuro import utils as ut
from gsp_neuro.deps.cmtk_viz import (
    plot_lausanne2018_surface_ctx,
)  # to make it accessible through this module


def plot_connectome(C):
    plt.figure(figsize=(8, 8))
    plt.matshow(C, fignum=0)
    plt.show()


def plot_signal_3d(brain, signal, mesh_type="inflated_whole", cmap = "Spectral"):
    
    sym_map = False
    if signal in brain.signals.keys():
        roi_values = brain.signals[signal]
    elif isinstance(signal, int):
        roi_values = brain.G.U[:, signal]
        sym_map = True
    else:
        raise Exception("Unknown signal")

    mesh_signals = ut.atlas2mesh_space(roi_values, brain.scale)
    mesh_files = dload.load_mesh(brain.data_path)

    hemi = mesh_type.split("_")[1]
    signal2plot = np.nan_to_num((mesh_signals[hemi]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        surf = plotting.view_surf(
            mesh_files[mesh_type], signal2plot, symmetric_cmap=sym_map, cmap=cmap, threshold = 1e-6
        )

    return surf

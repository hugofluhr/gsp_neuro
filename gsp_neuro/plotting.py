import warnings

import numpy as np
from matplotlib import pyplot as plt
from nilearn import plotting

from gsp_neuro import data_loading as dload
from gsp_neuro import utils as ut
from gsp_neuro.deps.cmtk_viz import (
    plot_lausanne2018_surface_ctx,
)  # to make it accessible through this module


def plot_connectome(brain, thresh=None, black_bg=True, cmap = 'Reds', output_file=None):
    if thresh == None:
        thresh = 94 + brain.scale
    plot = plotting.plot_connectome(
        brain.adjacency,
        brain.G.coords,
        node_size=brain.G.d,
        edge_threshold="{}%".format(thresh),
        edge_vmin=0,
        edge_vmax=np.max(brain.adjacency.flatten()),
        output_file=output_file,
        edge_cmap=cmap,
        black_bg=black_bg,
    )
    return plot

def plot_signal_2d(brain, signal, cmap="Spectral"):

    if signal in brain.signals.keys():
        roi_values = brain.signals[signal]
    elif isinstance(signal, int):
        roi_values = brain.G.U[:, signal]
    else:
        raise Exception("Unknown signal")

    fig = plot_lausanne2018_surface_ctx(roi_values, scale = brain.scale, cmap = cmap)
    return fig


def plot_signal_3d(
    brain, signal, mesh_type="inflated_whole", cmap="Spectral", black_bg=False
):

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
            mesh_files[mesh_type],
            signal2plot,
            symmetric_cmap=sym_map,
            cmap=cmap,
            threshold=1e-6,
            black_bg=black_bg,
        )

    return surf

def plot_atlas(brain, mesh_type, black_bg = False):

    roi_values = list(range(1, brain.adjacency.shape[0]+1))
    mesh_signals = ut.atlas2mesh_space(roi_values, brain.scale)
    mesh_files = dload.load_mesh(brain.data_path)

    hemi = mesh_type.split("_")[1]
    signal2plot = np.nan_to_num((mesh_signals[hemi]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        surf = plotting.view_surf(
            mesh_files[mesh_type],
            signal2plot,
            cmap='gist_ncar',
            symmetric_cmap=False,
            threshold=1e-6,
            black_bg=black_bg,
        )

    return surf
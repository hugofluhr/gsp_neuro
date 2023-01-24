import warnings

import numpy as np
from matplotlib import pyplot as plt
from seaborn import color_palette as palette
from nilearn import plotting
from pygsp import graphs

from gsp_neuro import data_loading as dload
from gsp_neuro import utils as ut
from gsp_neuro.deps.cmtk_viz import (
    plot_lausanne2018_surface_ctx,
)  # to make it accessible through this module


def plot_connectome(
    brain, thresh=None, black_bg=False, cmap="gist_heat", node_color=[[46/255,120/255,176/255.,0.5]], output_file=None
):
    if thresh == None:
        thresh = 94 + brain.scale
    plot = plotting.plot_connectome(
        brain.adjacency,
        brain.G.coords,
        node_size=30,  # brain.G.d,
        node_color=node_color,
        edge_threshold="{}%".format(thresh),
        edge_vmin=0,
        edge_vmax=np.max(brain.adjacency.flatten()),
        output_file=output_file,
        edge_cmap=cmap,
        black_bg=black_bg,
    )
    return plot


def plot_signal_2d(brain, signal, cmap="Spectral", outpath=None):

    if signal in brain.signals.keys():
        roi_values = brain.signals[signal]
    elif isinstance(signal, int):
        roi_values = brain.G.U[:, signal]
    else:
        raise Exception("Unknown signal")

    plot_lausanne2018_surface_ctx(
        roi_values, scale=brain.scale, cmap=cmap, outpath=outpath
    )


def plot_signal_3d(
    brain,
    signal,
    mesh_type="inflated_whole",
    cmap="Spectral",
    black_bg=False,
    vmin=None,
    vmax=None,
    title=None,
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
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            colorbar_height=0.6,
            # threshold=1e-6,  # problem with threshold because sometime signals have zero values
            black_bg=black_bg,
            title=title,
        )

    return surf


def plot_atlas(brain, mesh_type, black_bg=False):

    roi_values = list(range(1, brain.adjacency.shape[0] + 1))
    mesh_signals = ut.atlas2mesh_space(roi_values, brain.scale)
    mesh_files = dload.load_mesh(brain.data_path)

    hemi = mesh_type.split("_")[1]
    signal2plot = np.nan_to_num((mesh_signals[hemi]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        surf = plotting.view_surf(
            mesh_files[mesh_type],
            signal2plot,
            cmap="gist_ncar",
            symmetric_cmap=False,
            threshold=1e-6,
            black_bg=black_bg,
        )

    return surf


def plot_spectral_density(brain, signal, n_bands=3, savefig=False, out_path=None):
    normed_power, power_in_bands = ut.compute_spectral_density(brain, signal, n_bands)
    x = np.arange(1, int(brain.G.N))
    bands = np.array_split(x, n_bands)
    if signal == "thickness":
        signal_name = "Cortical Thickness"
    elif signal == "pial_lgi":
        signal_name = "Local Gyrification Index"

    fig, ax = plt.subplots(figsize=(6,6))

    # ax.plot(x,normed_power[x],'k', lw=1)
    markerline, _, _ = ax.stem(
        x, normed_power[x], "k", markerfmt="", linefmt="k-", basefmt=""
    )
    plt.setp(markerline, markersize=3)

    ymin, ymax = ax.get_ylim()
    ymin = 0.5 * ymin
    ymax = ymax + 0.1 * ymax

    cm = palette("mako", 6)
    alpha = 0.3
    text_kwargs = dict(ha="center", va="center")

    #TO DO : handle different number of bands!
    # ax.fill_between(
    #     [0, bands[0][-1] - 0.5], 0, ymax - abs(ymin), alpha=alpha, color=cm[-4]
    # )  # color='#fee8c8')
    # ax.text(
    #     np.mean(bands[0]),
    #     0.9 * ymax,
    #     "{:.1f}%".format(100 * power_in_bands[0]),
    #     text_kwargs,
    # )
    # ax.fill_between(
    #     [bands[0][-1] - 0.5, bands[1][-1] + 0.5],
    #     0,
    #     ymax - abs(ymin),
    #     alpha=alpha,
    #     color=cm[-3],
    # )  # color='#fdbb84')
    # ax.text(
    #     np.mean(bands[1]),
    #     0.9 * ymax,
    #     "{:.1f}%".format(100 * power_in_bands[1]),
    #     text_kwargs,
    # )
    # ax.fill_between(
    #     [bands[1][-1] + 0.5, bands[2][-1]],
    #     0,
    #     ymax - abs(ymin),
    #     alpha=alpha,
    #     color=cm[-2],
    # )  # color='#e34a33')
    # ax.text(
    #     np.mean(bands[2]),
    #     0.9 * ymax,
    #     "{:.1f}%".format(100 * power_in_bands[2]),
    #     text_kwargs,
    # )

    ax.set_xlim(-1, bands[2][-1] + 1)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Graph Frequency Index [λ]")
    ax.set_ylabel("Normalized Power")

    # better to use lambdas on x-axis
    #ax.set_xticks(np.linspace(0, x[-1], 11))
    #ax.set_xticklabels(np.round(np.linspace(0, 1, 11), 2))

    #ax.set_title("Spectral Power Density of " + signal_name)
    ax.set_title("Spectral Power Density")

    plt.show()

    return fig

def plot_signal_on_pruned_graph(brain, signal, title=None):

    small_graph = brain.G.subgraph(np.arange(68))
    small_graph = graphs.Graph(ut.prune_adjacency(small_graph.W.toarray(), 92), coords=small_graph.coords[:,(1,2)])
    small_graph.plotting['edge_color'] = (0.5,0.5,0.5,1)

    plt.set_cmap('plasma')
    fig_signal, ax= small_graph.plot(brain.signals[signal][:68], edges=True, vertex_size=300, title = title)
    fig_signal.set_size_inches(8,7)
    plt.axis('off')

    return fig_signal
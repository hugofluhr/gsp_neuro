# Copyright (C) 2009-2022, Ecole Polytechnique Federale de Lausanne (EPFL) and
# Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland, and CMP3 contributors
# All rights reserved.
#
#  This software is distributed under the open-source license Modified BSD.

"""Module that defines CMTK utility functions for plotting Lausanne parcellation files."""

import matplotlib.pyplot as plt
from nilearn import datasets, plotting

from gsp_neuro.deps.cmtk_util import get_lausanne2018_parcellation_annot
from gsp_neuro.utils import atlas2mesh_space


def plot_lausanne2018_surface_ctx(
    roi_values, scale=1,
    cmap="Spectral",
    save_fig=False, outpath = None
):
    """
    Plots a set of values on the cortical surface of a given Lausanne 2018 parcellation scale.

    Parameters
    ----------
    roi_values : numpy array
        The values to be plotted on the surface. The array should
        have as many values as regions of interest

    scale : {'scale1', 'scale2', 'scale3', 'scale4', 'scale5'}
        Scale of the Lausanne 2018 atlas to be used

    cmap : string
        Colormap to use for plotting, default "Spectral"

    save_fig : bool
        Whether to save the generated figures, default: `False`

    output_dir : string
        Directory to save the figure, only used when
        `save_fig == True`

    filename : string
        Filename of the saved figure (without the extension),
        only used when `save_fig == True`

    fmt : string
        Format the figure is saved
        (Default: "png", also
        accepted are "pdf", "svg", and others, depending
        on the backend used)

    """
    # Surface mesh
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage3")

    signal_mesh_space = atlas2mesh_space(roi_values, scale)

    # Get min and max values
    vmin = min(roi_values)
    vmax = max(roi_values)

    # Center around 0
    max_val = max([abs(vmin), vmax])
    vmax = max_val
    vmin = -max_val

    # Creation of list to allow iteration
    # and reduce duplication of plotting.plot_surf_roi()
    hemis = [
        'right', 'left', 'right', 'left',
        'right', 'left', 'right', 'left',
    ]
    views = [
        'lateral', 'lateral', 'medial', 'medial',
        'ventral', 'ventral', 'dorsal', 'dorsal'
    ]
    surfaces = [f'pial_{hemi}' for hemi in hemis]
    bg_maps = [f'sulc_{hemi}' for hemi in hemis]
    roi_vectors = [signal_mesh_space['right'], signal_mesh_space['left']]*4

    # Initial a figure with [2 x 4] subplots
    fig, axs = plt.subplots(nrows=2, ncols=4,
                            subplot_kw={'projection': '3d'},
                            figsize=(20, 10))
    axs = axs.flatten()

    # Iterate over the list of views to render
    for i, (hemi, surf, bg_map, view, vector, ax) in enumerate(
        zip(hemis, surfaces, bg_maps, views, roi_vectors, axs)
    ):
        plotting.plot_surf_roi(fsaverage[f'{surf}'], roi_map=vector,
                               hemi=hemi, view=view,
                               bg_map=fsaverage[f'{bg_map}'], bg_on_data=True,
                               darkness=.5,
                               cmap=cmap, vmin=vmin, vmax=vmax,
                               axes=ax)


    # stuff from Joan code
    axs[2].view_init(elev=90, azim=270)
    axs[3].view_init(elev=270, azim=90)

    # Save the figure in the desired format if enabled
    if outpath != None:
        fig.savefig(outpath)
    
    #return fig


def plot_lausanne2018_surface_ctx_mini(
    roi_values, scale=1,
    cmap="Spectral",
    save_fig=False, outpath = None
):
    """
    Plots a set of values on the cortical surface of a given Lausanne 2018 parcellation scale.

    Parameters
    ----------
    roi_values : numpy array
        The values to be plotted on the surface. The array should
        have as many values as regions of interest

    scale : {'scale1', 'scale2', 'scale3', 'scale4', 'scale5'}
        Scale of the Lausanne 2018 atlas to be used

    cmap : string
        Colormap to use for plotting, default "Spectral"

    save_fig : bool
        Whether to save the generated figures, default: `False`

    output_dir : string
        Directory to save the figure, only used when
        `save_fig == True`

    filename : string
        Filename of the saved figure (without the extension),
        only used when `save_fig == True`

    fmt : string
        Format the figure is saved
        (Default: "png", also
        accepted are "pdf", "svg", and others, depending
        on the backend used)

    """
    # Surface mesh
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage")

    signal_mesh_space = atlas2mesh_space(roi_values, scale)

    # Get min and max values
    vmin = min(roi_values)
    vmax = max(roi_values)

    # Center around 0
    max_val = max([abs(vmin), vmax])
    vmax = max_val
    vmin = -max_val

    # Creation of list to allow iteration
    # and reduce duplication of plotting.plot_surf_roi()
    hemis = [
        'right', 'left'
    ]
    views = [
        'lateral', 'lateral'
    ]
    surfaces = [f'pial_{hemi}' for hemi in hemis]
    bg_maps = [f'sulc_{hemi}' for hemi in hemis]
    roi_vectors = [signal_mesh_space['right'], signal_mesh_space['left']]*4

    # Initial a figure with [2 x 4] subplots
    fig, axs = plt.subplots(nrows=1, ncols=2,
                            subplot_kw={'projection': '3d'},
                            figsize=(20, 10))
    axs = axs.flatten()

    # Iterate over the list of views to render
    for i, (hemi, surf, bg_map, view, vector, ax) in enumerate(
        zip(hemis, surfaces, bg_maps, views, roi_vectors, axs)
    ):
        plotting.plot_surf_roi(fsaverage[f'{surf}'], roi_map=vector,
                               hemi=hemi, view=view,
                               bg_map=fsaverage[f'{bg_map}'], bg_on_data=True,
                               darkness=.5,
                               cmap=cmap, vmin=vmin, vmax=vmax,
                               axes=ax)

    # Save the figure in the desired format if enabled
    if outpath != None:
        fig.savefig(outpath)
    
    #return fig
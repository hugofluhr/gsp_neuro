import os
import numpy as np
import matplotlib.pyplot as plt
import nilearn
from nilearn import plotting
import nibabel as nb
from sklearn.utils import Bunch
import pandas as pd



def plot_surface_ld(roi_values, scale, center_at_zero=False, cmap='magma',
                  vmin=None, vmax=None, fig=None, axs=None,label_dir = './'):
    
    annots = [os.path.join(label_dir,'data','plotting','label','rh.lausanne2008.scale{}.annot'.format(scale)),
              os.path.join(label_dir,'data','plotting','label','lh.lausanne2008.scale{}.annot'.format(scale))]
    annot_right = nb.freesurfer.read_annot(annots[0])
    annot_left = nb.freesurfer.read_annot(annots[1])
    labels_right = [elem.decode('utf-8') for elem in annot_right[2]]
    labels_left = [elem.decode('utf-8') for elem in annot_left[2]]
    desikan_atlas = Bunch(map_left=annot_left[0],
                          map_right=annot_right[0])
    parcellation_right = desikan_atlas['map_right']
    roi_vect_right = np.zeros_like(parcellation_right, dtype=float) * np.nan
    parcellation_left = desikan_atlas['map_left']
    roi_vect_left = np.zeros_like(parcellation_left, dtype=float)*np.nan
    roifname = os.path.join(label_dir,'data','plotting','Lausanne2008_Yeo7RSNs.xlsx')
    roidata = pd.read_excel(roifname, sheet_name='SCALE {}'.format(scale))
    cort = np.where(roidata['Structure'] == 'cort')[0]
    right_rois = ([roidata['Label Lausanne2008'][i] for i in range(len(roidata))
                   if ((roidata['Hemisphere'][i] == 'rh') &
                   (roidata['Structure'][i] == 'cort'))])
    left_rois = ([roidata['Label Lausanne2008'][i] for i in range(len(roidata))
                  if ((roidata['Hemisphere'][i] == 'lh') &
                  (roidata['Structure'][i] == 'cort'))])

    for i in range(len(right_rois)):
        label_id = labels_right.index(right_rois[i])
        ids_roi = np.where(parcellation_right == label_id)[0]
        roi_vect_right[ids_roi] = roi_values[i]

    for i in range(len(left_rois)):
        label_id = labels_left.index(left_rois[i])
        ids_roi = np.where(parcellation_left == label_id)[0]
        roi_vect_left[ids_roi] = roi_values[len(right_rois)+i]

    fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')
    if vmin is None:
        vmin = min([0, min(roi_values)])
    if vmax is None:
        vmax = max(roi_values)
    if center_at_zero:
        max_val = max([abs(vmin), vmax])
        vmax = max_val
        vmin = -max_val
    if fig is None:
        fig, axs = plt.subplots(1, 6, figsize=(18, 2),
                                subplot_kw={'projection': '3d'})

    plotting.plot_surf_roi(fsaverage['pial_left'], roi_map=roi_vect_left,
                           hemi='left', view='medial',
                           bg_map=fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig, axes=axs[0])

    plotting.plot_surf_roi(fsaverage['pial_right'], roi_map=roi_vect_right,
                           hemi='right', view='lateral',
                           bg_map=fsaverage['sulc_right'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig, axes=axs[1])

    plotting.plot_surf_roi(fsaverage['pial_right'], roi_map=roi_vect_right,
                           hemi='right', view='dorsal',
                           bg_map=fsaverage['sulc_right'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig, axes=axs[2])

    plotting.plot_surf_roi(fsaverage['pial_left'], roi_map=roi_vect_left,
                           hemi='left', view='dorsal',
                           bg_map=fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig, axes=axs[2])

    plotting.plot_surf_roi(fsaverage['pial_left'], roi_map=roi_vect_left,
                           hemi='left', view='ventral',
                           bg_map=fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig, axes=axs[3])

    plotting.plot_surf_roi(fsaverage['pial_right'], roi_map=roi_vect_right,
                           hemi='right', view='ventral',
                           bg_map=fsaverage['sulc_right'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig, axes=axs[3])

    plotting.plot_surf_roi(fsaverage['pial_left'], roi_map=roi_vect_left,
                           hemi='left', view='lateral',
                           bg_map=fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig, axes=axs[4])

    plotting.plot_surf_roi(fsaverage['pial_right'], roi_map=roi_vect_right,
                           hemi='right', view='medial',
                           bg_map=fsaverage['sulc_right'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig, axes=axs[5])

    axs[2].view_init(elev=90, azim=270)
    axs[3].view_init(elev=270, azim=90)
    for i in range(6):
        if i in [2, 3]:
            axs[i].dist = 5.7
        else:
            axs[i].dist = 6

    fig.tight_layout()

    return fig, axs


def plot_surface_hd(roi_values, scale, map_roi, output_dir, center_at_zero=False,
                  cmap='magma', vmin=None, vmax=None, fmt='pdf',label_dir = './'):

    annots = [os.path.join(label_dir,'data','plotting','label','rh.lausanne2008.scale{}.annot'.format(scale)),
              os.path.join(label_dir,'data','plotting','label','lh.lausanne2008.scale{}.annot'.format(scale))]

    annot_right = nb.freesurfer.read_annot(annots[0])
    annot_left = nb.freesurfer.read_annot(annots[1])

    labels_right = [elem.decode('utf-8') for elem in annot_right[2]]
    labels_left = [elem.decode('utf-8') for elem in annot_left[2]]

    desikan_atlas = Bunch(map_left=annot_left[0],
                          map_right=annot_right[0])

    parcellation_right = desikan_atlas['map_right']
    roi_vect_right = np.zeros_like(parcellation_right, dtype=float) * np.nan

    parcellation_left = desikan_atlas['map_left']
    roi_vect_left = np.zeros_like(parcellation_left, dtype=float) * np.nan

    roifname = os.path.join(label_dir,'data','plotting','Lausanne2008_Yeo7RSNs.xlsx')
    roidata = pd.read_excel(roifname, sheet_name='SCALE {}'.format(scale))
    cort = np.where(roidata['Structure'] == 'cort')[0]

    right_rois = ([roidata['Label Lausanne2008'][i] for i in
                   range(len(roidata)) if ((roidata['Hemisphere'][i] == 'rh') &
                   (roidata['Structure'][i] == 'cort'))])
    left_rois = ([roidata['Label Lausanne2008'][i] for i in
                  range(len(roidata)) if ((roidata['Hemisphere'][i] == 'lh') &
                  (roidata['Structure'][i] == 'cort'))])

    for i in range(len(right_rois)):
        label_id = labels_right.index(right_rois[i])
        ids_roi = np.where(parcellation_right == label_id)[0]
        roi_vect_right[ids_roi] = roi_values[i]

    for i in range(len(left_rois)):
        label_id = labels_left.index(left_rois[i])
        ids_roi = np.where(parcellation_left == label_id)[0]
        roi_vect_left[ids_roi] = roi_values[len(right_rois) + i]

    fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')

    if vmin is None:
        vmin = min([0, min(roi_values)])
    if vmax is None:
        vmax = max(roi_values)

    if center_at_zero:
        max_val = max([abs(vmin), vmax])
        vmax = max_val
        vmin = -max_val

    fig, axs = plt.subplots(subplot_kw={'projection': '3d'})
    plotting.plot_surf_roi(fsaverage['pial_right'], roi_map=roi_vect_right,
                           hemi='right', view='lateral',
                           bg_map=fsaverage['sulc_right'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig)
    fig.savefig('{}/map_{}_scale{}_right_lateral.{}'.format(output_dir,
                map_roi, scale, fmt), format=fmt)

    fig, axs = plt.subplots(subplot_kw={'projection': '3d'})
    plotting.plot_surf_roi(fsaverage['pial_left'], roi_map=roi_vect_left,
                           hemi='left', view='lateral',
                           bg_map=fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig)
    fig.savefig('{}/map_{}_scale{}_left_lateral.{}'.format(output_dir,
                map_roi, scale, fmt), format=fmt)

    fig, axs = plt.subplots(subplot_kw={'projection': '3d'})
    plotting.plot_surf_roi(fsaverage['pial_right'], roi_map=roi_vect_right,
                           hemi='right', view='medial',
                           bg_map=fsaverage['sulc_right'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig)
    fig.savefig('{}/map_{}_scale{}_right_medial.{}'.format(output_dir,
                map_roi, scale, fmt), format=fmt)

    fig, axs = plt.subplots(subplot_kw={'projection': '3d'})
    plotting.plot_surf_roi(fsaverage['pial_left'], roi_map=roi_vect_left,
                           hemi='left', view='medial',
                           bg_map=fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig)
    fig.savefig('{}/map_{}_scale{}_left_medial.{}'.format(output_dir,
                map_roi, scale, fmt), format=fmt)

    fig, axs = plt.subplots(subplot_kw={'projection': '3d'})
    plotting.plot_surf_roi(fsaverage['pial_right'], roi_map=roi_vect_right,
                           hemi='right', view='ventral',
                           bg_map=fsaverage['sulc_right'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig)
    fig.savefig('{}/map_{}_scale{}_right_ventral.{}'.format(output_dir,
                map_roi, scale, fmt), format=fmt)

    fig, axs = plt.subplots(subplot_kw={'projection': '3d'})
    plotting.plot_surf_roi(fsaverage['pial_left'], roi_map=roi_vect_left,
                           hemi='left', view='ventral',
                           bg_map=fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig)
    fig.savefig('{}/map_{}_scale{}_left_ventral.{}'.format(output_dir,
                map_roi, scale, fmt), format=fmt)

    fig, axs = plt.subplots(subplot_kw={'projection': '3d'})
    plotting.plot_surf_roi(fsaverage['pial_right'], roi_map=roi_vect_right,
                           hemi='right', view='dorsal',
                           bg_map=fsaverage['sulc_right'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig)
    fig.savefig('{}/map_{}_scale{}_right_dorsal.{}'.format(output_dir,
                map_roi, scale, fmt), format=fmt)

    fig, axs = plt.subplots(subplot_kw={'projection': '3d'})
    plotting.plot_surf_roi(fsaverage['pial_left'], roi_map=roi_vect_left,
                           hemi='left', view='dorsal',
                           bg_map=fsaverage['sulc_left'], bg_on_data=True,
                           darkness=.5, cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           figure=fig)
    fig.savefig('{}/map_{}_scale{}_left_dorsal.{}'.format(output_dir,
                map_roi, scale, fmt), format=fmt)

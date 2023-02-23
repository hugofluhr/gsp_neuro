import re
import csv
import numpy as np
import nibabel as nib
from gsp_neuro.deps.cmtk_util import get_lausanne2018_parcellation_annot


def extract_roi(regions):
    single_string = False
    if isinstance(regions, str):
        regions = [regions]
        single_string = True

    rois = []
    col_idx = []
    for i, region in enumerate(regions):
        m = re.search("ctx-(.+?) ", region)
        if m:
            found = m.group(1)
            rois.append(found)
            col_idx.append(i)
    # print("{:5d} ROIs found".format(len(rois)))
    if single_string:
        rois = rois[0]
    return rois, col_idx

def nROIs(scale):
    nROIs_per_scale = [68, 114, 216, 446, 1002]
    return nROIs_per_scale[scale-1]

def regions_in_file(file):
    with open(file, newline= "") as f:
        reader = csv.reader(f)
        headers = next(reader)
    regions = headers[9:-1:2]
    regions, _ = extract_roi(regions)
    return regions


def split_lr_rois(ROIs):
    rh_rois = [roi.replace("rh-", "") for roi in ROIs if roi.startswith("rh")]
    lh_rois = [roi.replace("lh-", "") for roi in ROIs if roi.startswith("lh")]
    return {"rh": rh_rois, "lh": lh_rois}


def  atlas2mesh_space(roi_values, scale):

    annots = [
        get_lausanne2018_parcellation_annot(scale=f"{scale}", hemi="rh"),
        get_lausanne2018_parcellation_annot(scale=f"{scale}", hemi="lh"),
    ]

    # Read annot files
    annot_right = nib.freesurfer.read_annot(annots[0])
    annot_left = nib.freesurfer.read_annot(annots[1])

    # Create vector to store intensity values (one value per vertex)
    roi_vect_right = np.zeros_like(annot_right[0], dtype=float)
    roi_vect_left = np.zeros_like(annot_left[0], dtype=float)

    # Convert labels to strings, labels are the same as 2018 is symmetric
    labels = [str(elem, "utf-8") for elem in annot_right[2]]

    # Create roi vectors
    for i in range(len(labels[1:])):  # skip 'unknown'
        ids_roi = np.where(annot_right[0] == i + 1)[0]
        roi_vect_right[ids_roi] = roi_values[i]

    for i in range(len(labels[1:])):  # skip 'unknown'
        ids_roi = np.where(annot_left[0] == i + 1)[0]
        roi_vect_left[ids_roi] = roi_values[i + len(labels) - 1]

    return {
        "left": roi_vect_left,
        "right": roi_vect_right,
        "whole": np.concatenate((roi_vect_left, roi_vect_right), axis=0),
    }


def compute_spectral_density(brain, signal, n_bands = 3):
    
    # start at 1 to get rid of continuous component
    x = np.arange(1, int(brain.G.N))
    bands = np.array_split(x, n_bands)
    signal_gft = brain.G.gft(brain.signals[signal])
    spectral_power = np.square(signal_gft)
    norm_factor = np.sum(spectral_power[1:])
    normed_power = 1/norm_factor * spectral_power

    power_in_bands = tuple(np.sum(np.square(signal_gft[bands[i]]))/norm_factor for i in range(n_bands))
    return normed_power, power_in_bands    


def prune_adjacency(adjacency, thresh=80):
    new_adj = np.zeros_like(adjacency)
    new_adj[adjacency>np.percentile(adjacency,thresh)] = adjacency[adjacency>np.percentile(adjacency,thresh)]
    return new_adj
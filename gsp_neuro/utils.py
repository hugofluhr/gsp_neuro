import re
import csv
import numpy as np


def extract_roi(regions):
    single_string = False
    if isinstance(regions, str):
        regions = [regions]
        single_string = True
    
    rois = []
    col_idx = []
    for i, region in enumerate(regions):
        m = re.search('ctx-(.+?) ', region)
        if m:
            found = m.group(1)
            rois.append(found)
            col_idx.append(i)
    #print("{:5d} ROIs found".format(len(rois)))
    if single_string:
        rois = rois[0]
    return rois, col_idx

def regions_in_file(file):
    with open(file, newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
    regions = headers[9:-1:2]
    regions, _ = extract_roi(regions)
    return regions

def split_lr_rois(ROIs):
    rh_rois = [roi.replace('rh-','') for roi in ROIs if roi.startswith('rh')]
    lh_rois = [roi.replace('lh-','') for roi in ROIs if roi.startswith('lh')]
    return {'rh' : rh_rois, 'lh': lh_rois}

def atlas2mesh_space(parcellation, labels, left_df, right_df, roi_values):
    roi_vect_right = np.zeros_like(parcellation['right'], dtype=float) * np.nan
    roi_vect_left = np.zeros_like(parcellation['left'], dtype=float) * np.nan

    for i in range(len(right_df)):
        label_id = labels['right'].index(right_df.loc[i,'Structures Names'])
        ids_roi = np.where(parcellation['right'] == label_id)[0]
        roi_vect_right[ids_roi] = roi_values[i]

    for i in range(len(left_df)):
        label_id = labels['left'].index(left_df.loc[i,'Structures Names'])
        ids_roi = np.where(parcellation['left'] == label_id)[0]
        roi_vect_left[ids_roi] = roi_values[len(right_df)+i]

    return {'left':roi_vect_left, 'right':roi_vect_right}
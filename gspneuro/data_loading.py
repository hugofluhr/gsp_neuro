import csv
from glob import glob
from os import path as op

import nibabel as nib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat

from gspneuro.deps.cmtk_util import get_lausanne2018_parcellation_annot


def get_ids_csv(table_path):
    rows = []
    with open(table_path, "r") as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append({head: row[idx].strip() for idx, head in enumerate(header)})
    return rows


def get_subjects(data_path, subjects_IDS_path=None):
    if subjects_IDS_path is None:
        subjects_IDS_path = op.join(data_path, "Ids.csv")

    subjects_ids = [sub.get("subjID") for sub in get_ids_csv(subjects_IDS_path)]
    subject_folders = [
        subfolder
        for subfolder in glob(data_path + "BIOPSYCHO_CTRLS/sub-*")
        if subfolder[-8:] in subjects_ids
    ]
    subjects = [op.normpath(op.basename(sub_path)) for sub_path in subject_folders]
    subjects.sort()
    return subjects

def get_atrophies_subjects(atrophy_dir="/Users/hugofluhr/chuv/data/typical_atrophies/"):
    file_path = glob(
        op.join(atrophy_dir, "HUGO-Project_atlas-LFIIHIFIF_desc-scale1-stats-table.csv"))[0]
    
    df = pd.read_csv(file_path)
    return np.sort(df.participant_id.unique())

def load_patient_infos(atrophy_dir="/Users/hugofluhr/chuv/data/typical_atrophies/"):
    info_file_path = op.join(atrophy_dir, "patient_description.json")
    with open(info_file_path, 'r') as json_file:
        info = json.load(json_file)
    return info

def get_subject_dir(data_path, subject_ID):
    if "consensus" in subject_ID:
        return op.join(data_path, "BIOPSYCHO_CTRLS", "consensus")
    else:
        return op.join(data_path, "BIOPSYCHO_CTRLS", subject_ID)


def get_sub_connectomes_paths(subject_path, scale=1):

    connectome_paths = []
    # to handle single directory case
    if isinstance(subject_path, str):
        subject_path = [subject_path]
    for sub_path in subject_path:
        if "consensus" in sub_path:
            connectome_paths.append(
                glob(op.join(sub_path, "*scale{}.npz".format(scale)), recursive=True)[0]
            )
        else:
            connectome_paths.append(
                glob(
                    op.join(sub_path, "**/*scale{}*.mat".format(scale)), recursive=True
                )[0]
            )

    connectome_paths.sort()
    return connectome_paths


def load_connectome(connectome_path, field="fibDensity"):
    data = loadmat(connectome_path).get("newConnMat")
    fields = list(data[0, 0].dtype.fields.keys())
    # fields : ['nFiber', 'Length', 'fibDensity', 'nVoxMat', 'gFA', 'MD']
    try:
        connectome = data[0, 0][field]
        length_mat = data[0, 0]["Length"]
        # print("Succesfully loaded a connectome with {} nodes.".format(connectome.shape[0]))
        connectome = connectome.astype(np.single)
        # connectome = np.divide(connectome, length_mat,out=np.zeros_like(connectome), where=(length_mat!=0))
        return connectome.astype(np.single)
    except:
        print("No nFiber field in the file")


def load_consensus(consensus_path, weighted=True):
    data = np.load(consensus_path)
    if weighted:
        return data["G"] * data["av_weight"]
    else:
        return data['G']


def read_fscolorlut(lutFile):
    # Readind a color LUT file
    fid = open(lutFile)
    LUT = fid.readlines()
    fid.close()

    # Make dictionary of labels
    LUT = [row.split() for row in LUT]
    st_names = []
    st_codes = []
    st_red = []
    st_green = []
    st_blue = []
    for row in LUT:
        if (
            len(row) > 1 and row[0][0] != "#" and row[0][0] != "\\\\"
        ):  # Get rid of the comments
            st_codes.append(int(row[0]))
            st_names.append(row[1])
            st_red.append(row[2])
            st_green.append(row[3])
            st_blue.append(row[4])

    return pd.DataFrame.from_dict(
        {
            "st_code": st_codes,
            "st_name": st_names,
            "R": st_red,
            "G": st_green,
            "B": st_blue,
        }
    )


def read_coords(data_path="/Users/hugofluhr/chuv/data/", scale=1):
    file_path = glob(
        data_path + "requestionforconnectomes/*.scale{}.*regCoords.txt".format(scale)
    )[0]
    df_coord = pd.read_csv(file_path)
    df_coord.rename(columns=lambda x: x.strip(), inplace=True)
    df_coord["Structures Names"] = df_coord["Structures Names"].str.strip()

    return df_coord


def get_ctx_indices(df):
    return list(df[df["Structures Names"].str.contains("ctx")].index)


def get_signal(data_path, subject, signal, scale, mtype="mean"):
    file_path = glob(
        op.join(data_path, "biopyscho_ctrls_signals/{}-sc{}*.csv".format(signal, scale))
    )[0]
    df = pd.read_csv(file_path)

    df_coords = read_coords(scale=scale)
    rois2keep = list(
        df_coords[df_coords["Structures Names"].str.contains("ctx")]["Structures Names"]
    )

    roi_columns = {
        roi: [i for i, s in enumerate(list(df.columns)) if roi in s]
        for roi in rois2keep
    }
    if mtype == "mean":
        data_cols = [roi_columns[roi][0] for roi in roi_columns.keys()]
    elif mtype == "std":
        data_cols = [roi_columns[roi][1] for roi in roi_columns.keys()]

    signal = df.iloc[df.index[df["participant_id"] == subject], data_cols].values

    return signal[0]

def get_atrophy_signal(subj_id, scale=3, metric='thickness', basedir="/Users/hugofluhr/chuv/data/typical_atrophies/"):
    file_path = glob(
        op.join(basedir, "HUGO-Project_atlas-LFIIHIFIF_desc-scale{}-stats-table.csv".format(scale))
    )[0]
    
    df = pd.read_csv(file_path)
    rows_of_metric = (df.metric==metric) & (df.statistics=='mean')
    ctx_columns = df.columns.str.contains('ctx')
    
    return df.loc[rows_of_metric & (df.participant_id==subj_id), ctx_columns].values[0]


def load_mesh(data_path):
    mesh_files = {
        "inflated_left": op.join(data_path, "requestionforconnectomes/lh.inflated.gii"),
        "pial_left": op.join(data_path, "requestionforconnectomes/lh.pial.gii"),
        "inflated_right": op.join(
            data_path, "requestionforconnectomes/rh.inflated.gii"
        ),
        "pial_right": op.join(data_path, "requestionforconnectomes/rh.pial.gii"),
        "pial_whole": op.join(data_path, "requestionforconnectomes/both-hemi.pial"),
        "inflated_whole": op.join(
            data_path, "requestionforconnectomes/both-hemi.inflated"
        ),
    }

    return mesh_files


def load_atlas(scale):
    annots = [
        get_lausanne2018_parcellation_annot(scale=scale, hemi="rh"),
        get_lausanne2018_parcellation_annot(scale=scale, hemi="lh"),
    ]

    # Read annot files
    annot_right = nib.freesurfer.read_annot(annots[0])
    annot_left = nib.freesurfer.read_annot(annots[1])

    atlas = {"right": annot_right[0], "left": annot_left[0]}
    return atlas

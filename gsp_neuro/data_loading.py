import csv
from os.path import join as pjoin
from scipy.io import loadmat
import numpy as np
import pandas as pd
from glob import glob

def get_ids_csv(table_path):
	rows = []
	with open(table_path, 'r') as file:
		csvreader = csv.reader(file)
		header = next(csvreader)
		for row in csvreader :
			rows.append({head: row[idx].strip() for idx,head in enumerate(header)})
	return rows

def get_sub_connectomes_paths(subject_path, scale=1):
	
	connectome_paths = []
	# to handle single directory case
	if isinstance(subject_path, str):
		subject_path = [subject_path]

	for sub_path in subject_path :	
		connectome_paths.append(glob(pjoin(sub_path,'**/*scale{}*.mat'.format(scale)), recursive=True)[0])

	connectome_paths.sort()
	return connectome_paths

def load_connectome(connectome_path, field = "nFiber"):
	data = loadmat(connectome_path).get("newConnMat")
	fields = list(data[0,0].dtype.fields.keys())
	# fields : ['nFiber', 'Length', 'fibDensity', 'nVoxMat', 'gFA', 'MD']
	try :
		connectome = data[0,0][field]
		length_mat = data[0,0]['Length']
		#print("Succesfully loaded a connectome with {} nodes.".format(connectome.shape[0]))
		connectome = connectome.astype(np.single)
		#connectome = np.divide(connectome, length_mat,out=np.zeros_like(connectome), where=(length_mat!=0))
		return connectome.astype(np.single)
	except  :
		print("No nFiber field in the file")


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
		if len(row) > 1 and row[0][0] != '#' and row[0][0] != '\\\\':  # Get rid of the comments
			st_codes.append(int(row[0]))
			st_names.append(row[1])
			st_red.append(row[2])
			st_green.append(row[3])
			st_blue.append(row[4])

	return pd.DataFrame.from_dict({'st_code':st_codes, 'st_name':st_names,'R':st_red,'G':st_green,'B':st_blue})

def read_coords(data_path = '/Users/hugofluhr/chuv/data/', scale=1):
	file_path = glob(data_path + 'requestionforconnectomes/*.scale{}.*regCoords.txt'.format(scale))[0]
	df_coord = pd.read_csv(file_path)
	df_coord.rename(columns=lambda x: x.strip(), inplace=True)
	df_coord["Structures Names"] = df_coord["Structures Names"].str.strip()

	return df_coord

	
def get_ctx_indices(df):
    return list(df[df["Structures Names"].str.contains('ctx')].index)


def get_signal(data_path, subject, signal, scale, mtype='mean'):
    file_path = glob(pjoin(data_path, "biopyscho_ctrls_signals/{}-sc{}*.csv".format(signal, scale)))[0]
    df = pd.read_csv(file_path)

    df_coords = read_coords(scale=scale)
    rois2keep = list(df_coords[df_coords["Structures Names"].str.contains('ctx')]['Structures Names'])

    roi_columns = {roi : [i for i, s in enumerate(list(df.columns)) if roi in s] for roi in rois2keep}
    if mtype == 'mean':
        data_cols = [roi_columns[roi][0] for roi in roi_columns.keys()]
    elif mtype == 'std':
        data_cols = [roi_columns[roi][1] for roi in roi_columns.keys()]

    signal = df.iloc[df.index[df['participant_id']==subject],data_cols].values

    return signal[0]

def load_mesh(data_path):
	mesh = {'inflated_left' : pjoin(data_path,'requestionforconnectomes/lh.inflated.gii'),
        'pial_left' : pjoin(data_path,'requestionforconnectomes/lh.pial.gii'),
        'inflated_right': pjoin(data_path,'requestionforconnectomes/rh.inflated.gii'),
        'pial_right' : pjoin(data_path,'requestionforconnectomes/rh.pial.gii')}
	
	return mesh

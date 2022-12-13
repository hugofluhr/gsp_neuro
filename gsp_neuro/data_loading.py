import csv
import os
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
		connectome_paths.append(glob(os.path.join(sub_path,'**/*scale{}*.mat'.format(scale)), recursive=True)[0])

	connectome_paths.sort()
	return connectome_paths

def load_connectome(connectome_path, field = "nFiber"):
	data = loadmat(connectome_path).get("newConnMat")
	fields = list(data[0,0].dtype.fields.keys())

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

def read_coords(data_path = '/home/localadmin/Bureau/HUGO/data/', scale=1):
	file_path = glob(data_path + 'requestionforconnectomes/*.scale{}.*regCoords.txt'.format(scale))[0]
	df_coord = pd.read_csv(file_path)
	df_coord.rename(columns=lambda x: x.strip(), inplace=True)
	df_coord["Structures Names"] = df_coord["Structures Names"].str.strip()

	return df_coord

	
def get_ctx_indices(df):
    return list(df[df["Structures Names"].str.contains('ctx')].index)
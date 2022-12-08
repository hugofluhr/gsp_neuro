import csv
import os
from scipy.io import loadmat
import numpy as np

def get_ids_csv(table_path):
	rows = []
	with open(table_path, 'r') as file:
	    csvreader = csv.reader(file)
	    header = next(csvreader)
	    for row in csvreader :
	        rows.append({head: row[idx].strip() for idx,head in enumerate(header)})
	return rows

def get_sub_connectomes_paths(subject_path):
    connectome_paths = []
    for path, subdirs, files in os.walk(subject_path):
        for name in files:
            if "connmat.mat" in name:
                connectome_paths.append(os.path.join(path, name))
    connectome_paths.sort()
    return connectome_paths

def load_connectome(connectome_path, field = "nFiber"):
    data = loadmat(connectome_path).get("newConnMat")
    fields = list(data[0,0].dtype.fields.keys())
    try :
        connectome = data[0,0][field]
        print("Succesfully loaded a connectome with {} nodes.".format(connectome.shape[0]))
        return connectome.astype(np.single)
    except  :
        print("No nFiber field in the file")

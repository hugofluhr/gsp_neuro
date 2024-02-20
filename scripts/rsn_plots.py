import numpy as np
import pandas as pd

import sys
sys.path.append('/Users/hugofluhr/chuv/repositories/gsp_neuro/')
from gspneuro import data_loading as dload 
from gspneuro import plotting as viz
from gspneuro import utils as ut
from gspneuro.brain import Brain
from tqdm import tqdm

def thresh(gft, n):
    if n==0:
        return np.zeros_like(gft)
    a_gft = np.abs(gft)
    sorted_indices = np.argsort(a_gft)[::-1]
    t = a_gft[sorted_indices[n-1]]
    m = a_gft<t
    thresholded = np.copy(gft)
    thresholded[m]=0.
    return thresholded

if __name__=='__main__':
    data_base_directory = "/Users/hugofluhr/chuv/data/"
    scale = 3
    consensus = Brain(data_base_directory, 'consensus_bin', scale)
    consensus.load_graph()  

    rsn_csv=f'/Users/hugofluhr/chuv/data/yeoatlasandlausanne2018parcellation/Polona_Laus2018.Scale{scale}-to-Yeo_symmetric.csv'
    df=pd.read_csv(rsn_csv)
    df['max_index'] = df.iloc[:, 1:].idxmax(axis=1).apply(lambda x: int(x[-1]))

    RSNs = [(df.max_index==i+1).values.astype(int) for i in range(7)]
    GFTs = [consensus.G.gft(r) for r in RSNs]

    # Reconstructing using only the dominant modes
    reconstructed = []
    for i in range(7):
        thresholded_gft=thresh(GFTs[i], 20)
        reconstructed.append(consensus.G.igft(thresholded_gft))

    # Plotting
    for i in tqdm(range(7)):
        outfile_rsn = f'/Users/hugofluhr/chuv/repositories/gsp_neuro/figures/RSNs/rsn_{i}.png'
        outfile_recons = f'/Users/hugofluhr/chuv/repositories/gsp_neuro/figures/RSNs/reconstructed_rsn_{i}.png'
        viz.plot_signal_surf_full(RSNs[i], outfile_rsn, cmap='summer')
        viz.plot_signal_surf_full((reconstructed[i]>0.5).astype(int), outfile_recons, cmap='summer')
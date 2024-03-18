import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/hugofluhr/chuv/repositories/gsp_neuro/')
from gspneuro import plotting as viz
from gspneuro.brain import Brain
from multiprocessing import Pool
from functools import partial
import os
from tqdm import tqdm

def create_directory(directory):
    if not os.path.exists(directory):
       os.makedirs(directory)

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

def lp(gft, n):
    filtered = np.zeros_like(gft)
    filtered[:n] = gft[:n]
    return filtered

def process_rsn(i, RSNs, brain, params, out_directory):
    # Reconstruction
    rsn = RSNs[i]
    consensus = brain
    gft = consensus.G.gft(rsn)

    if params['filter_type'] == 'original':
        filtered_rsn = rsn
    elif params['filter_type'] == 'thresh':
        filtered_rsn = consensus.G.igft(thresh(gft, params['nb_components']))
    elif params['filter_type'] == 'lp':
        filtered_rsn = consensus.G.igft(lp(gft, params['nb_components']))
    else:
        raise ValueError('Invalid filter type')
    
    outfile_recons = out_directory + f'rsn_{i+1}.png'
    if params['filter_type'] == 'original':
        outfile_rsn = out_directory + f'rsn_{i+1}.png'
        if not os.path.isfile(outfile_rsn):
            viz.plot_signal_surf_full(RSNs[i], outfile_rsn, cmap='summer', scale=params['scale'])

    if os.path.isfile(outfile_recons):
        return
    if params['bin']:
        viz.plot_signal_surf_full((filtered_rsn > 0.5).astype(int), outfile_recons, cmap='summer', scale=params['scale'])
    else:
        viz.plot_signal_surf_full(filtered_rsn, outfile_recons, cmap='viridis', sym_map=False, scale=params['scale'])

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=3, help='Scale parameter')
    parser.add_argument('--filter_type', type=str, default='lp', help='Filter type')
    parser.add_argument('--nb_components', type=int, default=20, help='Number of components')
    parser.add_argument('--bin', type=int, default=1, help='Binarized flag')
    parser.add_argument('--rsn_original', type=bool, default=False, help='RSN original flag')
    args = parser.parse_args()

    rsn_original = args.rsn_original
    scale = args.scale
    filter_type = args.filter_type
    nb_components = args.nb_components
    binarized = args.bin

    print('Parameters: Scale:', scale, 'Filter Type:', filter_type, 'Number of Components:', nb_components, 'Binarized:', binarized, 'RSN Original:', rsn_original)

    # load data
    data_base_directory = "/Users/hugofluhr/chuv/data/"
    consensus = Brain(data_base_directory, 'consensus_bin', scale)
    consensus.load_graph()  

    rsn_csv = "/Users/hugofluhr/chuv/data/yeoatlasandlausanne2018parcellation/Polona_Laus2018.Scale{}-to-Yeo_symmetric.csv".format(scale)
    df = pd.read_csv(rsn_csv)
    df['max_index'] = df.iloc[:, 1:].idxmax(axis=1).apply(lambda x: int(x[-1]))

    RSNs = [(df.max_index==i+1).values.astype(int) for i in range(7)]

    # create output directory if it does not exist
    if rsn_original:
        out_directory = "/Users/hugofluhr/chuv/repositories/gsp_neuro/figures/february_2024/RSNs/scale{}_original/".format(scale)
        params = {'scale':scale, 'filter_type':'original'}
    else:
        out_directory = "/Users/hugofluhr/chuv/repositories/gsp_neuro/figures/february_2024/RSNs/scale{}_{}_{}{}{}".format(scale, filter_type, nb_components, "_bin" if binarized else "", "/")
        params = {'scale':scale, 'filter_type': filter_type, 'nb_components':nb_components,'bin': binarized}
    create_directory(out_directory)

    process_partial = partial(process_rsn, RSNs = RSNs, brain = consensus, params = params, out_directory = out_directory)
    # Processing networks
    with Pool(7) as pool:
        for _ in tqdm(pool.imap_unordered(process_partial, range(len(RSNs)))):
            pass
    
    print('>>>>> Done!')
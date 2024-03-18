import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/hugofluhr/chuv/repositories/gsp_neuro/')
from gspneuro.brain import Brain
import os
from matplotlib import pyplot as plt

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

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=3, help='Scale parameter')
    parser.add_argument('--filter_type', type=str, default='lp', help='Filter type')
    parser.add_argument('--plot_type', type=str, default='mse', help='Plot type')
    args = parser.parse_args()

    scale = args.scale
    filter_type = args.filter_type
    plot_type = args.plot_type

    print('Parameters: Scale:', scale, 'Filter Type:', filter_type)

    # load data
    data_base_directory = "/Users/hugofluhr/chuv/data/"
    consensus = Brain(data_base_directory, 'consensus_bin', scale)
    consensus.load_graph()  

    rsn_csv = "/Users/hugofluhr/chuv/data/yeoatlasandlausanne2018parcellation/Polona_Laus2018.Scale{}-to-Yeo_symmetric.csv".format(scale)
    df = pd.read_csv(rsn_csv)
    df['max_index'] = df.iloc[:, 1:].idxmax(axis=1).apply(lambda x: int(x[-1]))

    RSNs = [(df.max_index==i+1).values.astype(int) for i in range(7)]

    # create output directory if it does not exist
    out_directory = "/Users/hugofluhr/chuv/repositories/gsp_neuro/figures/february_2024/RSNs/reconstruction_errors"
    create_directory(out_directory)

    N = consensus.G.N
    x_partial = np.zeros((len(RSNs), N, N))
    rec_errors = np.zeros((len(RSNs), N))
    mse = np.zeros((len(RSNs), N))

    # computing reconstructions
    for i, rsn in enumerate(RSNs):
        x = rsn-rsn.mean()
        x_hat=consensus.G.gft(x)
        for n in range(N):
            lp_filtered = lp(x_hat,n)
            x_recons = consensus.G.igft(lp_filtered)
            x_partial[i,n,:] = x_recons
            rec_errors[i,n] = 1-np.linalg.norm(x-x_recons)/np.linalg.norm(x)
            mse[i,n] = np.linalg.norm(x-x_recons)**2/N
    
    # plotting
    normalized_mse = mse / np.amax(mse, axis=1, keepdims=True)

    if plot_type == 'mse':
        fig, axs = plt.subplots(1,2, figsize=(12,5))
        axs[0].plot(normalized_mse.T)
        axs[0].legend([f'RSN {i+1}' for i in range(7)])
        axs[0].set_title('normalized MSE for all components')
        axs[1].plot(normalized_mse[:,:int(np.round(.1*N))].T)
        axs[1].set_title('first 10% of  components')
        fig.savefig(out_directory + f'/scale_{scale}_mse_{filter_type}.png')

    elif plot_type == 'accuracy':
        fig, axs = plt.subplots(1,2, figsize=(12,5))
        axs[0].plot(rec_errors.T)
        axs[0].legend([f'RSN {i+1}' for i in range(7)])
        axs[0].set_title('Recovered signal for all components')
        axs[1].plot(rec_errors[:,:int(np.round(.1*N))].T)
        axs[1].set_title('first 10% of  components')
        fig.savefig(out_directory + f'/scale_{scale}_accuracy_{filter_type}.png')

    print('>>>>> Done!')
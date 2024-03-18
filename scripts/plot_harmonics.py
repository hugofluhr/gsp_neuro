import sys
sys.path.append('/Users/hugofluhr/chuv/repositories/gsp_neuro/')
from gspneuro import plotting as viz
from gspneuro.brain import Brain
import os
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import argparse

def process_eigmode(eigmode, consensus, outdir):
    roi_values = consensus.get_signal(eigmode)
    filename = "eigmode_{:02d}.png".format(eigmode)
    outpath = os.path.join(outdir, filename)
    viz.plot_signal_surf_full(roi_values=roi_values, outfile=outpath)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=3, help="The scale")
    parser.add_argument("--nb_modes", type=int, default=20, help="The number of eigenmodes")
    parser.add_argument("--weighted", type=bool, default=False, help="Whether weighted or not")
    args = parser.parse_args()

    scale = args.scale
    nb_modes = args.nb_modes
    weighted = args.weighted

    print('scale:', scale)
    print('nb eigenmodes:', nb_modes)
    print('weighted:', weighted)

    conf = input("Proceed? (y/n)")
    if conf != 'y':
        sys.exit()

    data_base_directory = "/Users/hugofluhr/chuv/data/"

    if weighted:
        consensus = Brain(data_base_directory, 'consensus_w', scale)
        outdir = '/Users/hugofluhr/chuv/repositories/gsp_neuro/figures/february_2024/consensus_harmonics_weighted_lr/'
    else:
        consensus = Brain(data_base_directory, 'consensus_bin', scale)
        outdir = '/Users/hugofluhr/chuv/repositories/gsp_neuro/figures/february_2024/consensus_harmonics_bin_lr/'

    consensus.load_graph(lap_type='normalized')

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    process_eigmode_partial = partial(process_eigmode, consensus=consensus, outdir = outdir)

    print('>>>> Starting to plot')

    with Pool(4) as pool:
        for _ in tqdm(pool.imap_unordered(process_eigmode_partial, range(nb_modes)), total=nb_modes):
            pass

    print('>>>> Done!')

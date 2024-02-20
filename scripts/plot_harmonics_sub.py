from gspneuro import plotting as viz
from gspneuro.brain import Brain
import os
from multiprocessing import Pool
import sys
from functools import partial
from tqdm import tqdm

def process_eigmode(eigmode, brain, outdir):
    roi_values = brain.get_signal(eigmode)
    filename = "eigmode_{:02d}.png".format(eigmode)
    outpath = os.path.join(outdir, filename)
    fig = viz.plot_signal_surf(roi_values=roi_values)
    fig.savefig(outpath, dpi=100)

if __name__=='__main__':
    if len(sys.argv) < 2:
        print("Please provide the scale as a command-line argument.")
        sys.exit(1)

    data_base_directory = "/Users/hugofluhr/chuv/data/"

    scale = int(sys.argv[1])
    nb_modes = int(sys.argv[2])
    weighted = bool(int(sys.argv[3]))

    print('scale :', scale)
    print('nb eigenmodes :', nb_modes)
    print('weighted :', weighted)

    conf = input("Proceed? (y/n)")
    if conf != 'y':
        sys.exit()

    brain = Brain(data_base_directory, 'sub-CHUVL048', scale)
    outdir = '/Users/hugofluhr/chuv/repositories/gsp_neuro/figures/sub01_harmonics_bin_lr/'

    brain.load_graph(lap_type='normalized')

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    process_eigmode_partial = partial(process_eigmode, brain=brain, outdir = outdir)

    print('>>>> Starting to plot')

    with Pool(4) as pool:
        for _ in tqdm(pool.imap_unordered(process_eigmode_partial, range(nb_modes)), total=nb_modes):
            pass

    print('>>>> Done!')

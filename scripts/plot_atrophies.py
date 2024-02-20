import argparse
from gspneuro import plotting as viz
import os
import sys
from gspneuro import data_loading as dload
import numpy as np
from tqdm import tqdm

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Your script description here")

    parser.add_argument('--scale', type=int, default=3, help="Scale value (default: 3)")
    parser.add_argument('--dpi', type=int, default=300, help="DPI value (default: 300)")

    args = parser.parse_args()

    data_base_directory = "/Users/hugofluhr/chuv/data/"

    scale = args.scale
    dpi = args.dpi

    print('\nScale :', scale)
    print('DPI :', dpi)

    atrophy_ids = dload.get_atrophies_subjects()
    print('\nSubjects :')
    print(atrophy_ids)
    print()

    conf = input("Proceed? (y/n)")
    if conf != 'y':
        sys.exit()

    # atrophies
    atrophy_thickness = [dload.get_atrophy_signal(atrophy_subj, scale=scale) for atrophy_subj in atrophy_ids]

    # control
    healthy_ids = dload.get_subjects(data_base_directory)
    healthy_thickness = [dload.get_signal(data_base_directory, healthy_subj, 'thickness', scale=scale) for healthy_subj in healthy_ids]
    # take mean of all controls
    avg_healthy_thick=np.stack(healthy_thickness).mean(axis=0)

    outdir = '/Users/hugofluhr/chuv/repositories/gsp_neuro/figures/atrophies/'

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    print('>>>> Starting to plot')
    for atr, sub_id in tqdm(zip(atrophy_thickness, atrophy_ids), total=len(atrophy_ids)):
        filename = "sub_{}.png".format(sub_id)
        outpath = os.path.join(outdir, filename)
        fig = viz.plot_signal_surf(roi_values=atr-avg_healthy_thick)
        fig.savefig(outpath, dpi=dpi)

    print('>>>> Done!')

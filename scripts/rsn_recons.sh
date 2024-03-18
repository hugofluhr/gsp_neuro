#!/bin/zsh
#conda activate gsp

#python scripts/rsn_plots.py --filter_type lp --nb_components 20 --scale 5
#python scripts/rsn_plots.py --filter_type lp --nb_components 30 --scale 5
#python scripts/rsn_plots.py --filter_type thresh --nb_components 20 --scale 5
#python scripts/rsn_plots.py --filter_type thresh --nb_components 30 --scale 5

python scripts/rsn_plots.py --bin 0 --scale 5 --rsn_original 1
python scripts/rsn_plots.py --filter_type lp --nb_components 20 --bin 0 --scale 5 #--rsn_original 1
python scripts/rsn_plots.py --filter_type lp --nb_components 50 --bin 0 --scale 5
python scripts/rsn_plots.py --filter_type lp --nb_components 100 --bin 0 --scale 5

python scripts/rsn_plots.py --filter_type thresh --nb_components 20 --bin 0 --scale 5
python scripts/rsn_plots.py --filter_type thresh --nb_components 50 --bin 0 --scale 5
python scripts/rsn_plots.py --filter_type thresh --nb_components 100 --bin 0 --scale 5
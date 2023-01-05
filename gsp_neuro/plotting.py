from matplotlib import pyplot as plt
from gsp_neuro.deps.cmtk_viz import plot_lausanne2018_surface_ctx # to make it accessible through this module

def plot_connectome(C):
    plt.figure(figsize=(8,8))
    plt.matshow(C, fignum=0)
    plt.show()

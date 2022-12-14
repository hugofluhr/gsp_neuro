from matplotlib import pyplot as plt
import numpy as np

def plot_connectome(C):
    plt.figure(figsize=(8,8))
    plt.matshow(C, fignum=0)
    plt.show()

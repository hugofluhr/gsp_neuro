from matplotlib import pyplot as plt

def plot_connectome(C):
    plt.figure(figsize=(8,8))
    plt.matshow(C, fignum=0)
    plt.show()
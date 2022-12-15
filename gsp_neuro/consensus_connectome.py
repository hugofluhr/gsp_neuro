import numpy as np
from scipy.spatial.distance import cdist


def distance_matrix(coordinates):
    """
    Get distance matrix from coordinates.

    Parameters
    ----------
    coordinates :
        x-y-z coordinates

    Returns
    -------
    dist_matrix :
        Distance matrix
    """

    dist_matrix = cdist(coordinates, coordinates)
    return dist_matrix


def fcn_group_bins(adjacencies, distance, nbins, hemiid = None):
    """
    Parameters
    ----------
    adjacencies:
       [node x node x subject] structural connectivity matrices.
    distance:
       [node x node] distance matrix
    hemiid:
       indicator array for left (0) and right (1) hemispheresn
    nbins:
       number of distance bins

    Returns
    -------
    G:
        group matrix (binary) with distance-based consensus
    Gc:
        group matrix (binary) with traditional consistency-based thresholding.
    """

    n, _, nsub = adjacencies.shape
    
    if hemiid is None:
        hemiid = np.zeros((n,))
        hemiid[n//2:] = 1

    min_d = np.min(distance[distance > 0])
    max_d = np.max(distance)
    dist_bins = np.linspace(min_d, max_d, nbins + 1) # equally space distance bins
    dist_bins[-1] = dist_bins[-1] + 1
    consistency = np.sum(adjacencies > 0, axis=2)
    grp, gc = np.zeros((2, n, n, 2))

    for j in range(2):
        # mask for inter- / intra- hemisphere connections
        if j == 0: #intra-hemisphere
            d = np.outer((hemiid == 0), (hemiid.T == 1))
        else: # inter-hemisphere
            d = np.outer((hemiid == 0), (hemiid.T == 0)) | np.outer((hemiid == 1), (hemiid.T == 1))
        d = np.bitwise_or(d, d.T).astype(float)

        m = distance * d # the distances of interest
        # D contains the distances only where a connection exists
        D = ((adjacencies > 0) * (distance * np.triu(d))[:, :, None]).flatten()
        D = D[np.nonzero(D)] # keep only non zero values, 1D array
        tgt = len(D) / nsub  # mean number of connections of interest per subject

        # Distance-based consensus
        g = np.zeros((n, n))
        for ibin in range(nbins): # looping over distance bins
            mask = np.triu((m >= dist_bins[ibin]) & (m < dist_bins[ibin + 1]), 1) #create a mask for distances/edges in current bin
            mask1d = mask.flatten()
            mask1d_indices = np.argwhere(mask1d).flatten() # 1d indices of edges of interest
            frac = int(
                np.round(
                    tgt * np.mean((D >= dist_bins[ibin]) & (D < dist_bins[ibin + 1])) # 
                )
            )
            idx = np.argsort(consistency[mask])
            idx = np.flip(idx)
            g[np.unravel_index(mask1d_indices[idx[:frac]], g.shape)] = 1
        grp[..., j] = g

        # Consistency based thresholding
        av_weight = np.divide(
            np.sum(adjacencies, axis=2),
            consistency,
            out=np.zeros((n, n)),
            where=consistency != 0,
        )
        # indices of current connections of interest
        I = np.where(np.triu(d, 1))
        w = av_weight[I]
        idx = np.flip(np.argsort(w))
        w = np.zeros((n, n))
        # keeping same number of edges as for other method
        idx_weights_to_keep = idx[:np.count_nonzero(g)]
        w[I[0][idx_weights_to_keep], I[1][idx_weights_to_keep]] = 1
        gc[..., j] = w

    G = np.sum(grp, axis=2)
    G = G + G.T
    Gc = np.sum(gc, axis=2)
    Gc = Gc + Gc.T
    return G, Gc


# chatGPT version :
import numpy as np

def chat_GPT_fcn_group_bins(A, dist, nbins, hemiid = None):
    # Create an array of distance bins
    distbins = np.linspace(np.min(np.nonzero(dist)), np.max(np.nonzero(dist)), nbins + 1)
    distbins[-1] += 1

    # Get the number of nodes and subjects
    n, _, nsub = A.shape

    if hemiid is None:
        hemiid = np.zeros((n,))
        hemiid[n//2:] = 1
    # Compute the consistency and average weight matrices
    C = np.sum(A > 0, axis=2)
    W = np.nan_to_num(np.sum(A, axis=2) / C)

    # Initialize the group and consistency matrices
    Grp = np.zeros((n, n, 2))
    Gc = Grp

    for j in range(2):
        # Create the inter- or intra-hemispheric edge mask
        if j == 0:
            d = (hemiid == 0) * (hemiid.T == 1)
            d = d | d.T
        else:
            d = (hemiid == 0) * (hemiid.T == 0) | (hemiid == 1) * (hemiid.T ==1)
            d = d | d.T

        # Compute the distances for the current edge mask
        m = dist * d
        D = np.nonzero(np.multiply(A > 0, np.multiply(dist, np.triu(d))))
        tgt = len(D) / nsub

        # Initialize the group matrix
        G = np.zeros((n, n))

        # Loop over the distance bins
        for ibin in range(nbins):
            # Find the edges that fall within the current bin
            mask = np.nonzero(np.triu(m >= distbins[ibin] & m < distbins[ibin + 1], 1))
            frac = round(tgt * np.sum(D >= distbins[ibin] & D < distbins[ibin + 1]) / len(D))

            # Sort the edges by consistency and select the top "frac" edges
            c = C[mask]
            idx = np.argsort(c)[::-1]
            G[mask[idx[:frac]]] = 1

        # Store the group matrix for the current hemisphere
        Grp[:, :, j] = G

        # Compute the consistency-based group matrix for the current hemisphere
        I = np.nonzero(np.triu(d, 1))
        w = W[I]
        idx = np.argsort(w)[::-1]
        w = np.zeros((n, n))
        w[I[idx[:np.count_nonzero(G)]]] = 1
        Gc[:, :, j] = w

    # Compute the final group and consistency matrices
    G = np.sum(Grp, axis=2)
    G = G + G.T
    Gc = np.sum(Gc, axis=2)
    Gc = Gc + Gc.T

    return G, Gc

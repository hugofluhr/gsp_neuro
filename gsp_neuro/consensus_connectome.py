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
    dist_bins = np.linspace(min_d, max_d, nbins + 1)
    dist_bins[-1] = dist_bins[-1] + 1
    consistency = np.sum(adjacencies > 0, axis=2)
    grp, gc = np.zeros((2, n, n, 2))

    for j in range(2):
        # mask for inter- / intra- hemisphere connections
        if j == 0:
            d = np.outer((hemiid == 0),(hemiid.T == 1))
        else:
            d = np.outer((hemiid == 0), (hemiid.T == 0)) | np.outer((hemiid == 1), (hemiid.T == 1))
        d = np.bitwise_or(d, d.T).astype(float)

        m = distance * d # the distances of interest
        # D contains the distances only where a connection exists
        D = ((adjacencies > 0) * (distance * np.triu(d))[:, :, None]).flatten()
        D = D[np.nonzero(D)] # keep only non zero values, 1D array
        tgt = len(D) / nsub  # to keep same amout for all subjects
        # Distance-based consensus
        
        g = np.zeros((n, n))
        for ibin in range(nbins):
            mask = np.triu((m >= dist_bins[ibin]) & (m < dist_bins[ibin + 1]), 1)
            mask1d = mask.flatten()
            mask1d_indices = np.argwhere(mask1d).flatten()
            frac = int(
                np.round(
                    tgt * np.mean((D >= dist_bins[ibin]) & (D < dist_bins[ibin + 1]))
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

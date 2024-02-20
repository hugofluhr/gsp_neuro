import numpy as np
from scipy.spatial.distance import pdist, squareform


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

    dist_matrix = squareform(pdist(coordinates))
    return dist_matrix


def fcn_group_bins(adjacencies, distance, nbins, hemiid=None):
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
        hemiid[n // 2 :] = 1

    min_d = np.min(distance[distance > 0])
    max_d = np.max(distance)
    dist_bins = np.linspace(min_d, max_d, nbins + 1)  # equally space distance bins
    dist_bins[-1] = dist_bins[-1] + 1
    consistency = np.sum(adjacencies > 0, axis=2)
    grp, gc = np.zeros((2, n, n, 2))

    # compute mean weight of each edge, mean using consistency (ignores 0 edges)
    av_weight = np.divide(
        np.sum(adjacencies, axis=2),
        consistency,
        out=np.zeros((n, n)),
        where=consistency != 0,
    )

    for j in range(2):
        # mask for inter- / intra- hemisphere connections
        if j == 0:  # intra-hemisphere
            d = np.outer((hemiid == 0), (hemiid.T == 1))
        else:  # inter-hemisphere
            d = np.outer((hemiid == 0), (hemiid.T == 0)) | np.outer(
                (hemiid == 1), (hemiid.T == 1)
            )
        d = np.bitwise_or(d, d.T).astype(float)

        m = distance * d  # the distances of interest
        # D contains the distances only where a connection exists
        D = ((adjacencies > 0) * (distance * np.triu(d))[:, :, None]).flatten()
        D = D[np.nonzero(D)]  # keep only non zero values, 1D array
        tgt = len(D) / nsub  # mean number of connections of interest per subject

        # Distance-based consensus
        g = np.zeros((n, n))
        for ibin in range(nbins):  # looping over distance bins
            mask = np.triu(
                (m >= dist_bins[ibin]) & (m < dist_bins[ibin + 1]), 1
            )  # create a mask for distances/edges in current bin
            mask1d = mask.flatten()
            mask1d_indices = np.argwhere(
                mask1d
            ).flatten()  # 1d indices of edges of interest in current bin

            # compute how many edges to keep for this distance bin
            # is equal to mean nb of connection per subject multiplied by fraction of distances in current bin
            nb_edges2keep_in_bin = int(
                np.round(
                    tgt * np.mean((D >= dist_bins[ibin]) & (D < dist_bins[ibin + 1]))  #
                )
            )

            # keep nb_edges2keep_in_bin edges with highest consistency across subjects
            idx = np.argsort(consistency[mask])
            idx = np.flip(idx)
            g[np.unravel_index(mask1d_indices[idx[:nb_edges2keep_in_bin]], g.shape)] = 1
        grp[..., j] = g


        # Traditional Consistency based thresholding

        # indices of current connections of interest
        I = np.where(np.triu(d, 1))
        w = av_weight[I]
        idx = np.flip(np.argsort(w))
        w = np.zeros((n, n))
        # keeping same number of edges as for other method
        idx_weights_to_keep = idx[: np.count_nonzero(g)]
        w[I[0][idx_weights_to_keep], I[1][idx_weights_to_keep]] = 1
        gc[..., j] = w

    G = np.sum(grp, axis=2)
    G = G + G.T
    Gc = np.sum(gc, axis=2)
    Gc = Gc + Gc.T
    return G, Gc, av_weight

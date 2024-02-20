# Copyright (C) 2009-2022, Ecole Polytechnique Federale de Lausanne (EPFL) and
# Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland, and CMP3 contributors
# All rights reserved.
#
#  This software is distributed under the open-source license Modified BSD.

"""Module that defines CMTK utility functions for retrieving Lausanne parcellation files."""

import os


def get_lausanne2018_parcellation_annot(scale=1, hemi='lh'):
    """Return the path of the Freesurfer ``.annot`` file corresponding to a specific scale and hemisphere.

    Parameters
    ----------
    scale : {1, 2, 3, 4, 5}
        Lausanne 2018 parcellation scale

    hemi : {'lh', 'rh'}
        Brain hemisphere

    Returns
    -------
    annot_file_path : string
        Absolute path to the queried ``.annot`` file

    """
    annot_path = "/Users/hugofluhr/chuv/data/requestionforconnectomes"
    query_annot_file = os.path.join(annot_path,
                                    f'{hemi}.atlas-laus2018_desc-scale{scale}.annot')
    
    return query_annot_file

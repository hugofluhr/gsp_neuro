from pygsp import graphs
from gsp_neuro import data_loading as dload

class Brain:
    def __init__(self, data_path, sub_ID, scale, field = 'fibDensity') -> None:
        self.data_path = data_path
        self.scale = scale
        self.subject_dir = dload.get_subject_dir(data_path, sub_ID)
        self.sub_ID = sub_ID
        self.path = dload.get_sub_connectomes_paths(self.subject_dir, self.scale)[0]
        self.field = field
        self.signals = {}

    def load_graph(self, lap_type = 'normalized', keep_sub_cortical = False):

        if self.sub_ID == 'consensus':
            self.G = graphs.Graph(dload.load_consensus(self.path), lap_type=lap_type)
        else:
            self.G = graphs.Graph(dload.load_connectome(self.path, field=self.field), lap_type=lap_type)
    
        # node coordinates :
        df_coords = dload.read_coords(scale=self.scale)
        self.G.set_coordinates(kind=df_coords[['XCoord(mm)','YCoord(mm)','ZCoord(mm)']].to_numpy())
        self.rois = list(df_coords[df_coords["Structures Names"].str.contains('ctx')]["Structures Names"].str.replace('ctx-',''))
        rois_ids = list(df_coords[df_coords["Structures Names"].str.contains('ctx')].index)
        self.G = self.G.subgraph(rois_ids)
        self.adjacency = self.G.W.toarray()
        self.G.compute_fourier_basis()

    def add_signal(self, signal_type):
        # TO DO can give dict as input 
        self.signals.update({signal_type : dload.get_signal(self.data_path, self.sub_ID, 
                                                signal_type, self.scale)})


class Subject:
    def __init__(self) -> None:
        pass
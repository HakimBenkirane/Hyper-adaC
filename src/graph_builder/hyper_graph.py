import pickle
import torch
import numpy as np
from wsi_handler.wsi import WholeSlideImage
from scipy.spatial import distance
from torch_geometric.data import Data

class HyperGraph:
    def __init__(self, wsi_path, h5_path, features_path, sample_name, extension='.svs', 
                    lambda_d=3e-3, lambda_f=1e-3, lambda_h=0.8):
        self.wsi = WholeSlideImage(wsi_path=wsi_path, sample_name=sample_name, file_extension=extension)
        self.wsi.load_patches(h5_path=h5_path)
        self.wsi.load_features(features_path=features_path)
        self.wsi.filter_features(lambda_d = lambda_d, lambda_f = lambda_f, lambda_h = lambda_h)
        self.features = self.wsi.features
        self.coords = self.wsi.coords
        self.sample_name = sample_name
        self._construct_incidence()
        self._construct_edge_index()
        self._to_geometric()

    def _construct_incidence(self, treshold=0.8, lambda_h=0.5):
        distance_matrix = np.exp(-lambda_h * distance.cdist(self.features, self.features))
        self.incidence_matrix = (distance_matrix >= treshold) * np.ones(distance_matrix.shape)
        self.incidence_matrix = torch.Tensor(self.incidence_matrix)

    def _construct_edge_index(self):
        self.edge_index = self.incidence_matrix.nonzero().t().contiguous()
        x = torch.tensor(self.edge_index)
        index = torch.LongTensor([1, 0])
        y = torch.zeros_like(x)
        y[index] = x
        self.edge_index = y

    def _to_geometric(self):
        self.graph = Data(x=self.features, edge_index=self.edge_index, edge_attr=self.features, pos=self.coords)

    def save_hypergraph(self, path):
        with open(path + self.sample_name + '.pkl', 'wb+') as graph_file:
            pickle.dump(self.graph, graph_file)
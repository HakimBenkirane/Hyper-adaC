import torch
import math
import torch.nn as nn
from src.graph_model.gcnn import GNN
from torch_geometric.utils import to_dense_batch
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class DeepSurv(nn.Module):
    ''' The module class performs building network according to config'''
    def __init__(self, config):
        super(DeepSurv, self).__init__()
        # parses parameters of network from configuration
        self.drop = config['drop']
        self.norm = config['norm']
        self.dims = config['dims']
        self.activation = config['activation']
        # builds network
        self.model = self._build_network()

    def _build_network(self):
        ''' Performs building networks according to parameters'''
        layers = []
        for i in range(len(self.dims)-1):
            if i and self.drop is not None: # adds dropout layer
                layers.append(nn.Dropout(self.drop))
            # adds linear layer
            layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            if self.norm: # adds batchnormalize layer
                layers.append(nn.BatchNorm1d(self.dims[i+1]))
            # adds activation layer
            layers.append(eval('nn.{}()'.format(self.activation)))
        # builds sequential network
        return nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)




class GNN_Surv(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        
        self.encoder = GNN(dim_features=1024, dim_target=output_dim)
        self.survival_config = {'drop': 0.16, 'norm': True, 'dims': [output_dim, 256, 1], 'activation': 'SELU', 'l2_reg':1e-2}
        self.classifier = DeepSurv(self.survival_config)


    def forward(self, data_wsi, return_attn=False):
        x, attn_scores = self.encoder(data_wsi)
        #x = to_dense_batch(x, data_wsi.batch)[0]
        attn_scores = to_dense_batch(attn_scores, data_wsi.batch)[0].cpu().detach().numpy()
        for i in range(attn_scores.shape[0]):
            attn_scores[i] = np.apply_along_axis(lambda x:x, axis=1, arr=MinMaxScaler().fit_transform(attn_scores[i]))
            attn_scores[i] = MinMaxScaler().fit_transform(attn_scores[i])
        coords = to_dense_batch(data_wsi.coords, data_wsi.batch)[0]
        risk_scores = self.classifier(x)
        if return_attn:
            return torch.sigmoid(risk_scores), attn_scores, coords
        else:
            return torch.sigmoid(risk_scores)





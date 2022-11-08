import torch
import numpy as np
import torch.nn as nn
import pdb
import pandas as pd
from os import listdir
from os.path import isfile, join

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
#from src.datasets import HyperGraphDataset
from wsi_handler.datasets import GraphSurvivalDataset
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index



device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_splits(filename):
    splits = pd.read_csv(filename, sep=',')
    return splits['train'].values, splits['val'].dropna(axis=0).values

def reindex_df(df):
    df = df.rename(index=lambda s: s[:-4])
    return df.groupby(df.index).first()


def prepare_data(cohort_name, split_num, ruche=False, training='pretrained', graph_type='adaptative'):
    if ruche:
        survival_path = '../../shared/datasets/pathomics-tcga/{}/Survival/{}.survival.tsv'.format(cohort_name, cohort_name)
    else:
        survival_path = '../TCGA/{}/Survival/{}.survival.tsv'.format(cohort_name, cohort_name)
    if training == 'pretrained':
        if graph_type == 'knn':
            graph_path = '../TCGA/{}/WSI_preprocess/pretrained/knn_graphs/'.format(cohort_name)
        elif graph_type == 'sample':
            graph_path = '../TCGA/{}/WSI_preprocess/pretrained/sample_graphs/'.format(cohort_name)
        elif graph_type == 'cluster':
            graph_path = '../TCGA/{}/WSI_preprocess/pretrained/cluster_graphs/'.format(cohort_name)
        elif graph_type == 'adaptative':
            graph_path = '../TCGA/{}/WSI_preprocess/pretrained/adaptative_graphs/'.format(cohort_name)
        elif graph_type == 'Hyper_adaC':
            graph_path = '../TCGA/{}/WSI_preprocess/pretrained/Hyper_adaC/'.format(cohort_name)
        else:
            raise ValueError('Graph type not recognized !')
    elif training == 'contrastive':
        if graph_type == 'knn':
            graph_path = '../TCGA/{}/WSI_preprocess/contrastive/knn_graphs/'.format(cohort_name)
        elif graph_type == 'sample':
            graph_path = '../TCGA/{}/WSI_preprocess/contrastive/sample_graphs/'.format(cohort_name)
        elif graph_type == 'cluster':
            graph_path = '../TCGA/{}/WSI_preprocess/contrastive/cluster_graphs/'.format(cohort_name)
        elif graph_type == 'adaptative':
            graph_path = '../TCGA/{}/WSI_preprocess/contrastive/adaptative_graphs/'.format(cohort_name)
        elif graph_type == 'Hyper_adaC':
            graph_path = '../TCGA/{}/WSI_preprocess/contrastive/Hyper_adaC/'.format(cohort_name)
        else:
            raise ValueError('Graph type not recognized !')
    else:
        raise ValueError('Training mode not recognized !')

    patients_train, patients_test = extract_splits('5foldcv/{}/splits_{}.csv'.format(cohort_name, split_num))

    survival_df = pd.read_csv(survival_path, sep='\t', index_col=0, header=0)
    survival_df = reindex_df(survival_df)


    sample_list = [f[:-4] for f in listdir(graph_path) if isfile(join(graph_path, f))]
    real_sample_list_train = []
    real_sample_list_test = []
    used_patients = []
    for sample_name in sample_list:
        if sample_name[:12] in survival_df.index:
            if sample_name[:12] in patients_train:
                real_sample_list_train.append(sample_name)
            elif sample_name[:12] in patients_test:
                real_sample_list_test.append(sample_name)
            used_patients.append(sample_name[:12])



    dataset_train = GraphSurvivalDataset(graph_path, real_sample_list_train, survival_df, 'OS.time', 'OS')
    dataset_test = GraphSurvivalDataset(graph_path, real_sample_list_test, survival_df, 'OS.time', 'OS')


    return dataset_train, dataset_test, real_sample_list_test


def get_optim(model, optimizer='adam', lr=1e-3, reg=1e-2):
    if optimizer == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=reg)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=reg)
    else:
        raise NotImplementedError
    return optimizer

def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  

    return torch.DoubleTensor(weight)

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


################
# Survival Utils
################


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



def CoxLoss(survtime, censor, hazard_pred, device):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_cox(hazardsdata, labels):
    # This accuracy is based on estimated survival events against true survival events
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)


def CIndex(hazards, labels, survtime_all):
    concord = 0.
    total = 0.
    N_test = labels.shape[0]
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total += 1
                    if hazards[j] < hazards[i]: concord += 1
                    elif hazards[j] < hazards[i]: concord += 0.5

    return(concord/total)


def CIndex_lifeline(hazards, labels, survtime_all):
    return(concordance_index(survtime_all, -hazards, labels))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name:str='checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)


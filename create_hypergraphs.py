import torch
from os import listdir
import tqdm
import argparse

from src.graph_builder.hyper_graph import HyperGraph

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cohorts', help='list of cohorts to process', type=str)
parser.add_argument('-wp', '--wsi_path', help='directory containing the WSI files', type=str)
parser.add_argument('-hp', '--h5_path', help='directory containing the h5 files storing the coordinates', type=str)
parser.add_argument('-fp', '--features_path', help='directory containing the features pt files', type=str)
parser.add_argument('-hp', '--h5_path', help='directory containing the h5 files storing the coordinates', type=str)
parser.add_argument('-hyp', '--hyper_path', help='directory in which to store the hypergraphs', type=str)

parser.add_argument('-ld', '--lambda_d', help='Parameter for the morphological similarity kernel', type=float, default=3e-3)
parser.add_argument('-lf', '--lambda_f', help='Parameter for the spatial proximity kernel', type=float, default=1e-3)
parser.add_argument('-lh', '--lambda_h', help='THreshold parameter for the agglomerative clustering', type=float, default=0.8)



args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

COHORTS = [item for item in args.cohorts.split(',')]
sample_list = [f[:-4] for f in listdir(args.wsi_path)]

for sample_name in tqdm.tqdm(sample_list):
    hyper_graph = HyperGraph(wsi_path=args.wsi_path, h5_path=args.h5_path, features_path=args.features_path,
                                lambda_d=args.lambda_d, lambda_f=args.lambda_f, lambda_h=args.lambda_h)
    hyper_graph.save_hypergraph(path=args.hyper_path)

    

from src.tools import train, prepare_data
from wsi_handler.wsi import WholeSlideImage
import numpy as np
import torch
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cohorts', help='list of cohorts to process', type=str)
parser.add_argument('-wp', '--wsi_path', help='directory containing the WSI files', type=str)
parser.add_argument('-hp', '--h5_path', help='directory containing the h5 files storing the coordinates', type=str)


args = parser.parse_args()

torch.manual_seed(123)

COHORTS = [item for item in args.cohorts.split(',')]

for cohort in COHORTS:
    print('------------Training Start for {}--------------'.format(cohort))
    c_index = []
    for i in range(5):
        dataset_train, dataset_val, samples_val = prepare_data(cohort, i, graph_type='adaptative', training='pretrained')
        val_index, risk_scores, attn_scores, coords = train((dataset_train, dataset_val), i, n_epochs=10, opt_name='adam', lr=1e-2, reg=1e-2)
        c_index.append(val_index)
        for sample, val_attn, val_coords in zip(samples_val, attn_scores, coords):
            wsi = WholeSlideImage(wsi_path=args.wsi_path, sample_name=sample)
            wsi.apply_heatmap_on_slide(val_attn[0], val_coords[0], sacling_factor=10)
    print('C-index on folds : {} +- {}'.format(np.mean(c_index), np.std(c_index)))
from torch.utils.data import Dataset
import pickle
import torch


class HyperGraphDataset(Dataset):
    def __init__(self, graph_path, lt_samples, survival_df):
        self.graph_path = graph_path
        self.lt_samples = lt_samples
        self.survival_df = survival_df

    def __len__(self):
        return len(self.lt_samples)


    def __getitem__(self, idx):
        sample = self.lt_samples[idx]
        with open(self.graph_path + sample + '.pkl', 'rb') as f:
            graph = pickle.load(f)
        os_time = float(self.survival_df.loc[sample[:12], 'OS.time'])
        os_event = float(self.survival_df.loc[sample[:12], 'OS'])

        return graph, os_time, os_event
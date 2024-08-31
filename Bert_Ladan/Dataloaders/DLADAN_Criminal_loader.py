import sys
sys.path.append('..')
import json, random
import os
from torch.utils.data import DataLoader, RandomSampler, Dataset
from transformers.tokenization_utils_base import BatchEncoding
import torch
import pickle as pkl
from Formatters import SentenceFormatter_D
from tqdm import tqdm
import ipdb
from law_processing import get_law_graph_adj, get_law_graph_large_adj


class DladanCriminalLoader(Dataset):
    def __init__(self, data_path, group_indexes, mode, *args, **kwargs) -> None:
        self.mode = mode
        self.group_indexes = group_indexes
        self.data = []
        data_dict = pkl.load(open(data_path, 'rb'))
        data_list = list(zip(*[list(v) for v in data_dict.values()]))
        for data in tqdm(data_list, ncols=80):
            input_ids = data[0]
            token_type_ids = data[1]
            attention_mask = data[2]
            accu_label = data[3]
            group_label = self.group_indexes[accu_label]
            self.data.append({
                "inputx": input_ids, 
                'mask': attention_mask,
                'segment': token_type_ids,
                'accu_label': accu_label,
                'group_label': group_label
                })
            
    def __getitem__(self, item):
        data = self.data[item % len(self.data)]
        return data
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.data)
        else:
            return len(self.data)

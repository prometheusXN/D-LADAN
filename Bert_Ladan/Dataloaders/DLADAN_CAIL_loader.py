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


class DladanCailLoader(Dataset):
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
            law_label = data[3]
            accu_label = data[4]
            time_label = data[5]
            group_label = self.group_indexes[law_label]
            self.data.append({
                "inputx": input_ids, 
                'mask': attention_mask,
                'segment': token_type_ids,
                'law_label': law_label,
                'accu_label': accu_label,
                'time_label': time_label,
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
        
        
class DladanCailLoader_W2V(Dataset):
    def __init__(self, data_path, group_indexes, mode, *args, **kwargs) -> None:
        self.mode = mode
        self.group_indexes = group_indexes
        self.data = []
        data_dict = pkl.load(open(data_path, 'rb'))
        data_list = list(zip(*[list(v) for v in data_dict.values()]))
        for data in tqdm(data_list, ncols=80):
            input_ids = data[0]
            law_label = data[1]
            accu_label = data[2]
            time_label = data[3]
            group_label = self.group_indexes[law_label]
            self.data.append({
                "inputx": input_ids, 
                'law_label': law_label,
                'accu_label': accu_label,
                'time_label': time_label,
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
            
        

if __name__ == "__main__":
    law_index_matrix, graph_list_1, graph_membership, adj_matrix = get_law_graph_adj(threshold=0.35)
    group_indexes = list(zip(*graph_membership))[1]
    test_path = '/home/nxu/Ladan_tnnls/Bert_Ladan/processed_dataset/CAIL_new/full_doc/small/test_bert_chinese.pkl'   
    test_Dataset = DladanCailLoader(test_path, group_indexes, mode='test')
    
    Formatter = SentenceFormatter_D(mode='train')
    
    def collate_fn(data):
        return Formatter.process(data, "train")

    test_dataset = DataLoader(dataset=test_Dataset, batch_size=2,
                               num_workers=1, drop_last=True,
                               shuffle=True, collate_fn=collate_fn,
                               # sampler=RandomSampler(Dataset)
                               )
    
    print(len(test_dataset))
    
    for step, data in tqdm(enumerate(test_dataset), total=len(test_dataset), position=0, leave=True):
        print(step)
        print(data)
        print(data['inputx'])
        inputx = data['inputx']
        ipdb.set_trace()
        break

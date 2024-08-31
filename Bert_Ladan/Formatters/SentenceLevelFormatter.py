import torch
import pickle as pkl
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import numpy as np


class SentenceFormatter:
    def __init__(self, mode, *args, **params):
        print("Records Bert input form")
        self.mode = mode
    
    def process(self, data, *args, **kwargs):
        inputx = []
        masks = []
        segments = []
        law_labels = []
        accu_labels = []
        term_labels = []
        for temp in data:
            inputx.append(temp['inputx'])
            masks.append(temp['mask'])
            segments.append(temp['segment'])
            law_labels.append(int(temp['law_label']))
            accu_labels.append(int(temp['accu_label']))
            term_labels.append(int(temp['term_label']))
        
        inputx = torch.LongTensor(inputx)
        masks = torch.LongTensor(masks)
        segments = torch.LongTensor(segments)
        law_label = torch.LongTensor(law_labels)
        accu_label = torch.LongTensor(accu_labels)
        term_label = torch.LongTensor(term_labels)
        
        res = {
            'inputx': inputx,
            'masks': masks,
            'segments': segments,
            'law_label': law_label,
            'accu_label': accu_label,
            'term_label': term_label,
        }
        return res
    
    
class SentenceFormatter_C:
    def __init__(self, mode, *args, **params):
        print("Records Bert input form")
        self.mode = mode
    
    def process(self, data, *args, **kwargs):
        inputx = []
        masks = []
        segments = []
        accu_labels = []
        for temp in data:
            inputx.append(temp['inputx'])
            masks.append(temp['mask'])
            segments.append(temp['segment'])
            accu_labels.append(int(temp['accu_label']))
        
        inputx = torch.LongTensor(inputx)
        masks = torch.LongTensor(masks)
        segments = torch.LongTensor(segments)
        accu_label = torch.LongTensor(accu_labels)
        
        res = {
            'inputx': inputx,
            'masks': masks,
            'segments': segments,
            'accu_label': accu_label,
        }
        return res
    

class SentenceFormatter_D:
    def __init__(self, mode, *args, **params):
        print("Records Bert input form")
        self.mode = mode
    
    def process(self, data, *args, **kwargs):
        inputx = []
        masks = []
        segments = []
        law_labels = []
        accu_labels = []
        term_labels = []
        group_labels = []
        for temp in data:
            inputx.append(temp['inputx'])
            masks.append(temp['mask'])
            segments.append(temp['segment'])
            law_labels.append(int(temp['law_label']))
            accu_labels.append(int(temp['accu_label']))
            term_labels.append(int(temp['time_label']))
            group_labels.append(int(temp['group_label']))
        
        inputx = torch.LongTensor(inputx)
        masks = torch.LongTensor(masks)
        segments = torch.LongTensor(segments)
        law_label = torch.LongTensor(law_labels)
        accu_label = torch.LongTensor(accu_labels)
        term_label = torch.LongTensor(term_labels)
        group_label = torch.LongTensor(group_labels)
        
        res = {
            'inputx': inputx,
            'masks': masks,
            'segments': segments,
            'law_label': law_label,
            'accu_label': accu_label,
            'time_label': term_label,
            'group_label': group_label
        }
        return res


class SentenceFormatter_DC:
    def __init__(self, mode, *args, **params):
        print("Records Bert input form")
        self.mode = mode
    
    def process(self, data, *args, **kwargs):
        inputx = []
        masks = []
        segments = []
        accu_labels = []
        group_labels = []
        for temp in data:
            inputx.append(temp['inputx'])
            masks.append(temp['mask'])
            segments.append(temp['segment'])
            accu_labels.append(int(temp['accu_label']))
            group_labels.append(int(temp['group_label']))
        
        inputx = torch.LongTensor(inputx)
        masks = torch.LongTensor(masks)
        segments = torch.LongTensor(segments)
        accu_label = torch.LongTensor(accu_labels)
        group_label = torch.LongTensor(group_labels)
        
        res = {
            'inputx': inputx,
            'masks': masks,
            'segments': segments,
            'accu_label': accu_label,
            'group_label': group_label
        }
        return res
    

class SentenceFormatter_W2V:
    def __init__(self, mode, *args, **params):
        print("Records Bert input form")
        self.mode = mode
    
    def process(self, data, *args, **kwargs):
        inputx = []
        law_labels = []
        accu_labels = []
        term_labels = []
        group_labels = []
        for temp in data:
            inputx.append(temp['inputx'])
            law_labels.append(int(temp['law_label']))
            accu_labels.append(int(temp['accu_label']))
            term_labels.append(int(temp['time_label']))
            group_labels.append(int(temp['group_label']))
        
        inputx = torch.LongTensor(inputx)
        law_label = torch.LongTensor(law_labels)
        accu_label = torch.LongTensor(accu_labels)
        term_label = torch.LongTensor(term_labels)
        group_label = torch.LongTensor(group_labels)
        
        res = {
            'inputx': inputx,
            'law_label': law_label,
            'accu_label': accu_label,
            'time_label': term_label,
            'group_label': group_label
        }
        return res
    

def collate_fn_fact(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch]).squeeze(1)
    token_type_ids = torch.stack([x['token_type_ids'] for x in batch]).squeeze(1)
    attention_mask = torch.stack([x['attention_mask'] for x in batch]).squeeze(1)
    masked_lm_labels = torch.stack([x['masked_lm_labels'] for x in batch]).squeeze(1)
    accu = [x['accu'] for x in batch]
    accu = torch.from_numpy(np.array(accu))
    law = [x['law'] for x in batch]
    law = torch.from_numpy(np.array(law))
    term = [x['term'] for x in batch]
    term = torch.from_numpy(np.array(term))
    res = {
        'inputx': input_ids,
        'masks': attention_mask,
        'segments': token_type_ids,
        'law_label': law,
        'accu_label': accu,
        'term_label': term,
    }
    return res

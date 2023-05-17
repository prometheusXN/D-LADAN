from torch.utils.data import Dataset, DataLoader
import json
import torch
import pickle as pk
import numpy as np
import random
from tqdm import tqdm
import time
from Config.parser import ConfigParser


class CAIL_dataset(Dataset):

    def __init__(self, dataset, law_charge_case_dict, CL4article_dict, law_num=103, neg_num=2, mode='train'):
        """
        :param dataset:
        :param law_charge_case_dict: {(int)law: {(int)charge: [(int)case_index]}}
        :param CL4article_dict: {law: {neg_law}}
        """
        self.mode = mode
        self.neg_num = neg_num
        self.law_num = law_num
        self.law_charge_case_dict = law_charge_case_dict
        fact_description = dataset['fact_list']
        case_index_list = dataset['index_list']
        law_label_lists = dataset['law_label_lists']
        accu_label_list = dataset['accu_label_lists']
        time_label_list = dataset['term_lists']

        data_input = list(zip(case_index_list, fact_description, law_label_lists, accu_label_list, time_label_list))
        self.CL4charge_dict = {}
        self.CL4article_dict = CL4article_dict
        self.data = []
        for data in tqdm(data_input):
            charge_pos = []
            charge_neg = []
            case_index, fact, law_label, accu_label, time_label = data
            for charges in self.law_charge_case_dict[str(law_label)].keys():
                if charges == str(accu_label):
                    charge_pos.append(charges)
                else:
                    charge_neg.append(charges)

            self.CL4charge_dict[str(case_index)] = {'pos': charge_pos, 'neg': charge_neg}
            neg_index_articles = self.CL4article_dict[str(law_label)]
            if len(neg_index_articles) == 0:
                article_neg_mask = 0
            else:
                article_neg_mask = 1

            if len(charge_neg) == 0:
                charge_neg_mask = 0
            else:
                charge_neg_mask = 1

            self.data.append({
                'case_index': case_index,
                'fact': fact,
                'law_label': law_label,
                'accu_label': accu_label,
                'time_label': time_label,
                'pos_article_idx': law_label,
                'neg_article_idx': [],
                'pos_charge_fact': [],
                'neg_charge_fact': [],
                'neg_charge_mask': charge_neg_mask,
                'neg_article_mask': article_neg_mask
            })
        del data_input
        self.ori_data = self.data.copy()
        self.shuffle_neg()

    def shuffle_neg(self):
        print('Now Generating CL instances...')
        for index in tqdm(range(len(self.data))):
            # print('\n')
            # t1 = time.time()
            law_label = self.data[index]['law_label']
            case_index = self.data[index]['case_index']
            accu_label = self.data[index]['accu_label']
            fact = self.data[index]['fact']
            neg_article_index: list = self.CL4article_dict[str(law_label)].copy()
            random.shuffle(neg_article_index)

            # generate negative sample of articles
            if len(neg_article_index) == 0:
                self.data[index]['neg_article_idx'] = [0 for i in range(self.neg_num)]
            elif len(neg_article_index) == 1:
                self.data[index]['neg_article_idx'] = [int(neg_article_index[0]) for i in range(self.neg_num)]
            else:
                self.data[index]['neg_article_idx'] = [int(neg_article_index[i]) for i in range(self.neg_num)]
            if len(self.data[index]['neg_article_idx']) < self.neg_num:
                print(index, neg_article_index)

            # generate positive sample of same article and charge
            pos_indexes = self.law_charge_case_dict[str(law_label)][str(accu_label)]
            pos_num = len(pos_indexes)

            if len(pos_indexes) == 1:
                self.data[index]['pos_charge_fact'] = fact
            else:
                rand_idx_pos = random.randint(0, pos_num - 1)
                while int(pos_indexes[rand_idx_pos]) == int(case_index):
                    rand_idx_pos = random.randint(0, pos_num - 1)
                    # shuffle a random index from the pos samples
                self.data[index]['pos_charge_fact'] = self.ori_data[int(pos_indexes[rand_idx_pos])]['fact']

            # generate negative sample of same article but different charges
            neg_charges = self.CL4charge_dict[str(case_index)]['neg']
            neg_indexes = []
            for charge in neg_charges:
                neg_indexes += self.law_charge_case_dict[str(law_label)][str(charge)]
            neg_num = len(neg_indexes)

            if len(neg_indexes) == 0:
                self.data[index]['neg_charge_fact'] = \
                    [np.zeros_like(fact, dtype=np.int), np.zeros_like(fact, dtype=np.int)]
            elif len(neg_indexes) == 1:
                self.data[index]['neg_charge_fact'] = \
                    [self.ori_data[neg_indexes[0]]['fact'], self.ori_data[neg_indexes[0]]['fact']]
            else:
                self.data[index]['neg_charge_fact'] = []
                neg_selected = []
                for i in range(self.neg_num):
                    rand_idx_neg = random.randint(0, neg_num - 1)
                    while rand_idx_neg in neg_selected:
                        rand_idx_neg = random.randint(0, neg_num - 1)
                    self.data[index]['neg_charge_fact'].append(self.ori_data[int(neg_indexes[rand_idx_neg])]['fact'])
                    neg_selected.append(rand_idx_neg)
                    # shuffle the negative num's cases from the negative sample set

    def __getitem__(self, item):
        data = self.data[item % len(self.data)]
        return data

    def __len__(self):
        if self.mode == "train":
            return len(self.data)
        else:
            return len(self.data)


class CAILFormatter:
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode

    def process(self, data, mode='train'):
        fact_list = []
        law_labels = []
        accu_labels = []
        time_labels = []
        pos_article_idx = []
        neg_article_idx = []
        pos_charge_fact = []
        neg_charge_fact = []
        neg_article_masks = []
        neg_charge_masks = []
        for each in data:
            fact_list.append(each['fact'])
            law_labels.append(each['law_label'])
            accu_labels.append(each['accu_label'])
            time_labels.append(each['time_label'])
            pos_article_idx.append(each['pos_article_idx'])
            neg_article_idx.append(each['neg_article_idx'])
            pos_charge_fact.append(each['pos_charge_fact'])
            neg_charge_fact.append(each['neg_charge_fact'])
            neg_article_masks.append(each['neg_article_mask'])
            neg_charge_masks.append(each['neg_charge_mask'])

        ter = {
            'fact': torch.LongTensor(fact_list),    # [batch_size, sentence_len]
            'law_label': torch.LongTensor(law_labels),  # [batch_size, ]
            'accu_label': torch.LongTensor(accu_labels),
            'time_label': torch.LongTensor(time_labels),
            'pos_article_index': torch.LongTensor(pos_article_idx),
            'neg_article_index': torch.LongTensor(neg_article_idx),
            'pos_charge_fact': torch.LongTensor(pos_charge_fact),
            'neg_charge_fact': torch.LongTensor(neg_charge_fact),
            'neg_article_masks': torch.LongTensor(neg_article_masks),
            'neg_charge_masks': torch.LongTensor(neg_charge_masks)
        }
        return ter


class Criminal_dataset(Dataset):
    def __init__(self, dataset, charge_num=149, mode='train'):
        self.mode = mode
        self.charge_num = charge_num
        fact_description = dataset['fact_index']
        charge_label_list = dataset['charge_label_list']
        data_input = list(zip(fact_description, charge_label_list))
        self.data = []
        for data in tqdm(data_input):
            fact, charge_label = data
            self.data.append({
                'fact': fact,
                'charge_label': charge_label,
            })

    def __getitem__(self, item):
        data = self.data[item % len(self.data)]
        return data

    def __len__(self):
        if self.mode == "train":
            return len(self.data)
        else:
            return len(self.data)


class CriminalFormatter:
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode

    def process(self, data, mode='train'):
        fact_list = []
        charge_labels = []

        for each in data:
            fact_list.append(each['fact'])
            charge_labels.append(each['charge_label'])

        ter = {
            'fact': torch.LongTensor(fact_list),    # [batch_size, sentence_len]
            'charge_label': torch.LongTensor(charge_labels)
        }
        return ter


def get_necessary_files(path, data_version, mode):
    data_path = path + data_version + '/{}_processed_thulac.pkl'.format(mode)
    law_charge_case_path = path + data_version + '/{}_law_charge_case.json'.format(mode)
    law2neg_path = path + data_version + '/law2neg.json'
    out_path = path + data_version + '/{}_dataset.pkl'.format(mode)
    return data_path, law_charge_case_path, law2neg_path, out_path


if __name__ == "__main__":

    PATHNAME = '/home/nxu/Ladan_tnnls/CL4LJP/processed_data/CAIL/'
    data_versions = ['data', 'big_data']
    modes = {'data': ['train', 'valid', 'test'],
             'big_data': ['train', 'test']}

    # data_versions = ['data']
    # modes = {'data': ['train'],
    #          'big_data': ['train', 'test']}

    dataset = None

    for data_version in data_versions:
        modes_ = modes[data_version]
        for mode in modes_:
            data_path, law_charge_case_path, law2neg_path, out_path = get_necessary_files(PATHNAME, data_version, mode)
            print(data_path)
            print(law_charge_case_path)
            print(law2neg_path)
            law_charge_case_dict = json.load(open(law_charge_case_path, 'r'))
            law2neg_dict = json.load(open(law2neg_path, 'r'))
            # print(len(law2neg_dict.keys()))
            data_list = pk.load(open(data_path, 'rb'))
            dataset = CAIL_dataset(data_list, law_charge_case_dict, law2neg_dict, law_num=103, neg_num=2, mode=mode)
            with open(out_path, 'wb') as f:
                pk.dump(dataset, f)
                print('The {} is dumped.'.format(out_path))
            print('\n')

    # law_charge_case_dict = json.load(open(law_charge_case_path, 'r'))
    # law2neg_dict = json.load(open(law2neg_path, 'r'))
    # train_dataset = pk.load(open(CAIL_small_train, 'rb'))
    #
    # train_set = CAIL_dataset(train_dataset, law_charge_case_dict, law2neg_dict, law_num=103, neg_num=2, mode='train')
    #
    # print(train_set[0])
    #
    # configFilePath = '../Config/LadanPPK_Criminal.config'
    # config = ConfigParser(configFilePath)
    # Formatter = CAILFormatter(config, mode='train')
    #
    # def collate_fn(data):
    #     return Formatter.process(data, "train")
    # #
    # train_dataset = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False, collate_fn=collate_fn)
    # for step, data in enumerate(tqdm(train_dataset)):
    #     fact: torch.Tensor = data['fact']
    #     law_label, accu_label, time_label = data['law_label'], data['accu_label'], data['time_label']
    #     pos_article_idx, neg_article_idx = data['pos_article_index'], data['neg_article_index']
    #     pos_charge_fact, neg_charge_fact = data['pos_charge_fact'], data['neg_charge_fact']
    #     neg_article_masks, neg_charge_masks = data['neg_article_masks'], data['neg_charge_masks']
    #
    #     print(fact.shape)
    #     print(pos_article_idx.shape, neg_article_idx.shape)
    #     print(pos_charge_fact.shape, neg_charge_fact.shape)
    #     break

import sys
sys.path.append('..')
import pickle as pk
import string
import re
from zhon.hanzi import punctuation
import zhon
import numpy as np
from tqdm import tqdm
import json

subdataset_names = ['data', 'data_20w', 'data_38w']
datafile_list = ['train', 'valid', 'test']


def sentence2index_matrix(fact_list, word2id, max_length=512):
    length_max = 0
    for i in range(len(fact_list)):
        blank = word2id['BLANK']
        text = [blank for j in range(max_length)]
        content = fact_list[i].split()
        length = len(content)

        if length > length_max:
            length_max = length

        for j in range(len(content)):
            if(j == max_document_length):
                break
            if not content[j] in word2id:
                text[j] = word2id['UNK']
            else:
                text[j] = word2id[content[j]]

    return fact_list, length_max


def sentence2index_sequence(fact_list, word2id, max_length=512):
    index_list = []
    for i in tqdm(range(len(fact_list))):
        blank = word2id['BLANK']
        text = [blank for j in range(max_length)]
        content = fact_list[i].split()
        length = len(content)

        for j in range(len(content)):
            if(j == max_document_length):
                break
            if not content[j] in word2id:
                text[j] = word2id['UNK']
            else:
                text[j] = word2id[content[j]]
        index_list.append(text)

    return index_list


word2id_file = open('../Criminal_Dataset/word2id_Criminal', 'rb')
word2id = pk.load(word2id_file)
print(len(word2id))

for subdata_name in subdataset_names:
    dataname = ''
    if subdata_name == 'data':
        dataname = 'Criminal_small'
    if subdata_name == 'data_20w':
        dataname = 'Criminal_medium'
    if subdata_name == 'data_38w':
        dataname = 'Criminal_large'
    for file_name in datafile_list:
        max_document_length = 1500
        data_path = '../Criminal_Dataset/{}/'.format(subdata_name) + file_name
        f = open(data_path, 'r')
        content = f.readlines()
        f.close()
        facts = []
        charge_labels = []

        index = 0
        for line in tqdm(content):
            z = line.strip().split('\t')
            index += 1
            if(len(z) !=3):
                # print(index)
                # print(z)
                continue
            facts.append(z[0])
            charge_labels.append(int(z[1]))
        fact_list = sentence2index_sequence(facts, word2id, max_length=max_document_length)
        # print(len(list(set(charge_labels))))
        data_dict = {
            'fact_list': facts,
            'fact_index': fact_list,
            'charge_label_list': charge_labels
        }

        pk.dump(data_dict, open('/home/nxu/Ladan_tnnls/CL4LJP/processed_data/Criminal/{}/'.format(dataname)+'{}.pkl'.format(file_name), 'wb'))
        print('{}_{}_dataset is processed over'.format(dataname, file_name) + '\n')
        # json.dump(law_charge_case_dict,
        #           open('{}/CAIL/{}/{}_law_charge_case.json'.format(out_path, data_version, mode), 'w',
        #                encoding='utf-8'),
        #           ensure_ascii=False, indent=4)

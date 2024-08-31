import sys
import json
from transformers import BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding
import pickle as pkl
from tqdm import tqdm
import torch, ipdb


def get_input(tokenizer: BertTokenizer, sentence_list: list, max_length=512) -> dict:
    input_idx = tokenizer.encode_plus(sentence_list, max_length=max_length, padding='max_length', truncation=True)
    return dict(input_idx)

if __name__ == "__main__":
    # subdataset_names = ['data', 'data_20w', 'data_38w']
    subdataset_names = ['data_20w', 'data_38w']
    datafile_list = ['train', 'valid', 'test']
    pretrained_bert_fold = "/home/nxu/Ladan_tnnls/R-former/bert-base-chinese"
    # https://huggingface.co/bert-base-uncased
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(pretrained_bert_fold)
    for subdata_name in subdataset_names:
        dataname = ''
        if subdata_name == 'data':
            dataname = 'Criminal_small'
        if subdata_name == 'data_20w':
            dataname = 'Criminal_medium'
        if subdata_name == 'data_38w':
            dataname = 'Criminal_large'
        for file_name in datafile_list:
            max_document_length = 512
            data_path = '../../Criminal_Dataset/{}/'.format(subdata_name) + file_name
            f = open(data_path, 'r')
            content = f.readlines()
            f.close()
            input_ids_list = []
            token_type_ids_list = []
            attention_mask_lists = []
            charge_labels = []
            index = 0
            for i in tqdm(content, ncols=80):
                z = i.strip().split('\t')
                index += 1
                if(len(z) !=3):
                    # print(index)
                    # print(z)
                    continue
                fact = z[0].replace(" ", '') + '。'
                # print(fact)
                if fact != '' and len(fact) >= 10:
                    if len(fact) > 510:
                        fact = fact[:255] + fact[-255:]

                inputs = get_input(tokenizer, fact, max_length=max_document_length)
                charge_labels.append(int(z[1]))
                input_ids_list.append(inputs['input_ids'])
                token_type_ids_list.append(inputs['token_type_ids'])
                attention_mask_lists.append(inputs['attention_mask'])

            print('case num: ', len(input_ids_list))
            data_dict = {'input_ids_list': input_ids_list, 
                         'token_type_ids_list': token_type_ids_list, 
                         'attention_mask_lists': attention_mask_lists, 
                         'charge_label_list': charge_labels}
            pkl.dump(data_dict, open('../processed_dataset/Criminal/{}/'.format(dataname)+'{}_chinese.pkl'.format(file_name), 'wb'))
            
            
 



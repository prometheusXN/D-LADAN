import sys
import json
from transformers import BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding
import pickle as pkl
from tqdm import tqdm
import torch, ipdb
import numpy as np


pretrained_bert_fold = "/home/nxu/Ladan_tnnls/R-former/bert-base-chinese"
# https://huggingface.co/bert-base-uncased


def fact_clean(sentence: str) -> str:
    sentence = sentence.replace('。。', '。').replace('\n', '').replace('\t', '').replace(' ', '').strip()
    if sentence[0] == '。':
        sentence = sentence[1:]
    if sentence[-1] == '。':
        sentence = sentence[:-1]
    return sentence


def padding_or_truncating(sentence_list, max_heigth):
    sentence_num = len(sentence_list)
    word_num = len(sentence_list[0])
    appending = [0] * word_num
    if sentence_num >= max_heigth:
        sentence_list = sentence_list[:max_heigth]
    else:
        appending_num = max_heigth - sentence_num
        appending_list = [appending] * appending_num
        sentence_list += appending_list
    return sentence_list
'

def fixed_sentences(input_idx: BatchEncoding, sentence_num) -> BatchEncoding:
    for key in input_idx.keys():
        feature_matrix = input_idx[key]
        input_idx[key] = padding_or_truncating(feature_matrix, sentence_num)
    return input_idx


def get_input(tokenizer: BertTokenizer, sentence_list: list, sentence_num, max_length=250) -> dict:
    for i in range(len(sentence_list)):
        sentence_list[i] += '。'
    input_idx = tokenizer.batch_encode_plus(sentence_list, max_length=max_length, padding='max_length', truncation=True)
    input_idx = fixed_sentences(input_idx, sentence_num=sentence_num)
    return dict(input_idx)

# 由于bert对于中文是按照字进行编码，所以每一个句子的长度需要适当地增加长度的限制
max_length = 512

if __name__ == "__main__":
    file_list = ['train', 'valid', 'test']
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(pretrained_bert_fold)
    
    for i in range(len(file_list)):
        
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_lists = []
        law_label_lists = []
        accu_label_lists = []
        term_lists = []
        num = 0
        min_count = 100
        
        with open('../../data/{}_cs.json'.format(file_list[i]), 'r', encoding='utf-8') as f:
            idx = 0
            for line in tqdm(f.readlines(), ncols=80):
                idx += 1
                line = json.loads(line)
                fact = line['fact_cut'].replace(" ", '')
                sentence_list = []
                if fact != '' and len(fact) >= 10:
                    if len(fact) > 510:
                        fact = fact[:255] + fact[-255:]
                        fact = fact[:510]
                    sentence_list.append(fact)
                    
                if len(sentence_list) < 1: continue
                
                inputs = tokenizer.batch_encode_plus(sentence_list, max_length=max_length, padding='max_length', truncation=True)
                
                attention_mask = np.array(inputs['attention_mask'], dtype=np.float)
                # print(attention_mask.shape)
                index_num = np.sum(attention_mask)
                if index_num < (len(sentence_list) * 7):
                    if index_num < min_count:
                        min_count = index_num
                    continue
                
                input_ids_list.append(inputs['input_ids'][0])
                token_type_ids_list.append(inputs['token_type_ids'][0])
                attention_mask_lists.append(inputs['attention_mask'][0])
                law_label_lists.append(line['law'])
                accu_label_lists.append(line['accu'])
                term_lists.append(line['term'])
                num += 1
                # break
                
        attention_mask = np.array(attention_mask_lists, dtype=np.float)
        print(attention_mask.shape)
            
        data_dict = {'input_ids_list': input_ids_list, 
                     'token_type_ids_list': token_type_ids_list, 
                     'attention_mask_lists': attention_mask_lists, 
                     'law_label_lists': law_label_lists, 
                     'accu_label_lists': accu_label_lists, 
                     'term_lists': term_lists}
        pkl.dump(data_dict, open('../processed_dataset/CAIL_new/full_doc/small/{}_bert_chinese.pkl'.format(file_list[i]), 'wb'))
        print(num)
        print(min_count)
        print('{}_dataset is processed over'.format(file_list[i])+'\n')
        
    file_list = ['train', 'test']
    for i in range(len(file_list)):
        
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_lists = []
        law_label_lists = []
        accu_label_lists = []
        term_lists = []
        num = 0
        min_count = 100
        
        with open('../../big_data/{}_cs_new.json'.format(file_list[i]), 'r', encoding='utf-8') as f:
            idx = 0
            for line in tqdm(f.readlines(), ncols=80):
                idx += 1
                line = json.loads(line)
                fact = line['fact_cut'].replace(" ", '')
                sentence_list = []
                if fact != '':
                    sentence_list.append(fact)
                        
                if len(sentence_list) < 1: continue

                if len(fact) > 510:
                    fact = fact[:255] + fact[-255:]
                
                inputs = tokenizer.batch_encode_plus(sentence_list, max_length=max_length, padding='max_length', truncation=True)
                
                attention_mask = np.array(inputs['attention_mask'], dtype=np.float)
                index_num = np.sum(attention_mask)
                if index_num < (len(sentence_list) * 7):
                    if index_num < min_count:
                        min_count = index_num
                
                input_ids_list.append(inputs['input_ids'][0])
                token_type_ids_list.append(inputs['token_type_ids'][0])
                attention_mask_lists.append(inputs['attention_mask'][0])
                law_label_lists.append(line['law'])
                accu_label_lists.append(line['accu'])
                term_lists.append(line['term'])
                num += 1
            
        attention_mask = np.array(attention_mask_lists, dtype=np.float)
        print(attention_mask.shape)

        data_dict = {'input_ids_list': input_ids_list, 
                     'token_type_ids_list': token_type_ids_list, 
                     'attention_mask_lists': attention_mask_lists, 
                     'law_label_lists': law_label_lists, 
                     'accu_label_lists': accu_label_lists, 
                     'term_lists': term_lists}
        pkl.dump(data_dict, open('../processed_dataset/CAIL_new/full_doc/large/{}_bert_chinese.pkl'.format(file_list[i]), 'wb'))
        print(num)
        print(min_count)
        print('{}_dataset is processed over'.format(file_list[i])+'\n')

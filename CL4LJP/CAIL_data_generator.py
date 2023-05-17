import pickle as pk
import numpy as np
import json
from string import punctuation
from tqdm import tqdm

add_punc='，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥'
all_punc = punctuation + add_punc

stop_word_file = ''


def punc_delete(fact_list):
    fact_filtered = []
    for word in fact_list:
        fact_filtered.append(word)
        if word in all_punc:
            fact_filtered.remove(word)
    return fact_filtered


def data_generator(word2id_dict, out_path, law_num, max_length=1500, data_version='data', mode='train'):
    print('law num is ', law_num)
    fact_lists = []
    law_label_lists = []
    accu_label_lists = []
    term_lists = []
    index_list = []
    num = 0
    law_charge_case_dict = {i: {} for i in range(law_num)}
    with open('../{}/{}_cs.json'.format(data_version, mode), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    idx = 0
    for line in tqdm(lines):

        line = json.loads(line)
        fact = line['fact_cut'].strip().split(' ')
        fact = punc_delete(fact)
        id_list = []
        word_num = 0
        for j in range(int(min(len(fact), max_length))):
            if fact[j] in word2id_dict:
                id_list.append(int(word2id_dict[fact[j]]))
                word_num += 1
            else:
                id_list.append(int(word2id_dict['UNK']))
        while len(id_list) < max_length:
            id_list.append(int(word2id_dict['BLANK']))

        if word_num <= 10:
            # print(fact)
            # print(idx)
            continue

        id_numpy = np.array(id_list)
        fact_lists.append(id_numpy)
        law_label = line['law']
        charge_label = line['accu']
        time_label = line['term']

        if law_label in law_charge_case_dict.keys():
            if charge_label in law_charge_case_dict[law_label].keys():
                law_charge_case_dict[law_label][charge_label].append(idx)
            else:
                law_charge_case_dict[law_label][charge_label] = [idx]
        else:
            law_charge_case_dict[law_label] = {charge_label: [idx]}

        index_list.append(idx)
        law_label_lists.append(law_label)
        accu_label_lists.append(charge_label)
        term_lists.append(time_label)
        idx += 1
        num += 1
        f.close()
    data_dict = {
        'index_list': index_list,
        'fact_list': fact_lists,
        'law_label_lists': law_label_lists,
        'accu_label_lists': accu_label_lists,
        'term_lists': term_lists
    }

    pk.dump(data_dict, open('{}/CAIL/{}/{}_processed_thulac.pkl'.format(out_path, data_version, mode), 'wb'))
    print(num)
    print('{}_dataset is processed over'.format(mode)+'\n')
    json.dump(law_charge_case_dict,
              open('{}/CAIL/{}/{}_law_charge_case.json'.format(out_path, data_version, mode), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)


if __name__ == "__main__":
    with open('../data/w2id_thulac.pkl', 'rb') as f:
        word2id_dict = pk.load(f)
        f.close()

    # print(word2id_dict)

    file_list = ['train', 'valid', 'test']
    data_version = ['data', 'big_data']

    law_num = 103
    charge_num = 119
    max_length = 1500

    out_path = '/home/nxu/Ladan_tnnls/CL4LJP/processed_data'

    for mode in file_list:
        data_generator(word2id_dict, out_path=out_path,
                       law_num=law_num,
                       max_length=1500,
                       data_version='data', mode=mode)

    file_list = ['train', 'test']
    data_version = ['data', 'big_data']

    law_num = 118
    charge_num = 130
    max_length = 1500

    out_path = '/home/nxu/Ladan_tnnls/CL4LJP/processed_data'

    for mode in file_list:
        data_generator(word2id_dict, out_path=out_path,
                       law_num=law_num,
                       max_length=1500,
                       data_version='big_data', mode=mode)

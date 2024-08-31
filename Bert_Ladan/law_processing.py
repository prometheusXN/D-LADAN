import pandas as pd
import numpy as np
import json, re, pickle, os
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import pickle as pkl
from string import punctuation
from zhon.hanzi import punctuation
import zhon
from transformers import BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm
import ipdb
import jieba, thulac

add_punc=zhon.hanzi.punctuation
all_punc = punctuation + add_punc


def law_to_list(path, remain_new_line=False):
    with open(path, 'r', encoding='utf-8') as f:
        law = []
        for line in f:
            if line == '\n' or re.compile(r'第.*[节|章]').search(line[:10]) is not None:
                continue
            try:
                tmp = re.compile(r'第.*条').search(line.strip()[:8]).group(0)
                if remain_new_line:
                    law.append(line)
                else:
                    law.append(line.strip())
            except (TypeError, AttributeError):
                if remain_new_line:
                    law[-1] += line
                else:
                    law[-1] += line.strip()
    return law


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def hanzi_to_num(hanzi_1):
    # for num<10000
    hanzi = hanzi_1.strip().replace('零', '')
    if hanzi == '':
        return str(int(0))
    d = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '': 0}
    m = {'十': 1e1, '百': 1e2, '千': 1e3, }
    w = {'万': 1e4, '亿': 1e8}
    res = 0
    tmp = 0
    thou = 0
    for i in hanzi:
        if i not in d.keys() and i not in m.keys() and i not in w.keys():
            return hanzi

    if (hanzi[0]) == '十': hanzi = '一' + hanzi
    for i in range(len(hanzi)):
        if hanzi[i] in d:
            tmp += d[hanzi[i]]
        elif hanzi[i] in m:
            tmp *= m[hanzi[i]]
            res += tmp
            tmp = 0
        else:
            thou += (res + tmp) * w[hanzi[i]]
            tmp = 0
            res = 0
    return int(thou + res + tmp)


def process_law(law):
    # single article
    # cut=get_cutter()
    condition_list = []
    for each in law.split('。')[:-1]:
        contents = []
        if ';' or '；' or '：' or ':' in each:
            for item in re.split(r'[;；：:]', each):
                content = re.split('的，处|，处', item)[0]
                contents += (content + ';')
        else:
            content = re.split('的，处|，处', each)[0]
            contents += (content + ';')
        condition_list += contents

    n_word = [len(i) for i in condition_list]
    return condition_list, n_word


def process_law_old(law, cut):
    # single article
    # cut=get_cutter()
    condition_list = []
    for each in law.split('。')[:-1]:
        contents = []
        if ';' or '；' or '：' or ':' in each:
            for item in re.split(r'[;；：:]', each):
                content = re.split('的，处|，处', item)[0]
                content = cut(content)
                for i in content:
                    if i in all_punc:
                        content.remove(i)
                contents += content
        else:
            content = re.split('的，处|，处', each)[0]
            content = cut(content)
            for i in content:
                if i in all_punc:
                    content.remove(i)
            contents += content
        condition_list += contents

    n_word = [len(i) for i in condition_list]
    return condition_list, n_word


def cut_law(law_list, order=None, cut_sentence=True, cut_penalty=False):
    '''
    :param law_list:
    :param order:
    :param cut_sentence:
    :param cut_penalty:
    :param stop_words_filtered:
    :return:
    '''
    res = []
    if order is not None:
        key_list = [int(i) for i in order.keys()]
        filter = key_list
    for each in law_list:
        index, content = each.split('　')
        index = hanzi_to_num(index[1:-1])
        charge, content = content[1:].split('】')
        # if charge[-1]!='罪':
        #     continue
        if order is not None and index not in filter:
            continue
        if cut_penalty:
            context, n_words = process_law(content)
        elif cut_sentence:
            context, n_words = "", 0
            for sentence in content.split('。')[:-1]:
                if ';' or '；' or '：' or ':' in each:
                    for item in re.split(r'[;；：:]', sentence):
                        text_list = re.split('的，处|的，对|，处', item)
                        text = text_list[0]
                        if len(text_list) > 1:
                            context += (text + '的' + '；')
                            n_words += (len(text)+2)
                        else:
                            context += (text + '；')
                            n_words += (len(text)+1)
                else:
                    text_list = re.split('的，处|的，对|，处', item)
                    if len(text_list) > 1:
                        text = text_list[0]
                        context += (text + '的' + '；')
                        n_words += (len(text)+2)
                    else:
                        context += (text + '；')
                        n_words += (len(text)+1)
        else:
            context = content
            n_words = len(context)
        res.append([index, charge, context, n_words])
        res_1 = sorted(res, key=lambda x: x[-1])
    if order is not None:
        res = sorted(res, key=lambda x: order[str(x[0])])
        res_1 = sorted(res, key=lambda x: x[-1])
        return res
    return res


def get_cutter(dict_path="/home/nxu/Ladan_tnnls/law_processed/Thuocl_seg.txt", mode='thulac', stop_words_filtered=False):
    if stop_words_filtered:
        stopwords = stopwordslist('/home/nxu/Ladan_tnnls/law_processed/stop_word.txt')  # 这里加载停用词的路径
    else:
        stopwords = []
    if mode == 'jieba':
        jieba.load_userdict(dict_path)
        return lambda x: [a for a in list(jieba.cut(x)) if a not in stopwords]
    elif mode == 'thulac':
        thu = thulac.thulac(user_dict=dict_path, seg_only=True)
        return lambda x: [a for a in thu.cut(x, text=True).split(' ') if a not in stopwords]


def cut_law_old(law_list, order=None, cut_sentence=True, cut_penalty=False, stop_words_filtered=True):
    '''
    :param law_list:
    :param order:
    :param cut_sentence:
    :param cut_penalty:
    :param stop_words_filtered:
    :return:
    '''
    res = []
    cut = get_cutter(stop_words_filtered=stop_words_filtered)
    if order is not None:
        key_list = [int(i) for i in order.keys()]
        filter = key_list
    for each in tqdm(law_list, ncols=80):
        index, content = each.split('　')
        index = hanzi_to_num(index[1:-1])
        charge, content = content[1:].split('】')
        # if charge[-1]!='罪':
        #     continue
        if order is not None and index not in filter:
            continue
        if cut_penalty:
            context, n_words = process_law_old(content, cut)
        elif cut_sentence:
            context, n_words = [], []
            for sentence in content.split('。')[:-1]:
                contents = []
                if ';' or '；' or '：' or ':' in each:
                    for item in re.split(r'[;；：:]', sentence):
                        content = re.split('的，处|，处', item)
                        content = cut(content)
                        contents += content
                else:
                    content = re.split('的，处|，处', sentence)[0]
                    content = cut(content)
                    contents += content
                # contents = cut(sentence)
                context.append(contents)
                n_words.append(len(context[-1]))
        else:
            context = cut(content)
            n_words = len(context)
        res.append([index, charge, context, n_words])
    if order is not None:
        res = sorted(res, key=lambda x: order[str(x[0])])
    return res


def flatten(x):
    return [y for l in x for y in flatten(l)] if type(x) is list else [x]


def cos_similarity(a, b):
    return np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_input(tokenizer: BertTokenizer, sentence_list: list, max_length=400) -> dict:
    for i in range(len(sentence_list)):
        sentence_list[i] += '。'
    input_idx = tokenizer.batch_encode_plus(sentence_list, max_length=max_length, padding='max_length', 
                                            truncation=True, return_tensors='pt')
    return dict(input_idx)


def gen_law_relation(tokenizer, law_label2index_path='/home/nxu/Ladan_tnnls/law_processed/law_label2index.pkl', max_length=400):
    law_file_order = pkl.load(open(law_label2index_path, 'rb'))
    n_law = len(law_file_order)
    law_list = law_to_list('/home/nxu/Ladan_tnnls/law_processed/criminal_law.txt')
    laws = cut_law_old(law_list, order=law_file_order, cut_sentence=True, cut_penalty=True)
    #################################---------------get_data_index_matrix-----------------#################################
    law_1 = cut_law(law_list, order=law_file_order, cut_sentence=True, cut_penalty=False)
    law_1 = list(zip(*law_1))
    content= list(law_1[-2])
    # print(content[0])
    law_index_matrix = get_input(tokenizer, content, max_length=max_length)
    #################################---------------get_data_index_matrix-----------------#################################

    laws = list(zip(*laws))
    index = laws[0]
    laws = [' '.join(flatten(each)) for each in laws[-2]]
    tfidf = TfidfVectorizer().fit_transform(laws).toarray()
    # print("tf_idf features:")
    # print(tfidf)

    sim = np.zeros([n_law, n_law])
    for i in range(n_law):
        for j in range(n_law):
            sim[i, j] = cos_similarity(tfidf[i], tfidf[j])
    return sim, law_index_matrix, n_law


def group_generation(neigh_index):
    graph = []
    items = []
    graph_ship = {}
    for i in range(len(neigh_index)):
        if len(neigh_index[i]) == 0:
            graph.append([i])
        else:
            if neigh_index[i][0] in items:
                continue
            else:
                sub_graph = neigh_index[i]
                finding = neigh_index[i]
                exchange = []
                for j in finding:
                    exchange += neigh_index[j]
                exchange = list(set(exchange))  # 去重
                finding = exchange
                exchange = []
                while (set(sub_graph) >= set(finding)) is False:
                    sub_graph = list(set(sub_graph).union(set(finding)))
                    for j in finding:
                        exchange += neigh_index[j]
                    exchange = list(set(exchange))
                    finding = exchange
                    exchange = []
                graph.append(sub_graph)
                items += sub_graph
    for i in range(len(graph)):
        graph_1 = {j: i for j in graph[i]}
        graph_ship.update(graph_1)
    graph_ship = sorted(graph_ship.items())

    return graph, graph_ship


def get_law_graph(threshold, sent_len=400):
    pretrained_bert_fold = "/home/nxu/LEVENs/LEVEN-main/LCR_with_LawArticles/pretrain_model/xs/"
    tokenlizer: BertTokenizer = BertTokenizer.from_pretrained(pretrained_bert_fold)
    neigh_mat, law_index_matrix, law_num = gen_law_relation(tokenlizer, max_length=sent_len)
    # print(neigh_mat)
    neigh_index = np.where(neigh_mat > threshold)
    neigh_index = list(zip(*neigh_index))
    neigh_index = {i: [j for j in range(103) if (i, j) in neigh_index and j != i] for i in range(103)}
    graph_list_1, graph_membership = group_generation(neigh_index)
    return law_index_matrix, graph_list_1, graph_membership, neigh_index


def get_law_graph_adj(threshold, sent_len=400, pretrained_bert_fold="/home/nxu/LEVENs/LEVEN-main/LCR_with_LawArticles/pretrain_model/xs/"):
    tokenlizer: BertTokenizer = BertTokenizer.from_pretrained(pretrained_bert_fold)
    neigh_mat, law_index_matrix, law_num = \
        gen_law_relation(tokenlizer, law_label2index_path='/home/nxu/Ladan_tnnls/law_processed/law_label2index.pkl',
                         max_length=sent_len)
    zero_matrix = np.zeros_like(neigh_mat, dtype=np.float)
    one_metrix = np.ones_like(neigh_mat, dtype=np.float)
    adj_matrix = np.where(neigh_mat > threshold, one_metrix, zero_matrix)
    adj_matrix = adj_matrix - np.eye(N=law_num, dtype=np.float)

    neigh_index = np.where(neigh_mat > threshold)
    neigh_index = list(zip(*neigh_index))
    neigh_index = {i: [j for j in range(103) if (i, j) in neigh_index and j != i] for i in range(103)}
    graph_list_1, graph_membership = group_generation(neigh_index)
    return law_index_matrix, graph_list_1, graph_membership, adj_matrix


def get_law_graph_large_adj(threshold, sent_len=400, pretrained_bert_fold="/home/nxu/LEVENs/LEVEN-main/LCR_with_LawArticles/pretrain_model/xs/"):
    tokenlizer: BertTokenizer = BertTokenizer.from_pretrained(pretrained_bert_fold)
    neigh_mat, law_index_matrix, law_num = \
        gen_law_relation(tokenlizer, law_label2index_path='/home/nxu/Ladan_tnnls/law_processed/law_label2index_large.pkl', 
                         max_length=sent_len)
    zero_matrix = np.zeros_like(neigh_mat, dtype=np.float)
    one_metrix = np.ones_like(neigh_mat, dtype=np.float)
    adj_matrix = np.where(neigh_mat > threshold, one_metrix, zero_matrix)
    adj_matrix = adj_matrix - np.eye(N=law_num, dtype=np.float)

    neigh_index = np.where(neigh_mat > threshold)
    neigh_index = list(zip(*neigh_index))
    neigh_index = {i: [j for j in range(118) if (i, j) in neigh_index and j != i] for i in range(118)}
    graph_list_1, graph_membership = group_generation(neigh_index)
    return law_index_matrix, graph_list_1, graph_membership, adj_matrix


def get_relation_InterTask(law_num, accu_num, time_num, mode='small'):

    adjacent_matrix = np.zeros(shape=(law_num+accu_num+time_num, law_num+accu_num+time_num), dtype=np.int)
    if mode == 'small':
        law2accu = pickle.load(open('../data/law2accu.pkl', 'rb'))  # array
        law2term = pickle.load(open('../data/law2term.pkl', 'rb'))
        accu2term = pickle.load(open('../data/accu2term.pkl', 'rb'))
    else:
        law2accu = pickle.load(open('../data/law2accu_large.pkl', 'rb'))
        law2term = pickle.load(open('../data/law2term_large.pkl', 'rb'))
        accu2term = pickle.load(open('../data/accu2term_large.pkl', 'rb'))

    adjacent_matrix[:law_num, law_num:(law_num+accu_num)] = law2accu
    adjacent_matrix[:law_num, (law_num+accu_num):(law_num+accu_num+time_num)] = law2term
    adjacent_matrix[:, :law_num] = adjacent_matrix[:law_num].T

    return adjacent_matrix

def transform(word, word2id):
    if not (word in word2id.keys()):
        return word2id["UNK"]
    else:
        return word2id[word]


def parse(sent):
    result = []
    sent = sent.strip().split()
    for word in sent:
        if len(word) == 0:
            continue
        if word in all_punc:
            continue
        result.append(word)
    return result


def get_mean_vector(string, word2id, embeddings):
    word_list = parse(string)
    index_list = []
    for word in word_list:
        index_list.append(transform(word, word2id))
    vector_list = []
    for i in index_list:
        vector_list.append(embeddings[i][None, :])
    vector_matrix = np.concatenate(vector_list, axis=0)
    mean_vector = vector_matrix.mean(0, keepdims=True)
    return mean_vector


def get_matrix(content_dict, label_dict, word2id, embeddings, type='charge'):
    content_list = []
    index_num = len(label_dict)
    for i in range(index_num):
        if type == 'charge':
            content_list.append(content_dict[label_dict[str(i)]]['定义'])
        else:
            content_list.append(content_dict[label_dict[str(i)]])

    vector_list = []
    for content in content_list:
        vector = get_mean_vector(content, word2id, embeddings)
        vector_list.append(vector)

    return np.concatenate(vector_list, axis=0), content_list


def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return res


def get_neigh_dict(cos_sim_matrix, threshold):
    class_num = cos_sim_matrix.shape[0]
    neigh_index = np.where(cos_sim_matrix > threshold)
    neigh_index = list(zip(*neigh_index))
    neigh_index = {i: [j for j in range(class_num) if (i, j) in neigh_index and j != i] for i in range(class_num)}

    graph_list_1, graph_membership = group_generation(neigh_index)

    return graph_list_1, graph_membership


def get_accu_graph_adj(tokenizer: BertTokenizer, threshold, word2id_file, id2charge_path, emb_path):
    with open(word2id_file, 'rb') as f:
        word2id = pkl.load(f)
    embeddings = np.cast[np.float32](np.load(emb_path))
    id2charge_dict = json.load(open(id2charge_path, 'r'))
    charge_num = len(id2charge_dict)
    print('charge_num:', charge_num)
    charge_detail = json.load(open('/home/nxu/Ladan_tnnls/law_processed/charge2details.json', 'r'))
    charge_matrix, content_list = get_matrix(charge_detail, id2charge_dict, word2id, embeddings, type='charge')
    cos_sim_matrix = get_cos_similar_matrix(charge_matrix, charge_matrix)
    length = 0
    for i in range(len(content_list)):
        content_list[i] = content_list[i].replace(" ", "")
        if len(content_list[i]) > length: length = len(content_list[i])
    # print(content_list)
    # print(length) # "small: 151; large: 151"
    accu_index_matrix = get_input(tokenizer, content_list, max_length=153)
    
    zero_matrix = np.zeros_like(cos_sim_matrix, dtype=np.float)
    one_metrix = np.ones_like(cos_sim_matrix, dtype=np.float)
    adj_matrix = np.where(cos_sim_matrix > threshold, one_metrix, zero_matrix)
    adj_matrix = adj_matrix - np.eye(N=charge_num, dtype=np.float)

    graph_list, graph_membership = get_neigh_dict(cos_sim_matrix, threshold)
    return accu_index_matrix, graph_list, graph_membership, adj_matrix


if __name__ == '__main__':
    pretrained_bert_fold = "/home/nxu/LEVENs/LEVEN-main/LCR_with_LawArticles/pretrain_model/xs/"
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(pretrained_bert_fold)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    law_label2index_path='../law_processed/law_label2index.pkl'
    # law_label2index_path='../law_processed/law_label2index_large.pkl'
    law_file_order = pkl.load(open(law_label2index_path, 'rb'))
    print(law_file_order)
    # n_law = len(law_file_order)
    # law_list = law_to_list('../law_processed/criminal_law.txt')
    # sim, law_inputs, n_law = gen_law_relation(tokenizer, law_label2index_path, max_length=400)
    # print(law_inputs['input_ids'].shape)
    law_index_matrix, graph_list_1, graph_membership, adj_matrix = get_law_graph_adj(threshold=0.35)
    print(graph_membership)
    group_indexes = list(zip(*graph_membership))[1]
    print(group_indexes)
    # laws, res_1 = cut_law(law_list, order=law_file_order, cut_sentence=True, cut_penalty=False)
    # print(laws)
    # print(res_1)
    # law_1 = list(zip(*res_1))
    # print(list(law_1[-2]))
    # print(type(list(law_1[-2])))
    # print(len(res_1))
    # ipdb.set_trace()
    # max_length_small = 391
    # max_length_large = 391
    
    # id2charge_small = '/home/nxu/Ladan_tnnls/neurjudge/data/id2charge_small.json'
    # id2charge_large = '/home/nxu/Ladan_tnnls/neurjudge/data/id2charge_large.json'
    # emb_path = '../data/cail_thulac_new.npy'
    # word2id_path = '../data/w2id_thulac_new.pkl'
    # acc_threshold = 0.60
    # charge_index_matrix, graph_list_small, graph_membership, adj_matrix = \
    #     get_accu_graph_adj(tokenizer, acc_threshold, word2id_path, id2charge_small, emb_path)
        
    # print(charge_index_matrix['input_ids'].shape)
    

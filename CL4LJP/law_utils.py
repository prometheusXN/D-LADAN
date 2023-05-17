import numpy as np
from zhon.hanzi import punctuation
import zhon
from law_processed.law_processed import *
import jieba
import thulac
from tqdm import tqdm
import json
import torch

add_punc=zhon.hanzi.punctuation
all_punc = punctuation + add_punc
symbol = [",", ".", "?", "\"", "”", "。", "？", "", "，", ",", "、", "”"]


def get_cutter(dict_path="/home/nxu/Ladan_tnnls/law_processed/Thuocl_seg.txt", mode='thulac', stop_words_filtered=True):
    if stop_words_filtered:
        stopwords = stopwordslist('../law_processed/stop_word.txt')  # 这里加载停用词的路径
    else:
        stopwords = []
    if mode == 'jieba':
        jieba.load_userdict(dict_path)
        return lambda x: [a for a in list(jieba.cut(x)) if a not in stopwords]
    elif mode == 'thulac':
        thu = thulac.thulac(user_dict=dict_path, seg_only=True)
        return lambda x: [a for a in thu.cut(x, text=True).split(' ') if a not in stopwords]


def cut_sentence(sentence, cut):
    word_list = cut(sentence)
    word_string = " ".join(word_list)
    return word_string


def generate_law_dict(law_path, cutter):
    law_list = law_to_list(law_path)
    l_dict = {}
    for law in tqdm(law_list):
        index, content = law.split('　')
        index = hanzi_to_num(index[1:-1])
        charge, content = content[1:].split('】')
        content = cut_sentence(content, cutter)
        l_dict[str(index)] = content
    return l_dict


def parse(sent):
    result = []
    sent = sent.strip().split()
    for word in sent:
        if len(word) == 0:
            continue
        if word in symbol:
            continue
        result.append(word)
    return result


def transform(word2id, word):
    if not (word in word2id.keys()):
        return word2id["UNK"]
    else:
        return word2id[word]


def seq2tensor(sents, word2id, max_len=350):
    sent_len_max = max([len(s) for s in sents])
    sent_len_max = min(sent_len_max, max_len)
    init = np.ones((len(sents), sent_len_max), dtype=np.int) * word2id["BLANK"]
    sent_tensor = torch.LongTensor(init)

    sent_len = torch.LongTensor(len(sents)).zero_()
    for s_id, sent in enumerate(sents):
        sent_len[s_id] = len(sent)
        for w_id, word in enumerate(sent):
            if w_id >= sent_len_max: break
            sent_tensor[s_id][w_id] = transform(word2id, word)
    return sent_tensor, sent_len


def get_articles(law_dict_path, law_index_path, word2id, max_len=1500):
    law_dict = json.load(open(law_dict_path, 'r'))
    law_index_list = open(law_index_path, 'r', encoding='utf-8').readlines()
    arts = []
    for law_index in law_index_list:
        index = law_index.strip()
        arts.append(parse(law_dict[str(index)]))

    arts, arts_sent_lent = seq2tensor(arts, word2id, max_len=max_len)
    return arts


if __name__ == "__main__":
    law_path = '../law_processed/criminal_law.txt'
    cutter = get_cutter(stop_words_filtered=False)
    law_dict = generate_law_dict(law_path=law_path, cutter=cutter)
    law_dict_path = '../CL4LJP/law2detail.json'

    print(law_dict)
    json.dump(law_dict, open(law_dict_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
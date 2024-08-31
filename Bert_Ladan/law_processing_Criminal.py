import sys
sys.path.append('..')
import pickle as pk
from law_processing import *
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from zhon.hanzi import punctuation
import zhon


def split_laws(law_list):
    """
    :param law_list:
    :return: the law_list [index, charge_list, law_content]
    """
    laws = []
    for each in law_list:
        index, content = each.split('　')
        index = hanzi_to_num(index[1:-1])
        charge, content = content[1:].split('】')
        charges = re.split("罪" + r"[;、]", charge)
        num = len(charges)
        for i in range(num):
            if i == (num - 1):
                continue
            else:
                charges[i] += '罪'
        laws.append([index, charges, content])
    return laws


def search_law_for_changes(attribute_path='../Criminal_Dataset/attributes',
                           lawsource_path='../law_processed/criminal_law.txt'):
    law_list = law_to_list(lawsource_path)
    laws = split_laws(law_list)
    
    attribute_items = open(attribute_path, 'r').readlines()
    index2charge = {}
    charge2lawindex = {}
    chargeindex2lawindex = {}
    
    for item in attribute_items:
        item = item.strip().strip('\t').split('\t')
        charge_name = item[2]
        index = item[0]
        index2charge[index] = charge_name
        law_list = []
        for law in laws:
            charges = law[1]
            law_index = law[0]
            law_content = law[2]
            if charge_name in charges:
                law_list.append(law_index)
            if charge_name == "失火罪":
                law_list = [114, 115]
            if charge_name == "过失损坏广播电视设施、公用电信设施罪":
                law_list = [124]
            if charge_name == "隐匿、故意销毁会计凭证、会计帐簿、财务会计报告罪":
                law_list = [162]
            if charge_name == "扰乱无线电通讯管理秩序罪":
                law_list = [288]
            if charge_name == "窝藏、转移、收购、销售赃物罪":
                law_list = [312]
            if charge_name == "非法狩猎罪":
                law_list = [341]
            if charge_name == "故意杀人罪":
                law_list = [232]
            if charge_name == "故意伤害罪":
                law_list = [234]
            if charge_name == "强奸罪":
                law_list = [236]
            if charge_name == "非法拘禁罪":
                law_list = [238]
            if charge_name == "侮辱罪":
                law_list = [246]
            if charge_name == "诈骗罪":
                law_list = [210, 266]

        charge2lawindex[charge_name] = law_list
        chargeindex2lawindex[index] = law_list
    return chargeindex2lawindex, charge2lawindex, index2charge, laws


def law_content_filered(law, cut):
    condition_list = []
    for each in law.split('。')[:-1]:
        suffix = None
        if '：' in each:
            each, suffix = each.split('：')
            suffix = cut(suffix)
        contents = []
        if ';' or '；' or '：' or ':' in each:
            for item in re.split(r'[;；：:]', each):
                content = re.split('的，处|，处|，依照|，对', item)[0]
                content = cut(content)
                for i in content:
                    if i in (string.punctuation + zhon.hanzi.punctuation):
                        content.remove(i)
                contents += content
        else:
            content = re.split('的，处|，处|，依照|，对', each)[0]
            content = cut(content)
            for i in content:
                if i in (string.punctuation + zhon.hanzi.punctuation):
                    content.remove(i)
            contents += content

        condition_list += contents
    n_word = [len(i) for i in condition_list]
    return condition_list, n_word


def generate_tfidf_vectors(law_list):
    cut = get_cutter(stop_words_filtered=False, dict_path='../Criminal_Dataset/Thuocl_seg.txt')
    contexts = []
    for each in law_list:
        content = each[2]
        context, n_words = law_content_filered(content, cut)
        contexts.append(context)
    laws = [' '.join(flatten(each)) for each in contexts]
    tfidf = TfidfVectorizer().fit_transform(laws).toarray()
    return tfidf


def generate_sentence_split(law_list, cut_sentence=False):
    res = []
    for each in law_list:
        content = each[2]
        if cut_sentence:
            context, n_words = [], []
            for sentence in content.split('。')[:-1]:
                contents = ''
                if ';' or '；' or '：' or ':' in each:
                    for item in re.split(r'[;；：:]', sentence):
                        content = re.split('的，处|，处', item)[0]
                        contents += content
                else:
                    content = re.split('的，处|，处', sentence)[0]
                    contents += content
                context.append(contents)
                n_words.append(len(context[-1]))
        else:
            context, n_words = '', 0
            for sentence in content.split('。')[:-1]:
                contents = ''
                if ';' or '；' or '：' or ':' in each:
                    for item in re.split(r'[;；：:]', sentence):
                        # print(item)
                        content = re.split('的，处|，处|，依照|，对', item)[0]
                        contents += (content + '的;')
                    contents = contents[:-1]
                else:
                    content = re.split('的，处|，处|，依照|，对', sentence)[0]
                    contents += (content + '的')
                context+=contents
            n_words=[len(context)]
        #     print(context)
        # import ipdb; ipdb.set_trace()
        each[2] = context
        each.append(n_words)
    return law_list


def add_edge(matrix, edge_list, edge_dict):
    for group in edge_list:
        for head in group:
            for tail in group:
                if tail != head:
                    matrix[edge_dict[head], edge_dict[tail]] = 1.0
    return matrix


def get_law_graph_for_Criminal(threshold=0.30,
                               attribute_path='../Criminal_Dataset/attributes',
                               pretrained_bert_fold="/home/nxu/LEVENs/LEVEN-main/LCR_with_LawArticles/pretrain_model/xs/",
                               max_length=300):
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(pretrained_bert_fold)
    chargeindex2lawindex, charge2lawindex, index2charge, laws = \
        search_law_for_changes(attribute_path=attribute_path,
                               lawsource_path='../law_processed/criminal_law.txt')
        
    law_index_list = []
    for i in charge2lawindex.items():
        charge_name, law_indexes = i
        law_index_list += law_indexes

    law_index_list = list(set(law_index_list))
    laws_filtered = []
    for each in laws:
        index = each[0]
        if index not in law_index_list:
            continue
        laws_filtered.append(each)

    index2law_label = {}
    law_label2index = {}
    law_num = len(laws_filtered)
    for i in range(law_num):
        index2law_label[i] = laws_filtered[i][0]
        law_label2index[laws_filtered[i][0]] = i

    #--------------------relation computation-----------------------#
    tfidf = generate_tfidf_vectors(laws_filtered)
    sim = np.zeros([law_num, law_num])
    for i in range(law_num):
        for j in range(law_num):
            sim[i, j] = cos_similarity(tfidf[i], tfidf[j])

    neigh_mat = sim
    add_edge_list = [[117, 119], [263, 267, 269], [253, 264, 265], [185, 272], [183, 271, 382, 394], [185, 272, 384]]
    neigh_mat = add_edge(neigh_mat, add_edge_list, law_label2index)
    zero_matrix = np.zeros_like(neigh_mat, dtype=np.float)
    one_metrix = np.ones_like(neigh_mat, dtype=np.float)
    adj_matrix = np.where(neigh_mat > threshold, one_metrix, zero_matrix)
    adj_matrix = adj_matrix - np.eye(N=law_num, dtype=np.float)

    neigh_index = np.where(neigh_mat > threshold)
    neigh_index = list(zip(*neigh_index))
    neigh_index = {i: [j for j in range(law_num) if (i, j) in neigh_index and j != i] for i in range(law_num)}
    graph_list_1, graph_membership = group_generation(neigh_index)
    law_list = generate_sentence_split(laws_filtered)
    
    law_list = list(zip(*law_list))
    law_index_matrix = get_input(tokenizer, list(law_list[-2]), max_length=max_length)
    
    def trans_to_charge(graph_membership, chargeindex2lawindex):
        graph_dict = {}
        for item in graph_membership:
            law_index, group_index = item
            graph_dict[index2law_label[law_index]] = group_index
        chargeindex2groupindex = {i:[] for i in chargeindex2lawindex.keys()}
        for item in chargeindex2lawindex.items():
            charge_index, law_indexes = item
            for law in graph_dict.items():
                law_index, group_index = law
                if law_index in law_indexes:
                    chargeindex2groupindex[charge_index] += [group_index]
        graph_membership_charge = []
        for key in chargeindex2groupindex.keys():
            graph_membership_charge.append([int(key), list(set(chargeindex2groupindex[key]))[0]])
        return graph_membership_charge
    
    graph_membership_charge = trans_to_charge(graph_membership, chargeindex2lawindex)
    return law_index_matrix, graph_list_1, graph_membership, graph_membership_charge, adj_matrix


if __name__ == "__main__":
    law_index_matrix, graph_list_1, graph_membership, graph_membership_charge, adj_matrix = get_law_graph_for_Criminal()
    import ipdb; ipdb.set_trace()
    

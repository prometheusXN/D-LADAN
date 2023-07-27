import re
import pickle as pk
import numpy as np


text = '聚众扰乱车站、码头、民用航空站、商场、公园、影剧院、展览会、运动场或者其他公共场所秩序，聚众堵塞交通或者破坏交通秩序，抗拒、阻碍国家治安管理工作人员依法执行职务，情节严重的，对首要分子，处五年以下有期徒刑、拘役或者管制'
lists = re.split('，处|的，处', text)
for i in lists:
    print(i)

with open('../data/w2id_thulac.pkl', 'rb') as f:
    word2id_dict = pk.load(f)
    f.close()
for key in word2id_dict.keys():
    if key == 'BLANK':
        word2id_dict[key] = 0
    else:
        word2id_dict[key] += 1
emb_path = '../data/cail_thulac.npy'
embeddings = np.load(emb_path)
BLANK = embeddings[-1, :]
print(BLANK)
rest_embeddings = embeddings[:-1, :]
print(np.shape(rest_embeddings))
new_embeddings = np.concatenate([np.expand_dims(BLANK, axis=0), rest_embeddings], 0)
print(np.shape(new_embeddings))

new_embeddings_path = '../data/cail_thulac_new.npy'
np.save(new_embeddings_path, new_embeddings)



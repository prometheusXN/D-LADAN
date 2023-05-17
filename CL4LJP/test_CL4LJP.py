from torch.utils.data import DataLoader
from Config.parser import ConfigParser
import pickle as pk
from CL4LJP.utils import CAIL_dataset, CAILFormatter
import torch
from CL4LJP.model.CL4LJP_model import CL4LJP
import random
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
import numpy as np
from sklearn import metrics
from CL4LJP.law_utils import get_articles
from neurjudge.metrics import eval_data_types, filter_samples


def evaluation_multitask(y, prediction, task_num):
    metrics_acc = []
    for x in range(task_num):
        y_pred = prediction[x]
        y_true = y[x]
        accuracy_metric = metrics.accuracy_score(y_true, y_pred)
        macro_recall = metrics.recall_score(y_true, y_pred, average='macro')
        micro_recall = metrics.recall_score(y_true, y_pred, average='micro')
        macro_precision = metrics.precision_score(y_true, y_pred, average='macro')
        micro_precision = metrics.precision_score(y_true, y_pred, average='micro')
        macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')
        micro_f1 = metrics.f1_score(y_true, y_pred, average='micro')
        metrics_acc.append(
            (accuracy_metric, macro_recall, micro_recall, macro_precision, micro_precision, macro_f1, micro_f1))
    return metrics_acc


def read_dataset(path_name, data_version, mode):
    data_path = path_name + data_version + '/{}_dataset.pkl'.format(mode)
    dataset = pk.load(open(data_path, 'rb'))
    return dataset


if __name__ == "__main__":
    random.seed(666)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=str, default='4',
                        help='Select which GPU device to use')
    parser.add_argument('--data_version', '-dv', type=str, default='data', choices=['data', 'big_data'],
                        help='Set which dataset is used.')
    parser.add_argument('--embedding_trainable', '-et', type=bool, default=False,
                        help='Whether the word embedding is trainable')
    parser.add_argument('--version_index', '-vi', type=int, default=0,
                        help='The index for saving different model.')

    data_mode = {'data': ['tarin', 'valid', 'test']}
    args = parser.parse_args()
    data_version = args.data_version
    embedding_trainable = args.embedding_trainable
    version_index = args.version_index
    GPU = args.gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = '../model_save/CL4LJP/CL4LJP_{}'.format(data_version)

    # -------------------------------set_configs--------------------------#
    configFilePath = '../Config/CL4LJP.config'
    config = ConfigParser(configFilePath)
    batch_size = config.getint('data', 'batch_size')
    max_epoch = config.getint('train', 'epoch')

    emb_path = '../data/cail_thulac.npy'

    with open('../data/w2id_thulac.pkl', 'rb') as f:
        word2id_dict = pk.load(f)
        f.close()

    word_dict_len = len(word2id_dict)

    data_path = '/home/nxu/Ladan_tnnls/CL4LJP/processed_data/CAIL/'

    test_set = read_dataset(data_path, data_version, 'test')
    if not isinstance(test_set, CAIL_dataset):
        print('Dataset has an error type.')

    Formatter = CAILFormatter(config, mode='train')

    def collate_fn(data):
        return Formatter.process(data, "train")

    test_dataset = DataLoader(test_set, batch_size=128,
                              shuffle=False, drop_last=False,
                              collate_fn=collate_fn)

    law_dict_path = '/home/nxu/Ladan_tnnls/CL4LJP/law2detail.json'
    law_index_path = '/home/nxu/Ladan_tnnls/data/new_law.txt'

    articles = get_articles(law_dict_path, law_index_path, word2id_dict, 1500)
    articles = articles.to(device)

    model = CL4LJP(config=config, emb_path=emb_path, word2id_dict=word2id_dict, data_version='small', device=device)
    print("loading model from: ", model_path)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    learning_rate = 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    law_pred_list = []
    law_label_list = []
    accu_pred_list = []
    accu_label_list = []
    time_pred_list = []
    time_label_list = []
    print('Test')
    with torch.no_grad():
        for data in tqdm(test_dataset):
            fact, law_label, accu_label, time_label = \
                data['fact'].to(device), data['law_label'].to(device), \
                data['accu_label'].to(device), data['time_label'].to(device)

            pos_article_index, neg_article_index, neg_article_masks = \
                data['pos_article_index'].to(device), data['neg_article_index'].to(device), data[
                    'neg_article_masks'].to(device)

            pos_charge_fact, neg_charge_fact, neg_charge_masks = \
                data['pos_charge_fact'].to(device), data['neg_charge_fact'].to(device), data['neg_charge_masks'].to(
                    device)

            charge_pred, article_pred, time_pred = \
                model(fact, articles, accu_label, law_label, time_label, pos_article_index, neg_article_index,
                      pos_charge_fact, neg_charge_fact, neg_article_masks, neg_charge_masks, model_type="test")

            law_pred_list.append(article_pred.cpu().numpy())
            law_label_list.append(law_label.cpu().numpy())
            accu_pred_list.append(charge_pred.cpu().numpy())
            accu_label_list.append(accu_label.cpu().numpy())
            time_pred_list.append(time_pred.cpu().numpy())
            time_label_list.append(time_label.cpu().numpy())

    law_preds = np.argmax(np.concatenate(law_pred_list, axis=0), axis=1)
    law_labels = np.concatenate(law_label_list, axis=0)
    accu_preds = np.argmax(np.concatenate(accu_pred_list, axis=0), axis=1)
    accu_labels = np.concatenate(accu_label_list, axis=0)
    time_preds = np.argmax(np.concatenate(time_pred_list, axis=0), axis=1)
    time_labels = np.concatenate(time_label_list, axis=0)

    metric = evaluation_multitask([law_labels, accu_labels, time_labels],
                                  [law_preds, accu_preds, time_preds], 3)

    task = ['law', 'accu', 'time']
    print('Now_testing')
    ave_acc = 0
    for i in range(3):
        print('Metrics for {} prediction is: '.format(task[i]), metric[i])

    # tail_law_list = [37, 89, 74, 66, 25, 82, 32, 36, 5, 81, 7, 43, 17, 14, 48, 76, 96, 15, 72, 70, 54, 69, 49, 53, 85,
    #                  56, 44]
    # tail_charge_list = [37, 85, 96, 114, 99, 86, 106, 36, 71, 104, 72, 95, 34, 39, 51, 23, 109, 62, 54, 31, 59, 60, 81,
    #                     110, 76, 26, 19, 64]

    tail_law_list = [81, 7, 43, 17, 14, 48, 76, 96, 15, 72, 70, 54, 69, 49, 53, 85, 56, 44]
    tail_charge_list = [114, 99, 86, 106, 36, 71, 104, 72, 95, 34, 39, 51, 23, 109, 62, 54, 31, 59, 60, 81, 110, 76, 26,
                        19, 64]

    # law_preds = law_preds.tolist()
    # accu_preds = accu_preds.tolist()
    #
    # law_labels = law_labels.tolist()
    # accu_labels = accu_labels.tolist()

    law_labels, law_preds = filter_samples(law_preds, law_labels, tail_law_list)
    accu_labels, accu_preds = filter_samples(accu_preds, accu_labels, tail_charge_list)

    print('Law article:')
    eval_data_types(law_labels, law_preds, num_labels=103, label_list=None)

    print('Charge:')
    eval_data_types(accu_labels, accu_preds, num_labels=119, label_list=None)


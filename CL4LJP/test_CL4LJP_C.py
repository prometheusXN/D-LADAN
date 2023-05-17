from torch.utils.data import DataLoader
from Config.parser import ConfigParser
import pickle as pk
from CL4LJP.utils import Criminal_dataset, CriminalFormatter
import torch
from CL4LJP.model.CL4LJP_model_C import CL4LJP_C
import random
import torch.optim as optim
from torch.optim import lr_scheduler
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
    data_path = path_name + data_version + '/{}.pkl'.format(mode)
    dataset = pk.load(open(data_path, 'rb'))
    return dataset


if __name__ == "__main__":
    random.seed(666)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=str, default='3',
                        help='Select which GPU device to use')
    parser.add_argument('--data_version', '-dv', type=str, default='Criminal_small',
                        choices=['Criminal_small', 'Criminal_medium', 'Criminal_large'],
                        help='Set which dataset is used.')
    parser.add_argument('--embedding_trainable', '-et', type=bool, default=False,
                        help='Whether the word embedding is trainable')
    parser.add_argument('--version_index', '-vi', type=int, default=0,
                        help='The index for saving different model.')

    data_mode = ['tarin', 'valid', 'test']

    args = parser.parse_args()
    data_version = args.data_version
    embedding_trainable = args.embedding_trainable
    version_index = args.version_index
    GPU = args.gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = '../model_save/CL4LJP/CL4LJP_{}'.format(data_version)
    print('saving model:', model_path)

    # -------------------------------set_configs--------------------------#
    configFilePath = '../Config/CL4LJP_Criminal.config'
    config = ConfigParser(configFilePath)
    batch_size = config.getint('data', 'batch_size')
    max_epoch = config.getint('train', 'epoch')

    emb_path = '../Criminal_Dataset/WordEmbedding_Criminal.npy'

    with open('../Criminal_Dataset/word2id_Criminal', 'rb') as f:
        word2id_dict = pk.load(f)
        f.close()

    word_dict_len = len(word2id_dict)

    data_path = '/home/nxu/Ladan_tnnls/CL4LJP/processed_data/Criminal/'

    train_set = Criminal_dataset(read_dataset(data_path, data_version, 'train'), charge_num=149)
    valid_set = Criminal_dataset(read_dataset(data_path, data_version, 'valid'), charge_num=149)
    test_set = Criminal_dataset(read_dataset(data_path, data_version, 'test'), charge_num=149)

    if not isinstance(train_set, Criminal_dataset):
        print('Dataset has an error type.')

    Formatter = CriminalFormatter(config, mode='train')

    def collate_fn(data):
        return Formatter.process(data, "train")


    train_dataset = DataLoader(train_set, batch_size=128,
                               shuffle=True, drop_last=False,
                               collate_fn=collate_fn)

    valid_dataset = DataLoader(valid_set, batch_size=128,
                               shuffle=False, drop_last=False,
                               collate_fn=collate_fn)

    test_dataset = DataLoader(test_set, batch_size=128,
                              shuffle=False, drop_last=False,
                              collate_fn=collate_fn)

    model = CL4LJP_C(config=config, emb_path=emb_path, word2id_dict=word2id_dict, data_version='small', device=device)
    print("loading model from: ", model_path)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    learning_rate = 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-4, patience=4, verbose=True)
    best_acc = 0.0

    accu_pred_list = []
    accu_label_list = []
    print('test')
    with torch.no_grad():
        for data in tqdm(test_dataset):
            fact, charge_label = data['fact'].to(device), data['charge_label'].to(device)
            charge_pred = model(fact, charge_label, model_type="test")
            accu_pred_list.append(charge_pred.cpu().numpy())
            accu_label_list.append(charge_label.cpu().numpy())

    accu_preds = np.argmax(np.concatenate(accu_pred_list, axis=0), axis=1)
    accu_labels = np.concatenate(accu_label_list, axis=0)

    metric = evaluation_multitask([accu_labels], [accu_preds], 1)

    task = ['accu']
    print('Now_testing')
    ave_acc = 0
    for i in range(1):
        print('Metrics for {} prediction is: '.format(task[i]), metric[i])

    charge_low = [3, 7, 10, 14, 16, 22, 23, 26, 29, 32, 34, 35, 36, 40, 45, 58, 61, 63, 65, 67, 72, 74, 84, 85, 86, 89,
                  94, 98, 99, 100, 102, 105, 106, 108, 110, 112, 126, 129, 131, 133, 134, 135, 140, 141, 144, 145, 146,
                  147, 148]
    charge_medium = [1, 4, 5, 6, 13, 15, 18, 24, 25, 28, 30, 31, 37, 38, 39, 41, 42, 44, 46, 47, 51, 53, 54, 56, 57, 59,
                     62, 64, 66, 77, 80, 82, 83, 87, 88, 90, 96, 97, 101, 104, 109, 114, 116, 117, 120, 122, 123, 127,
                     132, 142, 143]
    charge_high = [0, 2, 8, 9, 11, 12, 17, 19, 20, 21, 27, 33, 43, 48, 49, 50, 52, 55, 60, 68, 69, 70, 71, 73, 75, 76,
                   78, 79, 81, 91, 92, 93, 95, 103, 107, 111, 113, 115, 118, 119, 121, 124, 125, 128, 130, 136, 137,
                   138, 139]

    print('Charge_Low:')
    eval_data_types(accu_labels, accu_preds, num_labels=149, label_list=charge_low)

    print('Charge_Medium:')
    eval_data_types(accu_labels, accu_preds, num_labels=149, label_list=charge_medium)

    print('Charge_High:')
    eval_data_types(accu_labels, accu_preds, num_labels=149, label_list=charge_high)



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
    parser.add_argument('--data_version', '-dv', type=str, default='Criminal_large',
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
    model.to(device)

    learning_rate = 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-4, patience=4, verbose=True)
    best_acc = 0.0
    for epoch in range(max_epoch):
        optim_step = 0
        tr_loss = 0
        task_loss = 0
        l_CL = 0
        model.train()
        optimizer.zero_grad()
        epoch_steps = 0
        print('epoch: {}/{}'.format(str(epoch), str(max_epoch)))
        for step, data in enumerate(tqdm(train_dataset)):
            fact, charge_label = data['fact'].to(device), data['charge_label'].to(device)
            out_dict = model(fact, charge_label, model_type="train")
            charge_loss, label_CL = out_dict['charge_loss'], out_dict['label_CL']
            loss = charge_loss + label_CL * 0.5
            tr_loss += loss.item()
            task_loss += charge_loss.item()
            l_CL += label_CL.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_steps += 1

        print('loss:', tr_loss / epoch_steps)
        print('task_loss:', task_loss / epoch_steps)
        print('label_CL:', l_CL / epoch_steps)

        accu_pred_list = []
        accu_label_list = []
        print('Valid')
        with torch.no_grad():
            for data in tqdm(valid_dataset):
                fact, charge_label = data['fact'].to(device), data['charge_label'].to(device)
                charge_pred = model(fact, charge_label, model_type="valid")
                accu_pred_list.append(charge_pred.cpu().numpy())
                accu_label_list.append(charge_label.cpu().numpy())

        accu_preds = np.argmax(np.concatenate(accu_pred_list, axis=0), axis=1)
        accu_labels = np.concatenate(accu_label_list, axis=0)

        metric = evaluation_multitask([accu_labels], [accu_preds], 1)

        task = ['accu']
        print('Now_validing')
        ave_acc = 0
        for i in range(1):
            print('Metrics for {} prediction is: '.format(task[i]), metric[i])
            ave_acc += metric[i][0]

        scheduler.step(metrics=ave_acc)
        if ave_acc >= best_acc:
            torch.save(model.state_dict(), model_path)
            print('saving model......' + model_path)
            best_acc = ave_acc

        accu_pred_list = []
        accu_label_list = []
        print('Test')
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


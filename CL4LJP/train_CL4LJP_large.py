import sys
sys.path.append("..")
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
from torch.optim import lr_scheduler


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
    parser.add_argument('--gpu', '-g', type=str, default='2',
                        help='Select which GPU device to use')
    parser.add_argument('--data_version', '-dv', type=str, default='big_data', choices=['data', 'big_data'],
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
    print('saving model:', model_path)

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

    train_set = read_dataset(data_path, data_version, 'train')
    valid_set = read_dataset(data_path, data_version, 'test')

    if not isinstance(train_set, CAIL_dataset):
        print('Dataset has an error type.')

    Formatter = CAILFormatter(config, mode='train')

    def collate_fn(data):
        return Formatter.process(data, "train")


    train_dataset = DataLoader(train_set, batch_size=256,
                               shuffle=True, drop_last=False,
                               collate_fn=collate_fn)

    valid_dataset = DataLoader(valid_set, batch_size=256,
                               shuffle=False, drop_last=False,
                               collate_fn=collate_fn)

    law_dict_path = '/home/nxu/Ladan_tnnls/CL4LJP/law2detail.json'
    law_index_path = '/home/nxu/Ladan_tnnls/big_data/new_law_big.txt'

    articles = get_articles(law_dict_path, law_index_path, word2id_dict, 1500)
    articles = articles.to(device)
    # print(articles)
    print(articles.shape)

    model = CL4LJP(config=config, emb_path=emb_path, word2id_dict=word2id_dict, data_version='large', device=device)
    model.to(device)

    learning_rate = 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-4, patience=2, verbose=True)
    best_acc = 0.0
    for epoch in range(max_epoch):
        tr_loss = 0
        model.train()
        optimizer.zero_grad()
        epoch_steps = 0
        print('epoch: {}/{}'.format(str(epoch), str(max_epoch)))
        for step, data in enumerate(tqdm(train_dataset)):
            fact, law_label, accu_label, time_label = \
                data['fact'].to(device), data['law_label'].to(device), \
                data['accu_label'].to(device), data['time_label'].to(device)

            pos_article_index, neg_article_index, neg_article_masks = \
                data['pos_article_index'].to(device), data['neg_article_index'].to(device), data['neg_article_masks'].to(device)

            pos_charge_fact, neg_charge_fact, neg_charge_masks = \
                data['pos_charge_fact'].to(device), data['neg_charge_fact'].to(device), data['neg_charge_masks'].to(device)

            out_dict = \
                model(fact, articles, accu_label, law_label, time_label, pos_article_index, neg_article_index,
                      pos_charge_fact, neg_charge_fact, neg_article_masks, neg_charge_masks, model_type="train")

            charge_loss, article_loss, time_loss, article_CL, charge_CL, label_CL = \
                out_dict['charge_loss'], out_dict['article_loss'], out_dict['time_loss'], out_dict['article_CL'], \
                out_dict['charge_CL'], out_dict['label_CL']

            loss = charge_loss + article_loss + time_loss + (article_CL + charge_CL + label_CL) * 0.5
            tr_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_steps += 1

        print('loss:', tr_loss / epoch_steps)

        law_pred_list = []
        law_label_list = []
        accu_pred_list = []
        accu_label_list = []
        time_pred_list = []
        time_label_list = []
        print('Valid')
        with torch.no_grad():
            for data in tqdm(valid_dataset):
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
                          pos_charge_fact, neg_charge_fact, neg_article_masks, neg_charge_masks, model_type="valid")

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
        print('Now_validing')
        ave_acc = 0
        for i in range(3):
            print('Metrics for {} prediction is: '.format(task[i]), metric[i])
            ave_acc += metric[i][0]

        ave_acc = ave_acc / 3.0
        if ave_acc >= best_acc:
            torch.save(model.state_dict(), model_path)
            print('saving model......' + model_path)
            best_acc = ave_acc
        # scheduler.step(metrics=ave_acc)
        train_set.shuffle_neg()

        # law_pred_list = []
        # law_label_list = []
        # accu_pred_list = []
        # accu_label_list = []
        # time_pred_list = []
        # time_label_list = []
        # print('Test')
        # with torch.no_grad():
        #     for data in tqdm(test_dataset):
        #         fact, law_label, accu_label, time_label = \
        #             data['fact'].to(device), data['law_label'].to(device), \
        #             data['accu_label'].to(device), data['time_label'].to(device)
        #
        #         pos_article_index, neg_article_index, neg_article_masks = \
        #             data['pos_article_index'].to(device), data['neg_article_index'].to(device), data[
        #                 'neg_article_masks'].to(device)
        #
        #         pos_charge_fact, neg_charge_fact, neg_charge_masks = \
        #             data['pos_charge_fact'].to(device), data['neg_charge_fact'].to(device), data['neg_charge_masks'].to(
        #                 device)
        #
        #         charge_pred, article_pred, time_pred = \
        #             model(fact, articles, accu_label, law_label, time_label, pos_article_index, neg_article_index,
        #                   pos_charge_fact, neg_charge_fact, neg_article_masks, neg_charge_masks, model_type="test")
        #
        #         law_pred_list.append(article_pred.cpu().numpy())
        #         law_label_list.append(law_label.cpu().numpy())
        #         accu_pred_list.append(charge_pred.cpu().numpy())
        #         accu_label_list.append(accu_label.cpu().numpy())
        #         time_pred_list.append(time_pred.cpu().numpy())
        #         time_label_list.append(time_label.cpu().numpy())
        #
        # law_preds = np.argmax(np.concatenate(law_pred_list, axis=0), axis=1)
        # law_labels = np.concatenate(law_label_list, axis=0)
        # accu_preds = np.argmax(np.concatenate(accu_pred_list, axis=0), axis=1)
        # accu_labels = np.concatenate(accu_label_list, axis=0)
        # time_preds = np.argmax(np.concatenate(time_pred_list, axis=0), axis=1)
        # time_labels = np.concatenate(time_label_list, axis=0)
        #
        # metric = evaluation_multitask([law_labels, accu_labels, time_labels],
        #                               [law_preds, accu_preds, time_preds], 3)
        #
        # task = ['law', 'accu', 'time']
        # print('Now_testing')
        # ave_acc = 0
        # for i in range(3):
        #     print('Metrics for {} prediction is: '.format(task[i]), metric[i])



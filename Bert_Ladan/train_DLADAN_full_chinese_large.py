import sys
# sys.path.append("..")
import argparse
import gc, json, os, torch, random
import numpy as np
import pickle as pkl
from DLADAN_bert import DLADAN_Bert_full
from types import SimpleNamespace
from transformers import BertTokenizer, BertConfig, TFBertModel
from torch.utils.data import RandomSampler
from tqdm import tqdm
from torch.optim import AdamW, Adam
from torch.optim import lr_scheduler
from timeit import default_timer as timer
from Dataloaders import DladanCailLoader
from Formatters import SentenceFormatter_D
from torch.utils.data import DataLoader, RandomSampler, Dataset
from sklearn import metrics
import torch.nn as nn
from law_processing import get_law_graph_adj, get_law_graph_large_adj


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(group_indexes):
    train_path = '/home/nxu/Ladan_tnnls/Bert_Ladan/processed_dataset/CAIL_new/full_doc/large/train_bert_chinese.pkl'
    test_path = '/home/nxu/Ladan_tnnls/Bert_Ladan/processed_dataset/CAIL_new/full_doc/large/test_bert_chinese.pkl'
    
    train_Dataset = DladanCailLoader(data_path=train_path, group_indexes=group_indexes, mode='train')
    test_Dataset = DladanCailLoader(data_path=test_path, group_indexes=group_indexes, mode='test')
    
    Formatter = SentenceFormatter_D(mode='train')
    def collate_fn(data):
        return Formatter.process(data, "train")
    train_dataset = DataLoader(dataset=train_Dataset, batch_size=128,
                               num_workers=1, drop_last=True,
                            #    shuffle=True, 
                               collate_fn=collate_fn,
                               sampler=RandomSampler(train_Dataset)
                               )
    
    test_dataset = DataLoader(dataset=test_Dataset, batch_size=224,
                               num_workers=1, drop_last=False,
                               shuffle=False, collate_fn=collate_fn,
                               # sampler=RandomSampler(Dataset)
                               )
    return train_dataset, test_dataset


def evaluation_matching(y_true, y_pred):
    accuracy_metric = metrics.accuracy_score(y_true, y_pred)
    macro_recall = metrics.recall_score(y_true, y_pred, average='macro')
    micro_recall = metrics.recall_score(y_true, y_pred, average='micro')
    macro_precision = metrics.precision_score(y_true, y_pred, average='macro')
    micro_precision = metrics.precision_score(y_true, y_pred, average='micro')
    macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')
    micro_f1 = metrics.f1_score(y_true, y_pred, average='micro')
    metrics_acc = \
        [accuracy_metric, macro_recall, micro_recall, macro_precision, micro_precision, macro_f1, micro_f1]
    return metrics_acc


def train_loop(model: DLADAN_Bert_full, train_dataset, valid_dataset, 
               epoches, scheduler, model_save_path, 
               warming_up_epoches, momentum_steps=1):
    
    best_f1 = 0

    warming_up = False
    synchronize_memory = False
    momentum_flag = False

    if warming_up_epoches != 0: # 设置warming up的memory 更新机制
        warming_up = True
        synchronize_memory = False
        momentum_flag = False
        # warming_up = False
        # momentum_flag = True
    
    for epoch_num in range(epoches):
        start_time = timer()
        currentt_epoch = epoch_num
        model.train()
        total_loss = 0
        total_loss_law = 0
        total_loss_accu = 0
        total_loss_time = 0
        total_loss_group_pri = 0
        total_loss_group_post = 0
        total_loss_group_post_a = 0
        epoch_steps = 0
        tqdm_obj = tqdm(train_dataset, total=len(train_dataset), position=0, ncols=100)
        for step, data in enumerate(tqdm_obj):
            if epoch_num < warming_up_epoches: pass
            elif epoch_num == warming_up_epoches and step == 0: # 设置 warming_up结束后的memory 同步机制
                warming_up = False
                synchronize_memory = True
                momentum_flag = False
            elif epoch_num == warming_up_epoches and step == 1: # memory同步后开始进行memory的动量更新机制
                warming_up = False
                synchronize_memory = False
                momentum_flag = True
            elif step % momentum_steps == 0:
                if model.momentum_flag == False:
                    momentum_flag = True
            else:
                if model.momentum_flag == True:
                    momentum_flag = False
                
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = data[key].cuda()
                    else:
                        data[key] = data[key]
            inputs = [data['inputx'], data['masks'], data['segments']]
            labels = [data['law_label'], data['accu_label'], data['time_label'], data['group_label']]

            out_dict = model(inputs=inputs, labels=labels, model_type='train', synchronize_memory=synchronize_memory, 
                             warming_up=warming_up, momentum_flag=momentum_flag)
            loss = out_dict['loss']
            total_loss += float(loss)
            total_loss_law += float(out_dict['loss_law'])
            total_loss_accu += float(out_dict['loss_charge'])
            total_loss_time += float(out_dict['loss_time'])
            
            total_loss_group_pri += float(out_dict['loss_graph_pri'])
            total_loss_group_post += float(out_dict['loss_graph_post'])
            total_loss_group_post_a += float(out_dict['loss_graph_post_a'])
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_steps += 1
            tqdm_obj.set_description('loss_g_pri:{:.6f},'.format(float(out_dict['loss_graph_pri'])) + 
                                     'loss_g_post:{:.6f},'.format(float(out_dict['loss_graph_post'])) + 
                                     'loss_g_post_a:{:.6f}'.format(float(out_dict['loss_graph_post_a'])))
            
        print('loss:', total_loss / epoch_steps, 
              'loss_law:', total_loss_law / epoch_steps, 
              'loss_accu:', total_loss_accu / epoch_steps, 
              'loss_time:', total_loss_time / epoch_steps,
              'loss_group_pri:', total_loss_group_pri / epoch_steps, 
              'loss_group_post:', total_loss_group_post / epoch_steps, 
              'loss_group_post_a:', total_loss_group_post_a / epoch_steps,)
        
        print('epoch:', epoch_num)
        model.eval()
        final_f1 = valid_loop(model, valid_dataset, mode="Valid")
        scheduler.step(metrics=final_f1)
        
        if final_f1 >= best_f1:
            torch.save(model.state_dict(), model_save_path)
            print('saving model......' + model_save_path)
            best_f1 = final_f1


def valid_loop(model, valid_dataset, mode='Valid'):
    pred_law_list = []
    pred_accu_list = []
    pred_time_list = []
    
    law_label_list = []
    accu_label_list = []
    time_label_list = []
    
    with torch.no_grad():
        for data in tqdm(valid_dataset, ncols=100):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = data[key].cuda()
                    else:
                        data[key] = data[key]
            inputs = [data['inputx'], data['masks'], data['segments']]
            labels = [data['law_label'], data['accu_label'], data['time_label'], data['group_label']]
            out_dict = model(inputs=inputs, labels=labels, model_type=mode, synchronize_memory=False, warming_up=False, momentum_flag=False)
            
            pred_law_list.append(out_dict['law_pred'].cpu().numpy())
            pred_accu_list.append(out_dict['charge_pred'].cpu().numpy())
            pred_time_list.append(out_dict['time_pred'].cpu().numpy())
            
            law_label_list.append(data['law_label'].long().cpu().numpy())
            accu_label_list.append(data['accu_label'].long().cpu().numpy())
            time_label_list.append(data['time_label'].long().cpu().numpy())
            del out_dict
            
    preds_law = np.argmax(np.concatenate(pred_law_list, axis=0), axis=1).tolist()
    preds_accu = np.argmax(np.concatenate(pred_accu_list, axis=0), axis=1).tolist()
    preds_time = np.argmax(np.concatenate(pred_time_list, axis=0), axis=1).tolist()
    
    law_labels = np.concatenate(law_label_list, axis=0).tolist()
    accu_labels = np.concatenate(accu_label_list, axis=0).tolist()
    time_labels = np.concatenate(time_label_list, axis=0).tolist()
    
    metrics_law = evaluation_matching(law_labels, preds_law)
    metrics_accu = evaluation_matching(accu_labels, preds_accu)
    metric_time = evaluation_matching(time_labels, preds_time)
    
    print('\nNow {}ing model'.format(mode))
    print('Law prediction task -- ','acc:', metrics_law[0], 
          'ma_recall:', metrics_law[1], 'mi_recall:', metrics_law[2], 
          'ma_precision:', metrics_law[3], 'mi_precision:', metrics_law[4], 
          'ma_F1:', metrics_law[5], 'mi_F1:', metrics_law[6])
    print('Charge prediction task -- ','acc:', metrics_accu[0], 
          'ma_recall:', metrics_accu[1], 'mi_recall:', metrics_accu[2], 
          'ma_precision:', metrics_accu[3], 'mi_precision:', metrics_accu[4], 
          'ma_F1:', metrics_accu[5], 'mi_F1:', metrics_accu[6])
    print('Time prediction task -- ','acc:', metric_time[0], 
          'ma_recall:', metric_time[1], 'mi_recall:', metric_time[2], 
          'ma_precision:', metric_time[3], 'mi_precision:', metric_time[4], 
          'ma_F1:', metric_time[5], 'mi_F1:', metric_time[6])
    
    F1_final = metrics_law[5] / 8.7 + metrics_accu[5] / 8.4 + metric_time[5] / 4.0
    
    return F1_final


if __name__ == "__main__":
    set_seed(666)
    
    # 0. prepare the device 
    gpu_list = [0, 1, 2, 3, 4, 5, 6, 7]
    gpu_list_str = ','.join(map(str, gpu_list))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list_str
    gpu_ids = [i for i in range(len(gpu_list))]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    
    # 1. load dataset
    config_file= '../Config/DLADAN_Bert_config.json'
    with open(config_file) as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    warming_up_epoch = 1
        
    law_inputs, group_list, graph_membership, law_adj_matrix = \
        get_law_graph_large_adj(threshold=0.35, sent_len=config.train.law_sentence_len,
                                pretrained_bert_fold=config.train.pretrain_model_path_1)
    group_indexes = list(zip(*graph_membership))[1]
    group_num = len(group_list)
    train_dataset, test_dataset = load_dataset(group_indexes)
    
    # 2. define model 
    model:DLADAN_Bert_full = \
        DLADAN_Bert_full(config=config, group_num=group_num, group_indexes=group_indexes,
                         law_input=law_inputs, law_adj_matrix=law_adj_matrix, accu_relation=True, mode='large')
    
    model.init_multi_gpu(device=gpu_ids)
    model = model.to(device)
    
    learning_rate = config.train.learning_rate
    print('learning_rate:', learning_rate)
    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
    bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': 1e-5}
    # optimizer = Adam([non_bert_params, bert_params], lr=3e-5)
    optimizer = AdamW([non_bert_params, bert_params], lr=2e-5, weight_decay=config.train.weight_decay)
    # optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=config.train.weight_decay)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, min_lr=1e-6, patience=2, verbose=True)
    epoches = config.train.epoch
    
    # 3. Training loop
    model_save_path = '/home/nxu/Ladan_tnnls/Bert_Ladan/model_save/CAIL_small/DLADAN_BERT_full_chinese_large.hdf5'
    train_loop(model=model, train_dataset=train_dataset, valid_dataset=test_dataset,
               epoches=epoches, scheduler=scheduler, model_save_path=model_save_path,
               warming_up_epoches=1, momentum_steps=1)

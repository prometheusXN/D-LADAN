# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# from tools.accuracy_init import init_accuracy_function


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.hidden_dim = config.getint('rnn', 'hidden_dim')
        self.fa = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        pass

    def forward(self, feature: Tensor, contexts: Tensor, masks: Tensor):
        # hidden: B * M * H, contexts: B * H
        ratio = torch.matmul(torch.tanh(self.fc(feature)), self.fa(contexts).unsqueeze(-1))
        # ratio: B * M * 1
        max_ratio = torch.max(ratio, dim=-1).values  # [B, M]
        max_ratio = max_ratio * masks + (masks - 1) * 1e9
        attention_score = F.softmax(max_ratio, dim=-1).unsqueeze(-1)  # [B, M, 1]
        result = torch.sum(attention_score * feature, dim=1)  # [B, H]
        return result


class AttentionRNN(nn.Module):
    def __init__(self, config, *args, **params):
        super(AttentionRNN, self).__init__()

        self.input_dim = config.getint('rnn', 'input_dim')
        self.hidden_dim = config.getint('rnn', 'hidden_dim')
        self.dropout_rnn = config.getfloat('rnn', 'dropout_rnn')
        self.dropout_fc = config.getfloat('rnn', 'dropout_fc')
        self.bidirectional = config.getboolean('rnn', 'bidirectional')
        if self.bidirectional:
            self.direction = 2
        else:
            self.direction = 1
        self.num_layers = config.getint("rnn", 'num_layers')
        self.output_dim = config.getint("rnn", "output_dim")
        self.sentence_num = config.getint('data', 'sentence_num')
        self.mode = config.get('rnn', 'mode')
        self.model_mode = config.get('model', 'mode')
        self.aggregate_mode = config.get('rnn', 'aggregate_mode')
        print('Records %d layers RNN' % self.num_layers)
        print('Records a %s input with %s' % (self.model_mode, self.aggregate_mode))
        if config.get('rnn', 'mode') == 'lstm':
            self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layers,
                               bidirectional=self.bidirectional, dropout=self.dropout_rnn)
        else:
            self.rnn = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout_rnn)
        self.fc_a = nn.Linear(self.hidden_dim*self.direction, self.hidden_dim*self.direction)
        self.attention = Attention(config)
        self.dropout = nn.Dropout(self.dropout_fc)

    def init_weight(self, config, gpu_list):
        try:
            label_weight = config.getfloat('model', 'label_weight')
        except Exception:
            return None
        weight_lst = torch.ones(self.output_dim)
        weight_lst[-1] = label_weight
        if torch.cuda.is_available() and len(gpu_list) > 0:
            weight_lst = weight_lst.cuda()
        return weight_lst

    def init_multi_gpu(self, device, *args, **params):
        self.rnn = nn.DataParallel(self.rnn, device_ids=device)
        self.fc_a = nn.DataParallel(self.fc_a, device_ids=device)
        self.attention = nn.DataParallel(self.attention, device_ids=device)

    def forward(self, input_sequence: torch.Tensor, masks: torch.Tensor, weights: torch.Tensor = None, mode = 'Legal'):
        """
        :param input_sequence: [batch_size, sentence_num, feature_dim]
        :param masks: [batch_size, sentence_num]
        :param weights: [batch_size, sentence_num]

        :return:
        """
        self.rnn.flatten_parameters()
        x = input_sequence  # B * M * Input_dim
        docs_length = masks.sum(dim=-1).long().tolist()
        x = nn.utils.rnn.pack_padded_sequence(x, docs_length, enforce_sorted=False, batch_first=True)
        rnn_out, hidden = self.rnn(x)  # rnn_out: B * M * 2H, hidden: 2 * B * H
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True, total_length=input_sequence.size()[1])
        rnn_out = rnn_out * masks.unsqueeze(dim=-1)

        if self.aggregate_mode == "mean_pooling":
            sentence_num = torch.sum(masks, dim=-1, keepdim=True).float()
            feature = torch.sum(rnn_out, dim=1) / sentence_num
        else:
            rnn_out = rnn_out + (masks.unsqueeze(-1) - 1) * 1e9
            feature = torch.max(rnn_out, dim=1).values

        # feature = torch.mean(rnn_out, dim=1)
        if mode == 'Legal':
            if weights is not None:
                weights = weights * masks + (masks - 1) * 1e9
                weights = F.softmax(weights, dim=-1).unsqueeze(-1)  # [batch_size, sentence_num, 1]
                feature = (rnn_out * weights).sum(dim=1)    # [batch_size, 2 * hidden_size]
                atten_out = self.fc_a(feature)
            else:
                feature = self.fc_a(feature)  # B * 2H
                atten_out = feature
        else:
            assert mode == 'Semantic'
            atten_out = self.attention(rnn_out, feature, masks)  # B * (2H)
            atten_out = self.fc_a(atten_out)
        atten_out = self.dropout(atten_out)
        return atten_out


class ContextAttention(nn.Module):
    def __init__(self, config, input_dim, with_value_fn=True):
        super(ContextAttention, self).__init__()
        self.hidden_dim = config.net.hidden_size
        self.fk = nn.Linear(input_dim, self.hidden_dim)
        if with_value_fn:
            self.fv = nn.Linear(input_dim, self.hidden_dim)
        self.fq = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.activate = nn.GELU()
        self.scale = self.hidden_dim ** (-0.5)
        self.dropout = nn.Dropout(config.rnn.dropout)
        self.with_value_fn = with_value_fn

    def forward(self, feature: Tensor, contexts: Tensor, masks: Tensor):
        # hidden: B * M * H, contexts: B * H
        ratio = torch.matmul(self.fk(feature), self.fq(contexts).unsqueeze(-1))
        # ratio: B * M * 1
        ratio = ratio * self.scale
        if self.with_value_fn:
            value = self.fv(feature)
        else:
            value = feature
        max_ratio = ratio.squeeze()   # [B, M]
        max_ratio = max_ratio * masks + (masks - 1) * 1e9
        attention_score = F.softmax(max_ratio, dim=-1).unsqueeze(-1)    # [B, M, 1]
        attention_score = self.dropout(attention_score)
        result = torch.sum(attention_score * value, dim=-2)    # [B, H]
        # result = self.activate(result)
        return result, attention_score


class ContextRNN(nn.Module):
    def __init__(self, config, *args, **params):
        super(ContextRNN, self).__init__()

        self.input_dim = config.getint('rnn', 'input_dim')
        self.hidden_dim = config.getint('rnn', 'hidden_dim')
        self.dropout_rnn = config.getfloat('rnn', 'dropout_rnn')
        self.dropout_fc = config.getfloat('rnn', 'dropout_fc')
        self.bidirectional = config.getboolean('rnn', 'bidirectional')
        if self.bidirectional:
            self.direction = 2
        else:
            self.direction = 1
        self.num_layers = config.getint("rnn", 'num_layers')
        self.output_dim = config.getint("rnn", "output_dim")
        self.sentence_num = config.getint('data', 'sentence_num')
        self.mode = config.get('rnn', 'mode')
        self.model_mode = config.get('model', 'mode')
        print('Records %d layers RNN' % self.num_layers)
        print('Records a %s input' % self.model_mode)
        if config.get('rnn', 'mode') == 'lstm':
            self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layers,
                               bidirectional=self.bidirectional, dropout=self.dropout_rn)
        else:
            self.rnn = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout_rnn)
        self.attention = ContextAttention(config)
        self.fc_a = nn.Linear(self.hidden_dim*self.direction, self.hidden_dim*self.direction)
        self.dropout = nn.Dropout(self.dropout_fc)

    def forward(self, input_sequence: Tensor, masks: Tensor, contexts: Tensor):
        """
        :param input_sequence: [batch_size, sentence_num, feature_dim]
        :param masks: [batch_size, sentence_num]
        :param contexts: [batch_size, feature_dim]
        :return:
        """
        self.rnn.flatten_parameters()
        x = input_sequence  # B * M * Input_dim
        docs_length = masks.sum(dim=-1).long().tolist()
        x = nn.utils.rnn.pack_padded_sequence(x, docs_length, enforce_sorted=False, batch_first=True)
        rnn_out, hidden = self.rnn(x)  # rnn_out: B * M * 2H, hidden: 2 * B * H
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True, total_length=input_sequence.size()[1])
        rnn_out = rnn_out * masks.unsqueeze(dim=-1) + (masks.unsqueeze(-1) - 1) * 1e9
        rnn_out = self.dropout(rnn_out)
        atten_out, attention_score = self.attention(rnn_out, contexts, masks)
        atten_out = self.fc_a(atten_out)
        # feature = torch.cat([feature, torch.mean(rnn_out, dim=1)], dim=-1)
        return atten_out, attention_score


class RNN(nn.Module):
    def __init__(self, config, input_dim=None, hidden_dim=None, *args, **params):
        super(RNN, self).__init__()

        self.input_dim = config.rnn.input_dim
        self.hidden_dim = config.rnn.hidden_dim
        if input_dim is not None:
            self.input_dim = input_dim
        if hidden_dim is not None:
            self.hidden_dim = hidden_dim
        self.dropout_rnn = config.rnn.dropout
        self.dropout_fc = config.rnn.dropout
        self.bidirectional = bool(config.rnn.bidirectional)
        if self.bidirectional:
            self.direction = 2
        else:
            self.direction = 1
        self.num_layers = config.rnn.num_layers
        print('Records %d layers RNN' % self.num_layers)
        if config.rnn.mode == 'lstm':
            self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layers,
                               bidirectional=self.bidirectional, dropout=self.dropout_rnn)
        else:
            self.rnn = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True, num_layers=self.num_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout_rnn)
        self.dropout = nn.Dropout(self.dropout_fc)

    def forward(self, input_sequence: Tensor, masks: Tensor):
        """
        :param input_sequence: [batch_size, sentence_num, feature_dim]
        :param masks: [batch_size, sentence_num]
        :return:
        """
        self.rnn.flatten_parameters()
        x = input_sequence  # B * M * Input_dim
        docs_length = masks.sum(dim=-1).long().tolist()
        for i in range(len(docs_length)):
            if docs_length[i] == 0:
                docs_length[i] = 1
        x = nn.utils.rnn.pack_padded_sequence(x, docs_length, enforce_sorted=False, batch_first=True)
        rnn_out, hidden = self.rnn(x)  # rnn_out: B * M * 2H, hidden: 2 * B * H
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True, total_length=input_sequence.size()[1])
        rnn_out = rnn_out * masks.unsqueeze(dim=-1)
        rnn_out = self.dropout(rnn_out)
        # [batch_size, sentence_num, 2* hidden_dim]
        return rnn_out


class AttentionOriContext(nn.Module):
    def __init__(self, config, input_dim, with_value_fn=True):
        super(AttentionOriContext, self).__init__()
        self.hidden_dim = config.net.hidden_size
        self.fk = nn.Linear(input_dim, self.hidden_dim)
        if with_value_fn:
            self.fv = nn.Linear(input_dim, self.hidden_dim)
        self.context = nn.Parameter(torch.rand(1, self.hidden_dim))
        # self.activation = nn.
        self.scale = self.hidden_dim ** (-0.5)
        self.dropout = nn.Dropout(config.rnn.dropout)
        self.with_value_fn = with_value_fn
        

    def forward(self, feature: Tensor, masks: Tensor):
        # feature: B * M * H, contexts: B * H
        key = self.fk(feature)
        if self.with_value_fn:
            value = self.fv(feature)
        else:
            value = feature
        ratio = torch.sum(key * self.context.unsqueeze(dim=1), dim=-1) * self.scale
        max_ratio = ratio * masks + (masks - 1) * 1e9
        attention_score = F.softmax(max_ratio, dim=-1).unsqueeze(-1)  # [B, M, 1]
        attention_score = self.dropout(attention_score)
        result = torch.sum(attention_score * value, dim=1)  # [B, H]
        
        return result, attention_score

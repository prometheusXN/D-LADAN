import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from torch.autograd import Variable
import numpy as np
from CL4LJP.model_component.CNN_Cell import CNNEncoder, CNNEncoder2D

epsilon = 1e-16


def supervised_CL(features, one_hot_labels, reduction="mean"):
    batch_size = features.shape[0]
    similarity_matrix = F.cosine_similarity(features.unsqueeze(dim=1), features.unsqueeze(dim=0), dim=-1) * 0.5 + 0.5
    mask = F.cosine_similarity(one_hot_labels.unsqueeze(dim=1), one_hot_labels.unsqueeze(dim=0), dim=-1)
    self_loop = torch.eye(batch_size, dtype=torch.float).cuda()  # [batch_size, batch_size]

    mask_all = 1.0 - self_loop

    similarity_matrix = torch.exp(similarity_matrix * 5.0) * mask_all

    sim = mask * similarity_matrix  # [batch_size, batch_size]
    no_sim = similarity_matrix - sim

    no_sim_sum = torch.sum(no_sim, dim=1)

    no_sim_sum_expend = no_sim_sum.repeat(batch_size, 1).T
    sim_sum = sim + no_sim_sum_expend
    loss = torch.div(sim, sim_sum)
    loss = loss + mask_all + self_loop
    loss = -torch.log(loss)  # [batch_size, batch_size]
    loss = torch.sum(torch.sum(loss, dim=1)) / (mask-self_loop).sum()
    return loss


def contrastive_loss(positive_matrix, negative_matrix, masks: torch.Tensor, temperature, reduction: str ='mean'):
    """
    :param positive_matrix: [batch_size, 1]
    :param negative_matrix: [batch_size, neg_num]
    :param masks:
    :param reduction:
    :return:
    """
    positive = torch.exp(positive_matrix * temperature).sum(dim=-1)
    negative = torch.exp(negative_matrix * temperature).sum(dim=-1)

    loss = -torch.log(positive / (positive + negative + epsilon))
    loss = loss * masks
    if masks.sum() == 0.0:
        return torch.tensor([0.0]).cuda()
    else:
        if reduction == 'mean':
            return loss.sum() / masks.sum()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss.sum() / masks.sum()


def contrastive_loss_mini_batch(features: torch.Tensor, one_hot_labels: torch.Tensor, temperature, reduction: str ='mean'):
    """
    :param features: [batch_size, feature_dim]
    :param one_hot_labels: [batch_size, law_num]
    :param reduction:
    :return:
    """
    batch_size = features.shape[0]
    self_loop = torch.eye(batch_size, dtype=torch.float).cuda()  # [batch_size, batch_size]

    similarity_matrix = \
        torch.cosine_similarity(features.unsqueeze(dim=1), features.unsqueeze(dim=0), dim=-1) * 0.5 + 0.5
    similarity_matrix = temperature * similarity_matrix
    mask_positive = \
        torch.cosine_similarity(one_hot_labels.unsqueeze(dim=1), one_hot_labels.unsqueeze(dim=0), dim=-1) - self_loop
    # [batch_size, batch_size]

    law_mask = mask_positive.sum(dim=-1).bool()  # [batch_size, ]
    mask_all = 1.0 - self_loop

    loss = -torch.log(((torch.exp(similarity_matrix) * mask_positive).sum(dim=-1) + epsilon) / (torch.exp(similarity_matrix) * mask_all).sum(dim=-1))
    loss = loss * law_mask.float()

    if law_mask.sum() == 0.0:
        return torch.tensor([0.0]).cuda()
    else:
        if reduction == 'mean':
            return loss.sum() / law_mask.float().sum()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss.sum() / law_mask.float().sum()


class CL4LJP(nn.Module):
    def __init__(self, config, emb_path, word2id_dict, data_version, device=None, embedding_trainable=False):
        super(CL4LJP, self).__init__()
        self.config = config
        self.emb_path = emb_path
        self.word2id_dict = word2id_dict
        self.word_dict_size = len(word2id_dict)
        self.embedding_dim = config.getint('data', 'vec_size')
        self.embedding_trainable = embedding_trainable
        self.law_num = config.getint('num_class_{}'.format(data_version), 'law_num')
        self.accu_num = config.getint('num_class_{}'.format(data_version), 'accu_num')
        self.time_num = config.getint('num_class_{}'.format(data_version), 'time_num')
        self.hidden_size = config.getint('net', 'hidden_size')
        print('law_num:', self.law_num)
        print('accu_num:', self.accu_num)
        print('time_num:', self.time_num)

        one_hot_law = torch.eye(self.law_num, dtype=torch.float)
        one_hot_charge = torch.eye(self.accu_num, dtype=torch.float)
        self.register_buffer('one_hot_law', one_hot_law)
        self.register_buffer('one_hot_charge', one_hot_charge)

        self.task_loss = nn.CrossEntropyLoss(reduction='mean')
        self.sample_loss = nn.CrossEntropyLoss(reduction='mean')

        self.contrastive_loss = contrastive_loss
        self.AL_contrastive_loss = contrastive_loss_mini_batch
        self.article_scalar = nn.Parameter(torch.tensor(2.0, dtype=torch.float), requires_grad=True)
        self.charge_scalar = nn.Parameter(torch.tensor(2.0, dtype=torch.float), requires_grad=True)
        self.AL_scalar = nn.Parameter(torch.tensor(2.0, dtype=torch.float), requires_grad=True)
        self.CL_scalar = nn.Parameter(torch.tensor(2.0, dtype=torch.float), requires_grad=True)
        # self.AL_contrastive_loss = supervised_CL

        embedding_matrix = np.cast[np.float32](np.load(self.emb_path))
        self.embedding_layer = nn.Embedding(self.word_dict_size, self.embedding_dim)
        self.embedding_layer.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix))
        self.embedding_layer.weight.requires_grad = self.embedding_trainable

        self.general_encoder = CNNEncoder(config=self.config)
        self.contrastive_encoder = CNNEncoder(config=self.config)

        # self.general_encoder = CNNEncoder2D(config=self.config)
        # self.contrastive_encoder = CNNEncoder2D(config=self.config)
        if device is not None:
            self.general_encoder.to(device)
            self.contrastive_encoder.to(device)

        self.charge_m = nn.Linear(self.general_encoder.feature_len * 2, self.hidden_size)
        self.article_m = nn.Linear(self.general_encoder.feature_len * 2, self.hidden_size)
        self.time_m = nn.Linear(self.general_encoder.feature_len * 2, self.hidden_size)
        self.charge_pred = nn.Linear(self.hidden_size, self.accu_num)
        self.article_pred = nn.Linear(self.hidden_size, self.law_num)
        self.time_pred = nn.Linear(self.hidden_size, self.time_num)
        self.dropout = nn.Dropout(p=0.5)

    @ staticmethod
    def cosine_similarity(a, b, dim: int = -1):
        """
        :param a: [batch_size, ..., x, 1, feature_dim]
        :param b: [batch_size, ..., 1, y, feature_dim]
        :param dim:
        :return: [batch_size, ..., x, y]
        """
        dim_num = len(a.shape)
        a = a.unsqueeze(dim=dim_num - 1)  # [..., batch_size, 1, feature_dim]
        b = b.unsqueeze(dim=dim_num - 2)  # [..., 1, batch_size, feature_dim]
        sim_matrix = torch.cosine_similarity(a, b, dim=dim)  # [..., batch_size, batch_size]
        return sim_matrix

    def forward(self, fact, article_list, charge_label, article_label, time_label,
                article_pos, article_neg: torch.Tensor, charge_pos, charge_neg: torch.Tensor,
                article_neg_masks, charge_neg_masks, model_type='train'):
        """
        :param fact: [batch_size, sentence_length]
        :param article_list: [article_num, sentence_length]
        :param charge_label:[batch_size, ]
        :param article_label:
        :param time_label:

        :param article_pos: [batch_size, ]
        :param article_neg: [batch_size, negative_num]
        :param charge_pos: [batch_size, sentence_length]
        :param charge_neg: [batch_size, negative_num, sentence_length]
        :param article_neg_masks: record whether the corresponding sample have the CL loss of article strategy
        :param charge_neg_masks:

        :param model_type:
        :return:
        """
        batch_size, negative_num = article_neg.shape
        charge_neg = charge_neg.reshape([batch_size * negative_num, -1]).squeeze()
        article_neg = article_neg.reshape([batch_size * negative_num, -1]).squeeze()
        # [batch_size * negative_num, feature_dim] and [batch_size * negative_num, ]
        fact_embedding = self.embedding_layer(fact)

        fact_general = self.general_encoder(fact_embedding)
        fact_contrastive: torch.Tensor = self.contrastive_encoder(fact_embedding)
        # [batch_size, feature_len]

        if model_type == 'train':
            fact_general = self.dropout(fact_general)
            fact_contrastive = self.dropout(fact_contrastive)

        fact_rep = torch.cat([fact_general, fact_contrastive], dim=-1)


        charge_pred = self.charge_pred(F.relu(self.charge_m(fact_rep)))
        article_pred = self.article_pred(F.relu(self.article_m(fact_rep)))
        time_pred = self.time_pred(F.relu(self.time_m(fact_rep)))
        #
        # charge_pred = self.charge_pred(fact_rep)
        # article_pred = self.article_pred(fact_rep)
        # time_pred = self.time_pred(fact_rep)

        if model_type != 'train':
            return charge_pred, article_pred, time_pred
        else:
            article_embedding = self.embedding_layer(article_list)
            charge_pos_embedding = self.embedding_layer(charge_pos)
            charge_neg_embedding = self.embedding_layer(charge_neg)
            charges_pos = self.contrastive_encoder(charge_pos_embedding)
            charges_neg = self.contrastive_encoder(charge_neg_embedding).reshape([batch_size, negative_num, -1])
            charges_pos = self.dropout(charges_pos)
            charges_neg = self.dropout(charges_neg)
            # [batch_size, negative_num, feature_dim]

            articles = self.contrastive_encoder(article_embedding)
            articles = self.dropout(articles)
            articles_pos = torch.index_select(articles, dim=0, index=article_pos)
            # [batch_size, feature_dim]
            articles_neg = torch.index_select(articles, dim=0, index=article_neg).reshape([batch_size, negative_num, -1])
            # [batch_size, negative_num, feature_dim]

            charge_pos_similarity = torch.cosine_similarity(fact_contrastive, charges_pos, dim=-1).unsqueeze(dim=-1) * 0.5 + 0.5
            charge_neg_similarity = torch.cosine_similarity(fact_contrastive.unsqueeze(dim=1), charges_neg, dim=-1) * 0.5 + 0.5

            article_pos_similarity = torch.cosine_similarity(fact_contrastive, articles_pos, dim=-1).unsqueeze(dim=-1) * 0.5 + 0.5
            article_neg_similarity = torch.cosine_similarity(fact_contrastive.unsqueeze(dim=1), articles_neg, dim=-1) * 0.5 + 0.5
            # [batch_size, negative_num]

            charge_loss = self.task_loss(charge_pred, charge_label)
            article_loss = self.task_loss(article_pred, article_label)
            time_loss = self.task_loss(time_pred, time_label)

            law_label_OH = torch.index_select(self.one_hot_law, dim=0, index=article_label)  # [batch_size, law_num]
            charge_label_OH = torch.index_select(self.one_hot_charge, dim=0, index=charge_label)
            label_AL = self.AL_contrastive_loss(fact_contrastive, law_label_OH, self.AL_scalar, reduction='mean')
            label_CL = self.AL_contrastive_loss(fact_contrastive, charge_label_OH, self.CL_scalar, reduction='mean')

            article_CL = self.contrastive_loss(article_pos_similarity, article_neg_similarity, article_neg_masks, self.article_scalar)
            charge_CL = self.contrastive_loss(charge_pos_similarity, charge_neg_similarity, charge_neg_masks, self.charge_scalar)

            return {
                'charge_loss': charge_loss,
                'article_loss': article_loss,
                'time_loss': time_loss,
                'article_CL': article_CL,
                'charge_CL': charge_CL,
                'label_AL': label_AL,
                'label_CL': label_CL,
            }

    def infer(self, fact):
        fact_embedding = self.embedding_layer(fact)
        fact_general = self.general_encoder(fact_embedding)
        fact_contrastive: torch.Tensor = self.contrastive_encoder(fact_embedding)
        fact_rep = torch.cat([fact_general, fact_contrastive], dim=-1)

        charge_pred = self.charge_pred(fact_rep)
        article_pred = self.article_pred(fact_rep)
        time_pred = self.time_pred(fact_rep)

        return charge_pred, article_pred, time_pred




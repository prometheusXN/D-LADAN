import torch
import numpy as np
import math
from torch import Tensor
import pickle as pkl
from scipy.optimize import minimize
import scipy


def dynamic_partition(data: torch.Tensor, partitions: torch.Tensor, num_partitions=None):
    assert len(partitions.shape) == 1, "Only one dimensional partitions supported"
    assert (data.shape[0] == partitions.shape[0]), "Partitions requires the same size as data"

    if num_partitions is None:
        num_partitions = max(torch.unique(partitions))

    return [data[partitions == i] for i in range(num_partitions)]


def partition_list(data, partitions, num_partitions=None):
    assert len(np.array(partitions).shape) == 1, "Only one dimensional partitions supported"
    assert (len(data) == len(partitions)), "Partitions requires the same size as data"
    if num_partitions is None:
        num_partitions = np.max(np.array(partitions)) + 1
        print(num_partitions)
    partitions = np.array(partitions)
    data = np.array(data)

    return [data[partitions == i] for i in range(num_partitions)]


def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()


def get_masks(y_true: Tensor, tolerant_matrix: Tensor):
    ones = torch.ones_like(y_true).to(y_true).float()
    zeros = torch.zeros_like(y_true).to(y_true).float()
    mask_1 = torch.where((1 - torch.matmul(y_true, tolerant_matrix)) > 0, ones, zeros).to(y_true).float()
    mask_2 = torch.where((mask_1 + y_true.float()) > 0, ones, zeros).to(y_true).float()
    return mask_2


def multilabel_categorical_crossentropy_tolerant(y_pred, y_true, tolerant_matrix):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    tolerant_mask = get_masks(y_true, tolerant_matrix)
    y_pred_neg = y_pred_neg - (1-tolerant_mask) * 1e-12
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()


def validate_loss(output, target, weight=None, pos_weight=None):
    # 处理正负样本不均衡问题
    if pos_weight is None:
        label_size = output.size()[1]
        pos_weight = torch.ones(label_size)
    # 处理多标签不平衡问题
    if weight is None:
        label_size = output.size()[1]
        weight = torch.ones(label_size)

    val = 0
    for li_x, li_y in zip(output, target):
        for i, xy in enumerate(zip(li_x, li_y)):
            x, y = xy
            loss_val = pos_weight[i] * y * math.log(x, math.e) + (1 - y) * math.log(1 - x, math.e)
            val += weight[i] * loss_val
    return -val / (output.size()[0] * output.size(1))


def optimal_threshold(y_true, y_pred):
    """最优阈值的自动搜索
    """
    loss = lambda t: -np.mean((y_true > 0.5) == (np.tanh(y_pred) > np.tanh(t)))
    result = minimize(loss, 1, method='Powell')
    return result.x, -result.fun


def calc_loss(y_true, y_pred):
    # 1. 取出真实的标签
    y_true = y_true[::2]  # tensor([1, 0, 1]) 真实的标签

    # 2. 对输出的句子向量进行l2归一化   后面只需要对应为相乘  就可以得到cos值了
    norms = (y_pred ** 2).sum(axis=1, keepdims=True) ** 0.5
    # y_pred = y_pred / torch.clip(norms, 1e-8, torch.inf)
    y_pred = y_pred / norms

    # 3. 奇偶向量相乘
    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20

    # 4. 取出负例-正例的差值
    y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
    # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
    y_true = y_true[:, None] < y_true[None, :]  # 取出负例-正例的差值
    y_true = y_true.float()
    y_pred = y_pred - (1 - y_true) * 1e12
    y_pred = y_pred.view(-1)
    if torch.cuda.is_available():
        y_pred = torch.cat((torch.tensor([0]).float().cuda(), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    else:
        y_pred = torch.cat((torch.tensor([0]).float(), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1

    return torch.logsumexp(y_pred, dim=0)


def calc_loss_v2(y_true, y_pred):
    """
    :param y_true: [batch_size, ]
    :param y_pred: [batch_size, ]
    :return:
    """
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_pred *= 20
    y_true = y_true[:, None] < y_true[None, :]  # [batch_size, batch_size]
    y_true = y_true.float()
    y_pred = y_pred - (1 - y_true) * 1e12
    y_pred = y_pred.view(-1)
    if torch.cuda.is_available():
        y_pred = torch.cat((torch.tensor([0]).float().cuda(), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    else:
        y_pred = torch.cat((torch.tensor([0]).float(), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1

    return torch.logsumexp(y_pred, dim=0)


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


if __name__ == "__main__":
    # partitions = torch.tensor([1, 0, 1, 0, 1])
    # xx = partitions.view(-1)
    # print(partitions.shape)
    # print(xx.shape)
    # i = 0
    # data = torch.tensor([[1], [2], [3], [4], [5]])
    # print(partitions == i)
    # print([data[partitions == i]])
    #
    parts = [0, 1, 2, 1, 2]
    datas = [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]]
    parted_data = partition_list(data=datas, partitions=parts)
    print(parted_data)
    # law_relation_path = "/home/nxu/LEVENs/LEVEN-main/LCR_with_LawArticles/Match_data/LeCaRD/law_relation.pkl"
    # with open(law_relation_path, 'rb') as f:
    #     law_matrix: Tensor = pkl.load(f)
    #
    # label_true = torch.zeros([54,], dtype=torch.long)
    # label_true[0] = 1
    # print(label_true)
    # loss = multilabel_categorical_crossentropy_tolerant(torch.zeros_like(label_true), label_true, law_matrix)

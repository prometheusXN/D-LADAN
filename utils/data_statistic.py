import sys
sys.path.append('..')
import numpy as np
from data_preprocess.cail_reader import *
import matplotlib.pyplot as plt
from sklearn import metrics


def court_dataset_WithLawLabel(dataset, label_num):
    """
    :param dataset: the datalist, where list[0]：input; list[1]: label of law; list[2]: label of charges;
                    list[3]: label of times.
    :return: a dict of dataset, where the key is the law label index, the value is data set with the corresponding label.
    """
    law_num_dict = {i:0 for i in range(label_num)}
    data_dict = {i:[] for i in range(label_num)}
    for i in dataset:
        law_indexs = np.argmax(i[1], axis=-1)
        law_num_dict[law_indexs] += 1
        data_dict[law_indexs].append(i)

    law_nums = list(law_num_dict.values())
    law_indexes = list(law_num_dict.keys())
    law_name_index = np.lexsort((np.array(law_indexes), np.array(law_nums)))
    # np.argsort(np.array(law_nums))
    law_nums = np.sort(np.array(law_nums))
    law_statistic = [law_name_index, law_nums]

    return law_statistic


def generate_label_by_court(dataset, label_num, rate=0.6):
    label_index, sample_court = court_dataset_WithLawLabel(dataset, label_num=label_num)
    low_num = int(label_num * rate)
    high_num = label_num -low_num

    few_shot_label = [0 for i in range(low_num)]
    many_shot_label = [1 for i in range(high_num)]
    shot_label = few_shot_label + many_shot_label

    shot_index = list(np.argsort(np.array(label_index)))

    label = [shot_label[i] for i in shot_index]
    return np.array(label)


def get_statistic(dataset, allow_law=True, allow_accu=True, allow_time=True, law_num=0, accu_num=0, time_num=0):
    law_statistic = []
    accu_statistic = []
    time_statistic = []
    law_num_dict = {i:0 for i in range(law_num)}
    accu_num_dict = {i:0 for i in range(accu_num)}
    time_num_dict = {i: 0 for i in range(time_num)}

    for i in dataset:
        law_index = np.argmax(i[1], axis=-1)
        accu_index = np.argmax(i[2], axis=-1)
        time_index = np.argmax(i[3], axis=-1)
        law_num_dict[law_index] += 1
        accu_num_dict[accu_index] += 1
        time_num_dict[time_index] += 1
    if allow_law:
        law_nums = list(law_num_dict.values())
        law_indexes = list(law_num_dict.keys())
        # law_name_index = np.lexsort((np.array(law_indexes), np.array(law_nums)))
        law_name_index = np.argsort(np.array(law_nums))
        law_nums = np.sort(np.array(law_nums))
        law_statistic = [law_name_index, law_nums]
        # law_statistic = law_num_dict

    if allow_accu:
        accu_nums = list(accu_num_dict.values())
        accu_indexes = list(accu_num_dict.keys())
        # accu_name_index = np.lexsort((np.array(accu_indexes), np.array(accu_nums)))
        accu_name_index = np.argsort(np.array(accu_nums))
        accu_nums = np.sort(np.array(accu_nums))
        accu_statistic = [accu_name_index, accu_nums]
        # accu_statistic = accu_num_dict

    if allow_time:
        time_nums = list(time_num_dict.values())
        time_indexes = list(time_num_dict.keys())
        # time_name_index = np.lexsort((np.array(time_indexes), np.array(time_nums)))
        time_name_index = np.argsort(np.array(time_nums))
        time_nums = np.sort(np.array(time_nums))
        time_statistic = [time_name_index, time_nums]

    return law_statistic, accu_statistic, time_statistic


def drow_statistic(data_nums: np.ndarray):
    x = []
    y = []
    index_x = 0
    data_num = data_nums.shape[-1]
    for i in range(data_num-1, -1, -1):
        x.append(index_x)
        y.append(data_nums[i])
        index_x += 1
    return x, y


def get_acc_list(pred, gold, label_nums):
    '''
    :param pred:
    :param gold:
    :param label_nums:
    :return:
    '''

    gold_index = tf.convert_to_tensor(np.argmax(gold, axis=-1), dtype=tf.int32)

    pred_T = tf.convert_to_tensor(pred, dtype=tf.float32)
    gold_T = tf.convert_to_tensor(gold, dtype=tf.float32)
    pred_list = tf.dynamic_partition(pred_T, gold_index, num_partitions=label_nums)
    gold_list = tf.dynamic_partition(gold_T, gold_index, num_partitions=label_nums)

    acc_list = []
    for i in range(len(pred_list)):
        pred = pred_list[i].numpy()
        truth = gold_list[i].numpy()
        num = len(pred)
        if num == 0:
            acc_list.append(0.0)
        else:
            y_pred = np.argmax(pred, axis=1)
            y_true = np.argmax(truth, axis=1)
            accuracy_metric = metrics.accuracy_score(y_true, y_pred)
            acc_list.append(accuracy_metric)
    return acc_list


def drow_statistic_graph(label_numbers, label_indexes, pred, gold, name=None, color='red', markerfacecolor='blue'):

    label_num = len(label_indexes)
    acc_list = get_acc_list(pred, gold, label_num)
    print(acc_list)
    acc_list_sorted = []
    for i in range(len(label_indexes)-1, -1, -1):
        index = label_indexes[i]
        acc_list_sorted.append(acc_list[index])

    x_law, y_law = drow_statistic(label_numbers)

    plt.figure(figsize=(8, 8), dpi=300)
    fig, ax1 = plt.subplots()
    if "Law" in name:
        ax1.set_xticks([0, 25, 50, 75, 100])
        ax1.set_xlabel('Law articles IDs (after sorting)', color='black', fontsize=20)
        ax1.set_yticks([0, 1000, 2000, 3000])
        ax1.set_ylim(-0.1, 4000)
    if "Accu" in name:
        ax1.set_xticks([0, 25, 50, 75, 100])
        ax1.set_xlabel('Charges IDs (after sorting)', color='black', fontsize=20)
        ax1.set_yticks([0, 1000, 2000, 3000])
        ax1.set_ylim(-0.1, 4000)
    if "Time" in name:
        ax1.set_xlabel('Terms of Penalty (after sorting)', color='black', fontsize=20)
    ax1.set_ylabel('Frequency', color='black', fontsize=20)
    ax1.bar(x_law, y_law, edgecolor='black', width=1.0, align='center', alpha=0.5, linewidth=1.0, color='black')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=16)
    ax1.tick_params(axis='x', labelcolor=markerfacecolor, labelsize=16)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color=markerfacecolor, fontsize=20)
    ax2.set_ylim(-0.05, 1.05)
    plt.plot(x_law, acc_list_sorted, color=color, linestyle='--', marker='v', markerfacecolor='blue', markersize=6)
    ax2.tick_params(axis='y', labelcolor=markerfacecolor, labelsize=16)
    fig.tight_layout()
    plt.savefig(name+".pdf")
    plt.clf()

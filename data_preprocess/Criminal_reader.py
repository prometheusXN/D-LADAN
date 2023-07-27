import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pickle as pkl
from tensorflow.keras.utils import to_categorical
import sys


def data_split(dataset, accu_num):
    fact_description = dataset['fact_list']
    accu_labels_train = dataset['charge_label_list']
    accu_labels = [to_categorical(data, accu_num) for data in accu_labels_train]

    data_input = list(zip(fact_description, accu_labels))

    return data_input


def read_Criminal(dir_path, version='small'):
    '''
    :param dir_path: the file_path of dataset, e.g., '/home/nxu/Ladan_tnnls/processed_dataset/Criminal'
    :param data_format:
    :param version:
    :return:
    '''
    versions = ['small', 'medium', 'large']
    accu_num = 149
    for ver in versions:
        if ver == version:
            dataformat = '/Criminal_{}/'.format(ver)
            train_dir = dir_path + dataformat + 'train.pkl'
            valid_dir = dir_path + dataformat + 'valid.pkl'
            test_dir = dir_path + dataformat + 'test.pkl'

            f_train = pkl.load(open(train_dir, 'rb'))
            f_valid = pkl.load(open(valid_dir, 'rb'))
            f_test = pkl.load(open(test_dir, 'rb'))

            train_set = data_split(f_train, accu_num)
            valid_set = data_split(f_valid, accu_num)
            test_set = data_split(f_test, accu_num)

            yield train_set, valid_set, test_set

    print('Incorrect file name')


def get_dataset_Criminal(data, batch_size, shuffle=True, group_indexes=None, PPK=True):
    idxs = np.array(range(len(data)))
    if shuffle:
        np.random.shuffle(idxs)
    facts, accu_labels, group_labels = [], [], []
    for i in idxs:
        fact, accu_label = data[i]
        facts.append(np.array(fact))
        accu_labels.append(np.array(accu_label))

    if group_indexes is not None:
        group_num = np.max(group_indexes)
        accu_labels = np.array(accu_labels)
        group_labels = to_categorical(np.sum(accu_labels.astype(int) * group_indexes, axis=-1), num_classes=group_num+1)

        if PPK:
            train_dataset = tf.data.Dataset.from_tensor_slices(np.array(facts))
            label = tf.data.Dataset.from_tensor_slices({
                                                        'accu': accu_labels,
                                                        'group_prior': group_labels,
                                                        'group_posterior': accu_labels
                                                        })
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices(np.array(facts))
            label = tf.data.Dataset.from_tensor_slices({
                                                        'accu': accu_labels,
                                                        'group_prior': group_labels
                                                        })
        if shuffle:
            dataset = tf.data.Dataset.zip((train_dataset, label)).batch(batch_size, drop_remainder=True)
        else:
            dataset = tf.data.Dataset.zip((train_dataset, label)).batch(batch_size, drop_remainder=True)
        return dataset, accu_labels

    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(np.array(facts))
    label = tf.data.Dataset.from_tensor_slices(({
                                                 'accu': np.array(accu_labels),
                                                 }))
    dataset = tf.data.Dataset.zip((train_dataset, label)).batch(batch_size, drop_remainder=True)
    return dataset, np.array(accu_labels)
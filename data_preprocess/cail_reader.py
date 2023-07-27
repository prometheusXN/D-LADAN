import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pickle as pkl
from tensorflow.keras.utils import to_categorical
import sys


def data_split(dataset, law_num, accu_num, time_num):
    fact_description = dataset['fact_list']
    law_labels_train = dataset['law_label_lists']
    accu_labels_train = dataset['accu_label_lists']
    time_labels_train = dataset['term_lists']
    law_labels = [to_categorical(data, law_num) for data in law_labels_train]
    accu_labels = [to_categorical(data, accu_num) for data in accu_labels_train]
    time_labels = [to_categorical(data, time_num) for data in time_labels_train]

    data_input = list(zip(fact_description, law_labels, accu_labels, time_labels))

    return data_input


def read_cails(dir_path, data_format='legal_basis', version='small') -> list:
    '''
    :param dir_path: the file_path of dataset, e.g., '/home/nxu/Ladan_tnnls/processed_dataset/CAIL'
    :param data_format:
    :param version:
    :return:
    '''

    if data_format == 'legal_basis':
        dataformat = '/two_level/'
        if version == 'small':
            law_num = 103
            accu_num = 119
            time_num = 11

            train_dir = dir_path+dataformat+version + '/train_processed_thulac.pkl'
            valid_dir = dir_path+dataformat+version + '/valid_processed_thulac.pkl'
            test_dir = dir_path+dataformat+version + '/test_processed_thulac.pkl'

            f_train = pkl.load(open(train_dir, 'rb'))
            f_valid = pkl.load(open(valid_dir, 'rb'))
            f_test = pkl.load(open(test_dir, 'rb'))

            train_set = data_split(f_train, law_num, accu_num, time_num)
            valid_set = data_split(f_valid, law_num, accu_num, time_num)
            test_set = data_split(f_test, law_num, accu_num, time_num)

            yield train_set, valid_set, test_set

        else:
            version = 'large'
            law_num = 118
            accu_num = 130
            time_num = 11

            train_dir = dir_path + dataformat + version + '/train_processed_thulac.pkl'
            test_dir = dir_path + dataformat + version + '/test_processed_thulac.pkl'

            f_train = pkl.load(open(train_dir, 'rb'))
            f_test = pkl.load(open(test_dir, 'rb'))

            train_set = data_split(f_train, law_num, accu_num, time_num)
            test_set = data_split(f_test, law_num, accu_num, time_num)

            yield train_set, test_set, test_set

    else:
        dataformat = '/normal/'
        if version == 'small':
            law_num = 103
            accu_num = 119
            time_num = 11

            train_dir = dir_path+dataformat+version + '/train_processed_thulac.pkl'
            valid_dir = dir_path+dataformat+version + '/valid_processed_thulac.pkl'
            test_dir = dir_path+dataformat+version + '/test_processed_thulac.pkl'

            f_train = pkl.load(open(train_dir, 'rb'))
            f_valid = pkl.load(open(valid_dir, 'rb'))
            f_test = pkl.load(open(test_dir, 'rb'))

            train_set = data_split(f_train, law_num, accu_num, time_num)
            valid_set = data_split(f_valid, law_num, accu_num, time_num)
            test_set = data_split(f_test, law_num, accu_num, time_num)

            yield train_set, valid_set, test_set

        else:
            version = 'large'
            law_num = 118
            accu_num = 130
            time_num = 11

            train_dir = dir_path + dataformat + version + '/train_processed_thulac_large.pkl'
            test_dir = dir_path + dataformat + version + '/test_processed_thulac_large.pkl'

            f_train = pkl.load(open(train_dir, 'rb'))
            f_test = pkl.load(open(test_dir, 'rb'))

            train_set = data_split(f_train, law_num, accu_num, time_num)
            test_set = data_split(f_test, law_num, accu_num, time_num)

            yield train_set, test_set


def get_dataset(data, batch_size, shuffle=True, group_indexes=None, PPK=True, accu_relation=None, group_indexes_A=None):
    idxs = np.array(range(len(data)))
    if shuffle:
        np.random.shuffle(idxs)
    facts, law_labels, accu_labels, time_labels, group_labels = [], [], [], [], []
    triplet_flags = []
    for i in idxs:
        fact, law_label, accu_label, time_label = data[i]
        facts.append(np.array(fact))
        law_labels.append(np.array(law_label))
        accu_labels.append(np.array(accu_label))
        time_labels.append(np.array(time_label))
        triplet_flags.append(int(0))

    if group_indexes is not None:
        group_num = np.max(group_indexes)
        law_labels = np.array(law_labels)
        group_labels = to_categorical(np.sum(law_labels.astype(int) * group_indexes, axis=-1), num_classes=group_num+1)

        if PPK:
            if accu_relation is None:
                train_dataset = tf.data.Dataset.from_tensor_slices(np.array(facts))
                label = tf.data.Dataset.from_tensor_slices({'law': law_labels,
                                                            'accu': np.array(accu_labels),
                                                            'time': np.array(time_labels),
                                                            'group_prior': group_labels,
                                                            'group_posterior': law_labels
                                                            })
            elif group_indexes_A is None:
                print('add_accu_label')
                train_dataset = tf.data.Dataset.from_tensor_slices(np.array(facts))
                label = tf.data.Dataset.from_tensor_slices({'law': law_labels,
                                                            'accu': np.array(accu_labels),
                                                            'time': np.array(time_labels),
                                                            'group_prior': group_labels,
                                                            'group_posterior': law_labels,
                                                            'group_posterior_accu': accu_labels,
                                                            })

            else:
                group_num_A = np.max(group_indexes_A)
                accu_labels = np.array(accu_labels)
                group_labels_A = to_categorical(np.sum(accu_labels.astype(int) * group_indexes_A, axis=-1),
                                                num_classes=group_num_A + 1)
                print('add_accu_prior_posterior')
                train_dataset = tf.data.Dataset.from_tensor_slices(np.array(facts))
                label = tf.data.Dataset.from_tensor_slices({'law': law_labels,
                                                            'accu': np.array(accu_labels),
                                                            'time': np.array(time_labels),
                                                            'group_prior': group_labels,
                                                            'group_posterior': law_labels,
                                                            'group_prior_accu': group_labels_A,
                                                            'group_posterior_accu': accu_labels,
                                                            })
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices(np.array(facts))
            label = tf.data.Dataset.from_tensor_slices({'law': law_labels,
                                                        'accu': np.array(accu_labels),
                                                        'time': np.array(time_labels),
                                                        'group_prior': group_labels
                                                        })
        if shuffle:
            dataset = tf.data.Dataset.zip((train_dataset, label)).batch(batch_size, drop_remainder=True)
        else:
            dataset = tf.data.Dataset.zip((train_dataset, label)).batch(batch_size, drop_remainder=True)
        return dataset, np.array(law_labels), np.array(accu_labels), np.array(time_labels)

    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(np.array(facts))
    label = tf.data.Dataset.from_tensor_slices(({'law': np.array(law_labels),
                                                 'accu': np.array(accu_labels),
                                                 'time': np.array(time_labels)
                                                 }))
    dataset = tf.data.Dataset.zip((train_dataset, label)).batch(batch_size, drop_remainder=True)
    return dataset, np.array(law_labels), np.array(accu_labels), np.array(time_labels)


def get_dataset_multi(data, batch_size, group_indexes, classifier_indexes, shuffle=True):
    idxs = np.array(range(len(data)))
    if shuffle:
        np.random.shuffle(idxs)
    facts, law_labels, accu_labels, time_labels = [], [], [], []
    triplet_flags = []
    for i in idxs:
        fact, law_label, accu_label, time_label = data[i]
        facts.append(np.array(fact))
        law_labels.append(np.array(law_label))
        accu_labels.append(np.array(accu_label))
        time_labels.append(np.array(time_label))
        triplet_flags.append(int(0))

    group_num = np.max(group_indexes)
    law_labels = np.array(law_labels)
    group_labels = to_categorical(np.sum(law_labels.astype(int) * group_indexes, axis=-1), num_classes=group_num + 1)

    classifier_num = np.max(classifier_indexes)
    classifier_labels = to_categorical(np.sum(law_labels.astype(int) * classifier_indexes, axis=-1), num_classes=classifier_num + 1)

    train_dataset = tf.data.Dataset.from_tensor_slices(np.array(facts))
    label = tf.data.Dataset.from_tensor_slices({'law': law_labels,
                                                'accu': np.array(accu_labels),
                                                'time': np.array(time_labels),
                                                'group': group_labels,
                                                'classifier': classifier_labels
                                                })
    if shuffle:
        dataset = tf.data.Dataset.zip((train_dataset, label)).shuffle(1280).batch(batch_size, drop_remainder=True)
    else:
        dataset = tf.data.Dataset.zip((train_dataset, label)).batch(batch_size, drop_remainder=True)
    return dataset, np.array(law_labels), np.array(accu_labels), np.array(time_labels)

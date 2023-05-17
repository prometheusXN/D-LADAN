import sys
sys.path.append('..')
import os
import tensorflow.keras.backend as K
import gc
import pickle as pk
from parser import ConfigParser
from Model.LADAN_model import ladan_model
from data_preprocess.cail_reader import *
from tensorflow.keras.callbacks import *
from utils.evalution_component import evaluation_multitask, filter_samples, eval_data_types
from utils.dataset_padding import padding_dataset
from utils.training_setup import setup_seed
from law_processed.law_processed import get_law_graph_adj
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from utils.data_statistic import *


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_config', '-dc', default=)
    # parser.add_argument('--data_config', '-dc', default=)
    # parser.add_argument('--gpu', '-g', default='0')

    #-------------------------------set_configs--------------------------#
    setup_seed(666)
    np.random.seed(666)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    os.environ['TF_KERAS'] = '1'
    os.environ['TF_EAGER'] = '1'
    gpus = tf.config.list_physical_devices('GPU')

    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        print('Invalid device or cannot modify virtual devices once initialized.')

    configFilePath = '../Config/Ladan.config'
    config = ConfigParser(configFilePath)

    batch_size = config.getint('data', 'batch_size')
    max_epoch = config.getint('train', 'epoch')
    learning_rate = config.getfloat('train', 'learning_rate')
    embedding_dim = config.getint('data', 'vec_size')
    law_num = config.getint('num_class_small', 'law_num')
    accu_num = config.getint('num_class_small', 'accu_num')
    time_num = config.getint('num_class_small', 'time_num')
    more_fc = config.getboolean("net", "more_fc")
    law_relation_threshold = config.getfloat('data', 'graph_threshold')

    with open('../data/w2id_thulac_new.pkl', 'rb') as f:
        word2id_dict = pk.load(f)
        f.close()

    word_dict_len = len(word2id_dict)
    emb_path = '../data/cail_thulac_new.npy'

    #------------------------------data_pre-prosess------------------------#
    data_path = '/home/nxu/Ladan_tnnls/processed_dataset/CAIL_new'
    dataset_fold = read_cails(dir_path=data_path,
                              data_format='legal_basis',
                              version='small')
    law_input, group_list, graph_membership, law_adj_matrix = get_law_graph_adj(law_relation_threshold,
                                                                                '../data/w2id_thulac_new.pkl', 15, 100)
    group_num = len(group_list)
    print(group_num)
    group_indexes = list(zip(*graph_membership))[1]

    train_set, valid_set, test_set = next(dataset_fold)
    train_set, _, _ = padding_dataset(train_set, batch_size)
    valid_set, _, _ = padding_dataset(valid_set, batch_size)
    train_D, _, _, _ = get_dataset(data=train_set, batch_size=batch_size, shuffle=True, group_indexes=group_indexes, PPK=False)
    valid_D, _, _, _ = get_dataset(data=valid_set, batch_size=batch_size, shuffle=False, group_indexes=group_indexes, PPK=False)

    #------------------------------get_model-------------------------------#

    Ladan_model = ladan_model(config=config, emb_path=emb_path, word2id_dict=word2id_dict, group_indexes=group_indexes,
                              group_num=group_num, law_input=law_input, law_adj_matrix=law_adj_matrix, trainable=True)

    Ladan_model.compile(optimizer=Adam(learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-7),
                        loss={'law': "categorical_crossentropy",
                              'accu': "categorical_crossentropy",
                              'time': "categorical_crossentropy",
                              'group_prior': "categorical_crossentropy"
                              },
                        metrics={'law': "accuracy",
                                 'accu': "accuracy",
                                 'time': "accuracy",
                                 'group_prior': "accuracy"
                                 },
                        loss_weights={'law': 1.0,
                                      'accu': 1.0,
                                      'time': 1.0,
                                      'group_prior': 0.1
                                      },
                        run_eagerly=False)

    early_stopping = EarlyStopping(monitor='val_ave_accuracy', patience=5)  # 早停法，防止过拟合
    plateau = ReduceLROnPlateau(monitor="val_ave_accuracy", verbose=1, mode='max', factor=0.5,
                                patience=3)  # 当评价指标不在提升时，减少学习率
    checkpoint = ModelCheckpoint(
        '../model_save/Ladan/Ladan_small.hdf5',
        monitor='val_ave_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=True)

    Ladan_model.fit_(
        train_D,
        # epochs=1,
        epochs=max_epoch,
        validation_data=valid_D,
        callbacks=[
            # early_stopping,
            plateau,
            checkpoint],
        # callbacks=[checkpoint]
    )

    del Ladan_model
    K.clear_session()

    #------------------------------------------now_testing-----------------------------------#
    sample_set, sample_num, step = padding_dataset(test_set[:batch_size], batch_size)
    test_set, sample_num_test, step_test = padding_dataset(test_set, batch_size)
    sample_D, _, _, _ = get_dataset(data=sample_set, batch_size=batch_size,
                                    shuffle=False, group_indexes=group_indexes, PPK=False)
    test_D, law_labels, accu_labels, time_labels = get_dataset(data=test_set, batch_size=batch_size,
                                                               shuffle=False, group_indexes=group_indexes, PPK=False)

    Ladan_model = ladan_model(config=config, emb_path=emb_path, word2id_dict=word2id_dict, group_indexes=group_indexes,
                            group_num=group_num, law_input=law_input, law_adj_matrix=law_adj_matrix, trainable=False)

    Ladan_model.compile(optimizer=Adam(learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-7),
                        loss={'law': "categorical_crossentropy",
                              'accu': "categorical_crossentropy",
                              'time': "categorical_crossentropy",
                              'group_prior': "categorical_crossentropy"
                              },
                        metrics={'law': "accuracy",
                                 'accu': "accuracy",
                                 'time': "accuracy",
                                 'group_prior': "accuracy"
                                 },
                        loss_weights={'law': 1.0,
                                      'accu': 1.0,
                                      'time': 1.0,
                                      'group_prior': 0.1
                                      },
                        run_eagerly=False)
    pred_sample = Ladan_model.predict(sample_D, steps=step, verbose=1)
    Ladan_model.load_weights(filepath="../model_save/Ladan/Ladan_small.hdf5", by_name=True)
    print('now_predicting')
    predictions = Ladan_model.predict(test_D, steps=step_test, verbose=1)
    print(sample_num_test)
    pred_law = predictions['law'][:sample_num_test, :]
    pred_accu = predictions['accu'][:sample_num_test, :]
    pred_time = predictions['time'][:sample_num_test, :]
    prediction = [pred_law, pred_accu, pred_time]

    gold_law = law_labels[:sample_num_test, :]
    gold_accu = accu_labels[:sample_num_test, :]
    gold_time = time_labels[:sample_num_test, :]
    y = [gold_law, gold_accu, gold_time]
    metric = evaluation_multitask(y, prediction, 3)
    task = ['law', 'accu', 'time']
    print('Now_testing')
    for i in range(3):
        print('Metrics for {} prediction is: '.format(task[i]), metric[i])

    print('\n')

    # law_statisitc, accu_statistic, time_statistic = get_statistic(train_set, law_num=law_num, accu_num=accu_num,
    #                                                               time_num=time_num)
    #
    # accu_file = open('../experiment_results/Ladan/accu_results_small', 'wb')
    # law_file = open('../experiment_results/Ladan/law_results_small', 'wb')
    # pk.dump([pred_accu,gold_accu, accu_statistic], accu_file)
    # pk.dump([pred_law, gold_law, law_statisitc], law_file)
    # accu_index, accu_numbers = accu_statistic
    # law_index, law_numbers = law_statisitc
    # time_index, time_numbers = time_statistic
    #
    # drow_statistic_graph(label_numbers=law_numbers, label_indexes=law_index, pred=pred_law, gold=gold_law,
    #                      name="Law_Ladan", color='cornflowerblue', markerfacecolor='royalblue')
    # drow_statistic_graph(label_numbers=accu_numbers, label_indexes=accu_index, pred=pred_accu, gold=gold_accu,
    #                      name="Accu_Ladan", color='cornflowerblue', markerfacecolor='royalblue')
    # drow_statistic_graph(label_numbers=time_numbers, label_indexes=time_index, pred=pred_time, gold=gold_time,
    #                      name="Time_Ladan", color='cornflowerblue', markerfacecolor='royalblue')

    tail_law_list = [81, 7, 43, 17, 14, 48, 76, 96, 15, 72, 70, 54, 69, 49, 53, 85, 56, 44]
    tail_charge_list = [114, 99, 86, 106, 36, 71, 104, 72, 95, 34, 39, 51, 23, 109, 62, 54, 31, 59, 60, 81, 110, 76, 26,
                        19, 64]

    law_preds = np.argmax(pred_law, axis=1)
    law_labels = np.argmax(gold_law, axis=1)

    accu_preds = np.argmax(pred_accu, axis=1)
    accu_labels = np.argmax(gold_accu, axis=1)

    law_labels, law_preds = filter_samples(law_preds, law_labels, tail_law_list)
    accu_labels, accu_preds = filter_samples(accu_preds, accu_labels, tail_charge_list)

    print('Law article:')
    eval_data_types(law_labels, law_preds, num_labels=103, label_list=None)

    print('Charge:')
    eval_data_types(accu_labels, accu_preds, num_labels=119, label_list=None)

    del Ladan_model
    K.clear_session()
    gc.collect()  # 清理内存

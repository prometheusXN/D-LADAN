import tensorflow as tf
import random
import os
import warnings
import tensorflow.keras.backend as K
import gc
import pickle as pk
from parser import ConfigParser
from Model.HAN_model import han_model
from data_preprocess.cail_reader import *
from tensorflow.keras.callbacks import *
from utils.evalution_component import evaluation_multitask
from utils.dataset_padding import padding_dataset
from utils.training_setup import setup_seed
from utils.data_statistic import *


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_config', '-dc', default=)
    # parser.add_argument('--data_config', '-dc', default=)
    # parser.add_argument('--gpu', '-g', default='0')

    #-------------------------------set_configs--------------------------#
    setup_seed(666)
    np.random.seed(666)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ['TF_KERAS'] = '1'
    os.environ['TF_EAGER'] = '1'
    gpus = tf.config.list_physical_devices('GPU')

    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        print('Invalid device or cannot modify virtual devices once initialized.')

    configFilePath = '../Config/Han.config'
    config = ConfigParser(configFilePath)

    batch_size = config.getint('data', 'batch_size')
    max_epoch = config.getint('train', 'epoch')
    learning_rate = config.getfloat('train', 'learning_rate')
    embedding_dim = config.getint('data', 'vec_size')
    law_num = config.getint('num_class_small', 'law_num')
    accu_num = config.getint('num_class_small', 'accu_num')
    time_num = config.getint('num_class_small', 'time_num')
    more_fc = config.getboolean("net", "more_fc")

    with open('../data/w2id_thulac.pkl', 'rb') as f:
        word2id_dict = pk.load(f)
        f.close()

    word_dict_len = len(word2id_dict)
    emb_path = '../data/cail_thulac.npy'

    #------------------------------data_pre-prosess------------------------#
    data_path = '/home/nxu/Ladan_tnnls/processed_dataset/CAIL'
    dataset_fold = read_cails(dir_path=data_path,
                              data_format='legal_basis',
                              version='small')
    train_set, valid_set, test_set = next(dataset_fold)
    train_set, _, _ = padding_dataset(train_set, batch_size)
    valid_set, _, _ = padding_dataset(valid_set, batch_size)
    train_D, _, _, _ = get_dataset(data=train_set, batch_size=batch_size, shuffle=True)
    valid_D, _, _, _ = get_dataset(data=valid_set, batch_size=batch_size, shuffle=False)

    #------------------------------get_model-------------------------------#

    # Han_model = han_model(config=config,
    #                       word_dict_size=word_dict_len,
    #                       embedding_dim=embedding_dim,
    #                       law_num=law_num,
    #                       accu_num=accu_num,
    #                       time_num=time_num,
    #                       emb_path=emb_path,
    #                       word2id_dict=word2id_dict)
    #
    # early_stopping = EarlyStopping(monitor='val_output_accu_accuracy', patience=5)  # 早停法，防止过拟合
    # plateau = ReduceLROnPlateau(monitor="val_output_accu_accuracy", verbose=1, mode='max', factor=0.5,
    #                             patience=3)  # 当评价指标不在提升时，减少学习率
    # checkpoint = ModelCheckpoint(
    #     '../model_save/Han/Han_small.hdf5',
    #     monitor='val_output_accu_accuracy', verbose=2, save_best_only=True, mode='max', save_weights_only=True)
    #
    # Han_model.fit(
    #     train_D,
    #     # epochs=1,
    #     epochs=max_epoch,
    #     validation_data=valid_D,
    #     callbacks=[early_stopping, plateau, checkpoint],
    # )
    #
    # del Han_model
    # K.clear_session()

    #------------------------------------------now_testing-----------------------------------#

    test_set, sample_num_test, step_test = padding_dataset(test_set, batch_size)
    test_D, law_labels, accu_labels, time_labels = get_dataset(data=test_set, batch_size=batch_size, shuffle=False)

    Han_model = han_model(config=config,
                          word_dict_size=word_dict_len,
                          embedding_dim=embedding_dim,
                          law_num=law_num,
                          accu_num=accu_num,
                          time_num=time_num,
                          emb_path=emb_path,
                          word2id_dict=word2id_dict)

    Han_model.load_weights(filepath="../model_save/Han/Han_small.hdf5", by_name=True)
    print('now_predicting')
    pred_law, perd_accu, pred_time = Han_model.predict(test_D, steps=step_test, verbose=1)
    print(pred_law.shape)
    pred_law = pred_law[:sample_num_test, :]
    pred_accu = perd_accu[:sample_num_test, :]
    pred_time = pred_time[:sample_num_test, :]
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

    law_statisitc, accu_statistic, time_statistic = get_statistic(train_set, law_num=law_num, accu_num=accu_num,
                                                                  time_num=time_num)

    accu_index, accu_numbers = accu_statistic
    law_index, law_numbers = law_statisitc
    time_index, time_numbers = time_statistic

    drow_statistic_graph(label_numbers=law_numbers, label_indexes=law_index, pred=pred_law, gold=gold_law,
                         name="Law_Han", color='blue')
    drow_statistic_graph(label_numbers=accu_numbers, label_indexes=accu_index, pred=pred_accu, gold=gold_accu,
                         name="Accu_Han", color='blue')
    drow_statistic_graph(label_numbers=time_numbers, label_indexes=time_index, pred=pred_time, gold=gold_time,
                         name="Time_Han", color='blue')
    del Han_model
    gc.collect()  # 清理内存
    K.clear_session()
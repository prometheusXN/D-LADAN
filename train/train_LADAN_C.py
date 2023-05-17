import sys
sys.path.append('..')
import argparse
import tensorflow.keras.backend as K
import gc
from Config.parser import ConfigParser
from Model.LADAN_model_C import Ladan_criminal
from data_preprocess.Criminal_reader import *
from tensorflow.keras.callbacks import *
from utils.evalution_component import evaluation_multitask
from utils.dataset_padding import padding_dataset
from utils.training_setup import setup_seed
from tensorflow.keras.optimizers import Adam
from Criminal_processes.law_process import *
from tensorflow.python.keras.engine.training import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=str, default='0')
    parser.add_argument('--embedding_trainable', '-et', type=bool, default=False)
    parser.add_argument('--data_version', '-dv', type=str, default='small')

    args = parser.parse_args()
    data_version = args.data_version
    model_path = "../model_save/Ladan/Ladan_Criminal_{}.hdf5".format(data_version)

    #-------------------------------set_configs--------------------------#
    setup_seed(666)
    np.random.seed(666)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['TF_KERAS'] = '1'
    os.environ['TF_EAGER'] = '0'
    gpus = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        print('Invalid device or cannot modify virtual devices once initialized.')

    configFilePath = '../Config/LadanPPK_Criminal.config'
    config = ConfigParser(configFilePath)

    batch_size = config.getint('data', 'batch_size')
    max_epoch = config.getint('train', 'epoch')
    learning_rate = config.getfloat('train', 'learning_rate')
    embedding_dim = config.getint('data', 'vec_size')
    accu_num = config.getint('num_class', 'accu_num')
    law_relation_threshold = config.getfloat('data', 'graph_threshold')

    with open('../Criminal_Dataset/word2id_Criminal', 'rb') as f:
        word2id_dict = pk.load(f)
        f.close()

    word_dict_len = len(word2id_dict)
    emb_path = '../Criminal_Dataset/WordEmbedding_Criminal.npy'

    # ------------------------------data_pre-prosess------------------------#
    data_path = '/home/nxu/Ladan_tnnls/processed_dataset/Criminal'
    dataset_fold = read_Criminal(dir_path=data_path, version=data_version)

    law_input, group_list, graph_membership_law, graph_membership_charge, law_adj_matrix = \
        get_law_graph_for_Criminal(law_relation_threshold, word2id_path='../Criminal_Dataset/word2id_Criminal')
    group_num = len(group_list)
    print(group_num)
    group_indexes_charge = list(zip(*graph_membership_charge))[1]
    group_indexes_law = list(zip(*graph_membership_law))[1]

    train_set, valid_set, test_set = next(dataset_fold)

    train_set, _, _ = padding_dataset(train_set, batch_size)
    valid_set, _, _ = padding_dataset(valid_set, batch_size)

    train_D, _ = get_dataset_Criminal(data=train_set, batch_size=batch_size, shuffle=True,
                                            group_indexes=group_indexes_charge, PPK=False)
    valid_D, _ = get_dataset_Criminal(data=valid_set, batch_size=batch_size, shuffle=False,
                                            group_indexes=group_indexes_charge, PPK=False)

    # ------------------------------get_model-------------------------------#

    Ladan_model = Ladan_criminal(config=config, word2id_dict=word2id_dict, emb_path=emb_path, group_num=group_num,
                                    law_input=law_input, law_adj_matrix=law_adj_matrix, group_indexes=group_indexes_law,
                                    trainable=True, accu_num=accu_num, embedding_trainable=args.embedding_trainable)

    Ladan_model.compile(optimizer=Adam(learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-7),
                        loss={
                              'accu': "categorical_crossentropy",
                              'group_prior': "categorical_crossentropy"
                              },
                        metrics={
                                 'accu': "accuracy",
                                 'group_prior': "accuracy"
                                 },
                        loss_weights={
                                      'accu': 1.0,
                                      'group_prior': 0.1
                                      },
                        run_eagerly=False)

    early_stopping = EarlyStopping(monitor='val_accu_accuracy', patience=5)  # 早停法，防止过拟合
    plateau = ReduceLROnPlateau(monitor="val_accu_accuracy", verbose=1, mode='max', factor=0.5, patience=3,
                                min_delta=1e-6)  # 当评价指标不在提升时，减少学习率
    checkpoint = ModelCheckpoint(model_path, monitor='val_accu_accuracy', verbose=1,
                                 save_best_only=True, mode='max', save_weights_only=True)

    Ladan_model.fit_warmingup(
        train_D,
        epochs=max_epoch,
        ave_acc= False,
        # epochs=1,
        # data_generator.__iter__(),
        # epochs=data_generator.get_epoch(),
        # steps_per_epoch=len(data_generator),
        validation_data=valid_D,
        callbacks=[
            # early_stopping,
            plateau,
            checkpoint
        ],
        # triplet_data=data_generator
    )

    del Ladan_model
    K.clear_session()

    # ------------------------------------------now_testing-----------------------------------#
    sample_set, sample_num, step = padding_dataset(test_set[:batch_size], batch_size)
    sample_D, _ = get_dataset_Criminal(data=sample_set, batch_size=batch_size, shuffle=False,
                                       group_indexes=group_indexes_charge, PPK=False)
    test_set, sample_num_test, step_test = padding_dataset(test_set, batch_size)
    test_D, accu_labels = get_dataset_Criminal(data=test_set, batch_size=batch_size, shuffle=False,
                                               group_indexes=group_indexes_charge, PPK=False)

    Ladan_model = Ladan_criminal(config=config, word2id_dict=word2id_dict, emb_path=emb_path, group_num=group_num,
                                    law_input=law_input, law_adj_matrix=law_adj_matrix, group_indexes=group_indexes_law,
                                    trainable=True, accu_num=accu_num, embedding_trainable=args.embedding_trainable)

    Ladan_model.compile(optimizer=Adam(learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-7),
                        loss={
                              'accu': "categorical_crossentropy",
                              'group_prior': "categorical_crossentropy"
                              },
                        metrics={
                                 'accu': "accuracy",
                                 'group_prior': "accuracy"
                                 },
                        loss_weights={
                                      'accu': 1.0,
                                      'group_prior': 0.1
                                      },
                        run_eagerly=False)
    # Ladan_model.summary()
    pred_sample = Ladan_model.predict(sample_D, steps=step, verbose=1)
    Ladan_model.load_weights(filepath=model_path, by_name=True)
    print('now_predicting')
    predictions = Ladan_model.predict(test_D, steps=step_test, verbose=1)

    pred_accu = predictions['accu'][:sample_num_test, :]
    gold_accu = accu_labels[:sample_num_test, :]

    metric = evaluation_multitask([gold_accu], [pred_accu], 1)

    print('Metrics for charge prediction is: ', metric[0])

    del Ladan_model
    K.clear_session()
    gc.collect()  # 清理内存
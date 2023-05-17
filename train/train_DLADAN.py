import sys
sys.path.append("..")
import argparse
import tensorflow.keras.backend as K
import gc
import pickle as pk
from Config.parser import ConfigParser
from Model.DLADAN_model import DLADAN
from data_preprocess.cail_reader import *
from tensorflow.keras.callbacks import *
from utils.evalution_component import evaluation_multitask, evaluation_label_list, filter_samples, eval_data_types
from utils.dataset_padding import padding_dataset
from utils.training_setup import setup_seed
from tensorflow.keras.optimizers import Adam
from law_processed.law_processed import get_law_graph_adj, get_law_graph_large_adj
from tensorflow.python.keras.engine.training import *
import numpy as np


class UpdatePosteriorGraphMomentum(Callback):
    def __init__(self, DataSampling, step_warmup=50, momentum_steps=1):
        self.momentum_steps = momentum_steps
        self.num_passed_batchs = 0
        self.step_warmup = step_warmup
        self.DataSampling = DataSampling
        super(UpdatePosteriorGraphMomentum, self).__init__()

    def weight_update(self):
        if self.num_passed_batchs < self.step_warmup:
            pass

        elif self.num_passed_batchs == self.step_warmup:
            self.model.set_synchronize_memory(synchronize_memory=True)
            self.model.set_warming_up(warming_up=False)

        elif (self.num_passed_batchs % self.momentum_steps) == 0:
            if self.model.synchronize_memory:
                self.model.set_synchronize_memory(synchronize_memory=False)
            if not self.model.momentum_flag:
                self.model.set_momentum_flag(momentum_flag=True)
            else:
                pass
        else:
            if self.model.momentum_flag:
                self.model.set_momentum_flag(momentum_flag=False)
            else:
                pass

    def on_train_batch_end(self, batch, logs=None):
        self.num_passed_batchs += 1

    def on_train_batch_begin(self, batch, logs=None):
        self.weight_update()

    def on_train_begin(self, logs=None):
        if self.num_passed_batchs == 0:
            self.model.set_synchronize_memory(synchronize_memory=False)
            self.model.set_warming_up(warming_up=True)
            self.model.set_momentum_flag(momentum_flag=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=str, default='6',
                        help='Select which GPU device to use')
    parser.add_argument('--keep_coefficient', '-k', type=float, default=0.9,
                        help='Set the keep_coefficient when the memory units are momentum updated.')
    parser.add_argument('--warm_steps', '-w', type=int, default=1,
                        help='Set the number of the warming_up epochs before momentum updating.')
    parser.add_argument('--momentum_steps', '-ms', type=int, default=1,
                        help='Set how much training steps between two momentum updating.')
    parser.add_argument('--data_version', '-dv', type=str, default='small', choices=['small', 'large'],
                        help='Set which dataset is used.')
    parser.add_argument('--accu_relation', '-ar', type=bool, default=True,
                        help='Set whether use the memory units of the charge prediction subtask.')
    parser.add_argument('--Decoder_mode', '-Dm', type=str, default='MPBFN', choices=['MTL', 'TOPJUDGE', 'MPBFN'],
                        help='Set which multitask decoder is used.')
    parser.add_argument('--embedding_trainable', '-et', type=bool, default=False,
                        help='Whether the word embedding is trainable')
    parser.add_argument('--version_index', '-vi', type=int, default=20230302,
                        help='The index for saving different model.')
    parser.add_argument('--run_version', '-rv', type=str, default='train', choices=['train', 'test'],
                        help='The index for saving different model.')

    args = parser.parse_args()
    keep_coefficient = args.keep_coefficient
    warm_steps = args.warm_steps
    momentum_steps = args.momentum_steps
    data_version = args.data_version
    accu_relation = args.accu_relation
    decoder_mode = args.Decoder_mode
    embedding_trainable = args.embedding_trainable
    version_index = args.version_index
    run_version = args.run_version

    model_path = '../model_save/LadanPPK/DLADAN_{}'.format(data_version) + '_{}'.format(decoder_mode) \
                 + '_w{}'.format(warm_steps) + '_ms{}'.format(momentum_steps)\
                 + '_et{}'.format(embedding_trainable) + '_{}.hdf5'.format(str(version_index))
    # model_path = '../model_save/LadanPPK/DLADAN_MPBFN.hdf5'
    print(model_path)
    # -------------------------------set_configs--------------------------#
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

    configFilePath = '../Config/LadanPPK.config'
    config = ConfigParser(configFilePath)

    batch_size = config.getint('data', 'batch_size')
    max_epoch = config.getint('train', 'epoch')
    learning_rate = config.getfloat('train', 'learning_rate')
    embedding_dim = config.getint('data', 'vec_size')
    law_num = config.getint('num_class_{}'.format(data_version), 'law_num')
    accu_num = config.getint('num_class_{}'.format(data_version), 'accu_num')
    time_num = config.getint('num_class_{}'.format(data_version), 'time_num')
    more_fc = config.getboolean("net", "more_fc")
    law_relation_threshold = config.getfloat('data', 'graph_threshold')

    with open('../data/w2id_thulac_new.pkl', 'rb') as f:
        word2id_dict = pk.load(f)
        f.close()

    word_dict_len = len(word2id_dict)
    emb_path = '../data/cail_thulac_new.npy'

    # ------------------------------data_pre-prosess------------------------#
    data_path = '/home/nxu/Ladan_tnnls/processed_dataset/CAIL_new'
    dataset_fold = read_cails(dir_path=data_path, data_format='legal_basis', version=data_version)

    if data_version == 'small':
        law_input, group_list, graph_membership, law_adj_matrix = get_law_graph_adj(law_relation_threshold,
                                                                                    '../data/w2id_thulac_new.pkl'
                                                                                    , 15, 100)
    else:
        law_input, group_list, graph_membership, law_adj_matrix = get_law_graph_large_adj(law_relation_threshold,
                                                                                          '../data/w2id_thulac_new.pkl',
                                                                                          15, 100)

    group_num = len(group_list)
    print(law_relation_threshold)
    group_indexes = list(zip(*graph_membership))[1]

    train_set, valid_set, test_set = next(dataset_fold)

    train_set, _, steps = padding_dataset(train_set, batch_size)
    valid_set, _, _ = padding_dataset(valid_set, batch_size)

    train_D, _, _, _ = get_dataset(data=train_set, batch_size=batch_size, shuffle=True, group_indexes=group_indexes,
                                   PPK=True, accu_relation=accu_relation)
    valid_D, _, _, _ = get_dataset(data=valid_set, batch_size=batch_size, shuffle=True, group_indexes=group_indexes,
                                   PPK=True, accu_relation=accu_relation)

    sample_set, sample_num, step = padding_dataset(test_set[:batch_size], batch_size)
    sample_D, _, _, _ = get_dataset(data=sample_set, batch_size=batch_size,
                                    shuffle=False, group_indexes=group_indexes, PPK=True,
                                    accu_relation=accu_relation)
    test_set, sample_num_test, step_test = padding_dataset(test_set, batch_size)
    test_D, law_labels, accu_labels, time_labels = get_dataset(data=test_set, batch_size=batch_size,
                                                               shuffle=False, group_indexes=group_indexes, PPK=True,
                                                               accu_relation=accu_relation)

    # ------------------------------get_model-------------------------------#
    if run_version == "train":
        DLadan_model = DLADAN(config=config, emb_path=emb_path, word2id_dict=word2id_dict, group_indexes=group_indexes,
                              law_num=law_num, accu_num=accu_num, time_num=time_num,
                              group_num=group_num, law_input=law_input, law_adj_matrix=law_adj_matrix, trainable=True,
                              accu_relation=accu_relation, decoder_mode=decoder_mode, keep_coefficient=keep_coefficient,
                              embedding_trainable=embedding_trainable)

        DLadan_model.compile(optimizer=Adam(learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-7),
                             loss={'law': "categorical_crossentropy",
                                   'accu': "categorical_crossentropy",
                                   'time': "categorical_crossentropy",
                                   'group_prior': "categorical_crossentropy",
                                   'group_posterior': "categorical_crossentropy",
                                   'group_posterior_accu': "categorical_crossentropy"
                                   },
                             metrics={'law': "accuracy",
                                      'accu': "accuracy",
                                      'time': "accuracy",
                                      'group_prior': "accuracy",
                                      'group_posterior': "accuracy",
                                      'group_posterior_accu': "accuracy"
                                      },
                             loss_weights={'law': 1.0,
                                           'accu': 1.0,
                                           'time': 1.0,
                                           'group_prior': 0.1,
                                           'group_posterior': 0.1,
                                           'group_posterior_accu': 0.1
                                           },
                             run_eagerly=False)

        early_stopping = EarlyStopping(monitor='val_ave_accuracy', patience=5)  # 早停法，防止过拟合
        plateau = ReduceLROnPlateau(monitor="val_ave_accuracy", verbose=1, mode='max', factor=0.5, patience=4,
                                    min_delta=1e-4)  # 当评价指标不在提升时，减少学习率
        checkpoint = ModelCheckpoint(model_path, monitor='val_ave_accuracy', verbose=1,
                                     save_best_only=True, mode='max', save_weights_only=True)
        update_graph = UpdatePosteriorGraphMomentum(DataSampling=None, step_warmup=warm_steps*steps,
                                                    momentum_steps=momentum_steps)

        DLadan_model.fit_warmingup(
            train_D,
            epochs=max_epoch,
            validation_data=valid_D,
            # validation_data=test_D,
            callbacks=[
                # early_stopping,
                plateau,
                checkpoint,
                update_graph
            ],
        )

        del DLadan_model
        K.clear_session()

    # ------------------------------------------now_testing-----------------------------------#
    else:
        DLadan_model = DLADAN(config=config, emb_path=emb_path, word2id_dict=word2id_dict, group_indexes=group_indexes,
                              law_num=law_num, accu_num=accu_num, time_num=time_num,
                              group_num=group_num, law_input=law_input, law_adj_matrix=law_adj_matrix, trainable=False,
                              accu_relation=accu_relation, decoder_mode=decoder_mode, keep_coefficient=keep_coefficient,
                              embedding_trainable=embedding_trainable)

        DLadan_model.compile(optimizer=Adam(learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-7),
                             loss={'law': "categorical_crossentropy",
                                   'accu': "categorical_crossentropy",
                                   'time': "categorical_crossentropy",
                                   'group_prior': "categorical_crossentropy",
                                   'group_posterior': "categorical_crossentropy",
                                   'group_posterior_accu': "categorical_crossentropy"
                                   },
                             metrics={'law': "accuracy",
                                      'accu': "accuracy",
                                      'time': "accuracy",
                                      'group_prior': "accuracy",
                                      'group_posterior': "accuracy",
                                      'group_posterior_accu': "accuracy"
                                      },
                             loss_weights={'law': 1.0,
                                           'accu': 1.0,
                                           'time': 1.0,
                                           'group_prior': 0.1,
                                           'group_posterior': 0.1,
                                           'group_posterior_accu': 0.1
                                           },
                             run_eagerly=False)

        pred_sample = DLadan_model.predict(sample_D, steps=step, verbose=1)
        DLadan_model.load_weights(filepath=model_path, by_name=True)

        print('now_predicting')
        predictions = DLadan_model.predict(test_D, steps=step_test, verbose=1)
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

        tail_law_list = [81, 7, 43, 17, 14, 48, 76, 96, 15, 72, 70, 54, 69, 49, 53, 85, 56, 44]
        tail_charge_list = [114, 99, 86, 106, 36, 71, 104, 72, 95, 34, 39, 51, 23, 109, 62, 54, 31, 59, 60, 81, 110, 76, 26,
                            19, 64]

        law_preds = np.argmax(pred_law, axis=1)
        law_labels = np.argmax(gold_law, axis=1)

        accu_preds = np.argmax(pred_accu, axis=1)
        accu_labels = np.argmax(gold_accu, axis=1)

        law_labels, law_preds = filter_samples(law_preds, law_labels, tail_law_list)
        accu_labels, accu_preds = filter_samples(accu_preds, accu_labels, tail_charge_list)

        print('Tail Law article:')
        eval_data_types(law_labels, law_preds, num_labels=103, label_list=None)

        print('Tail Charges:')
        eval_data_types(accu_labels, accu_preds, num_labels=119, label_list=None)

        del DLadan_model
        K.clear_session()
        gc.collect()  # 清理内存

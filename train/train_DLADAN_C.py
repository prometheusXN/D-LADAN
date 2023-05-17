import sys
sys.path.append('..')
import argparse
import tensorflow.keras.backend as K
import gc
from Config.parser import ConfigParser
from Ladan.LadanPPK_Criminal_model import Ladanppk_Criminal
from Model.DLADAN_model_C import DLADAN_C
from data_preprocess.Criminal_reader import *
from tensorflow.keras.callbacks import *
from utils.evalution_component import evaluation_multitask
from utils.dataset_padding import padding_dataset
from utils.training_setup import setup_seed
from tensorflow.keras.optimizers import Adam
from Criminal_processes.law_process import *
from tensorflow.python.keras.engine.training import *
from sklearn import metrics


class UpdatePosteriorGraphMomentum(Callback):
    def __init__(self, DataSampling, momentum_steps=1, step_warmup=50):
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
    parser.add_argument('--gpu', '-g', type=str, default='0')
    parser.add_argument('--keep_coefficient', '-k', type=float, default=0.9)
    parser.add_argument('--warm_steps', '-w', type=int, default=1)
    parser.add_argument('--embedding_trainable', '-et', type=bool, default=False)
    parser.add_argument('--data_version', '-dv', type=str, default='small', choices=['small', 'medium', 'large'],
                        help='Set which dataset is used.')
    parser.add_argument('--momentum_steps', '-ms', type=int, default=1,
                        help='Set how much training steps between two momentum updating.')
    parser.add_argument('--version_index', '-vi', type=int, default=20230302,
                        help='The index for saving different model.')

    args = parser.parse_args()
    data_version = args.data_version
    version_index = args.version_index
    momentum_steps = args.momentum_steps
    model_path = "../model_save/LadanPPK/LadanPPK_Criminal_{}_{}.hdf5".format(data_version, version_index)

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

    train_set, _, steps = padding_dataset(train_set, batch_size)
    valid_set, _, _ = padding_dataset(valid_set, batch_size)

    train_D, _ = get_dataset_Criminal(data=train_set, batch_size=batch_size, shuffle=True,
                                      group_indexes=group_indexes_charge, PPK=True)
    valid_D, _ = get_dataset_Criminal(data=valid_set, batch_size=batch_size, shuffle=False,
                                      group_indexes=group_indexes_charge, PPK=True)

    # ------------------------------get_model-------------------------------#

    Ladan_model = DLADAN_C(config=config, word2id_dict=word2id_dict, emb_path=emb_path, group_num=group_num,
                           law_input=law_input, law_adj_matrix=law_adj_matrix, group_indexes=group_indexes_law,
                           trainable=True, accu_num=accu_num, embedding_trainable=False, keep_coefficient=0.90)

    Ladan_model.compile(optimizer=Adam(learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-7),
                        loss={
                            'accu': "categorical_crossentropy",
                            'group_prior': "categorical_crossentropy",
                            'group_posterior': "categorical_crossentropy"
                        },
                        metrics={
                            'accu': "accuracy",
                            'group_prior': "accuracy",
                            'group_posterior': "accuracy"
                        },
                        loss_weights={
                            'accu': 1.0,
                            'group_prior': 0.1,
                            'group_posterior': 0.1
                        },
                        run_eagerly=False)

    early_stopping = EarlyStopping(monitor='val_accu_accuracy', patience=5)  # 早停法，防止过拟合
    plateau = ReduceLROnPlateau(monitor="val_accu_accuracy", verbose=1, mode='max', factor=0.5, patience=3,
                                min_delta=1e-6)  # 当评价指标不在提升时，减少学习率
    checkpoint = ModelCheckpoint(model_path, monitor='val_accu_accuracy',
                                 verbose=1,
                                 save_best_only=True, mode='max', save_weights_only=True)

    update_graph = UpdatePosteriorGraphMomentum(DataSampling=None, step_warmup=int(args.warm_steps) * steps,
                                                momentum_steps=momentum_steps)

    Ladan_model.fit_warmingup(
        train_D,
        epochs=max_epoch,
        ave_acc=False,
        # epochs=1,
        # data_generator.__iter__(),
        # epochs=data_generator.get_epoch(),
        # steps_per_epoch=len(data_generator),
        validation_data=valid_D,
        callbacks=[
            # early_stopping,
            plateau,
            checkpoint,
            update_graph
        ],
        # triplet_data=data_generator
    )

    del Ladan_model
    K.clear_session()

    # ------------------------------------------now_testing-----------------------------------#
    sample_set, sample_num, step = padding_dataset(test_set[:batch_size], batch_size)
    sample_D, _ = get_dataset_Criminal(data=sample_set, batch_size=batch_size, shuffle=False,
                                       group_indexes=group_indexes_charge, PPK=True)
    test_set, sample_num_test, step_test = padding_dataset(test_set, batch_size)
    test_D, accu_labels = get_dataset_Criminal(data=test_set, batch_size=batch_size, shuffle=False,
                                               group_indexes=group_indexes_charge, PPK=True)

    Ladan_model = DLADAN_C(config=config, word2id_dict=word2id_dict, emb_path=emb_path, group_num=group_num,
                           law_input=law_input, law_adj_matrix=law_adj_matrix, group_indexes=group_indexes_law,
                           trainable=True, accu_num=accu_num, embedding_trainable=False)

    Ladan_model.compile(optimizer=Adam(learning_rate=learning_rate, beta_1=.9, beta_2=.999, epsilon=1e-7),
                        loss={
                            'accu': "categorical_crossentropy",
                            'group_prior': "categorical_crossentropy",
                            'group_posterior': "categorical_crossentropy"
                        },
                        metrics={
                            'accu': "accuracy",
                            'group_prior': "accuracy",
                            'group_posterior': "accuracy"
                        },
                        loss_weights={
                            'accu': 1.0,
                            'group_prior': 0.1,
                            'group_posterior': 0.1
                        },
                        run_eagerly=False)

    pred_sample = Ladan_model.predict(sample_D, steps=step, verbose=1)
    Ladan_model.load_weights(filepath=model_path, by_name=True)
    print('now_predicting')
    predictions = Ladan_model.predict(test_D, steps=step_test, verbose=1)

    pred_accu = predictions['accu'][:sample_num_test, :]
    gold_accu = accu_labels[:sample_num_test, :]

    metric = evaluation_multitask([gold_accu], [pred_accu], 1)

    print('Metrics for charge prediction is: ', metric[0])

    number_dict = {'low': [],
                   'medium': [],
                   'large': []}
    attribute_path = '../Criminal_Dataset/attributes'
    attribute_items = open(attribute_path, 'r').readlines()
    for item in attribute_items:
        item = item.strip().strip('\t').split('\t')
        sample_num = item[3]
        if int(sample_num) <= 100:
            number_dict['low'].append(int(item[0]))
        elif int(sample_num) > 1000:
            number_dict['large'].append(int(item[0]))
        else:
            number_dict['medium'].append(int(item[0]))

    pred_count_list = [0 for i in range(accu_num)]  # 记录每一个标签被预测的样本数量，为计算recall和 precision做准备
    gold_count_list = [0 for i in range(accu_num)]
    correct_count_list = [0 for i in range(accu_num)]
    pred_indexes = np.argmax(pred_accu, axis=-1)
    gold_indexes = np.argmax(gold_accu, axis=-1)
    for i in pred_indexes:
        for j in range(accu_num):
            if i == j:
                pred_count_list[j] += 1

    for i in gold_indexes:
        for j in range(accu_num):
            if i == j:
                gold_count_list[j] += 1

    for i in range(len(gold_indexes)):
        if pred_indexes[i] == gold_indexes[i]:
            correct_count_list[pred_indexes[i]] += 1
        else:
            continue

    print(pred_count_list)
    print(gold_count_list)
    print(correct_count_list)
    epslon = 1e-32
    acc_list = [float(correct_count_list[i]) / float(gold_count_list[i]) for i in range(accu_num)]
    recall_list = [(float(correct_count_list[i])) / (float(pred_count_list[i]) + epslon) for i in range(accu_num)]

    f1_list = [2 * acc_list[i] * recall_list[i] / (acc_list[i] + recall_list[i] + 1e-10) for i in range(accu_num)]

    print(acc_list)
    print(recall_list)
    print(f1_list)

    group_acc_list = [[], [], []]
    group_recall_list = [[], [], []]
    group_f1_list = [[], [], []]
    for index in number_dict['low']:
        group_acc_list[0].append(acc_list[int(index)])
        group_recall_list[0].append(recall_list[int(index)])
        group_f1_list[0].append(f1_list[int(index)])

    for index in number_dict['medium']:
        group_acc_list[1].append(acc_list[int(index)])
        group_recall_list[1].append(recall_list[int(index)])
        group_f1_list[1].append(f1_list[int(index)])

    for index in number_dict['large']:
        group_acc_list[2].append(acc_list[int(index)])
        group_recall_list[2].append(recall_list[int(index)])
        group_f1_list[2].append(f1_list[int(index)])

    f1_low = np.array(group_f1_list[0], dtype=np.float)
    f1_medium = np.array(group_f1_list[1], dtype=np.float)
    f1_large = np.array(group_f1_list[2], dtype=np.float)

    f1_Lo = np.mean(f1_low)
    f1_M = np.mean(f1_medium)
    f1_La = np.mean(f1_large)

    print(f1_Lo)
    print(f1_M)
    print(f1_La)

    del Ladan_model
    K.clear_session()

    gc.collect()  # 清理内存



import numpy as np
from data_preprocess.cail_reader import *


def sampling_label(probability, sample_num):
    probability = list(probability)
    threshold = []
    top = 0.0
    for i in probability:
        top += i
        threshold.append(top)
    threshold = np.expand_dims(np.array(threshold), axis=0)

    if sample_num==1:
        sampling_value = np.random.uniform()
    else:
        sampling_value = np.random.uniform(size=(sample_num, 1))
    index = np.sum((threshold < sampling_value).astype(np.int), axis=-1)    # size[batch, ]
    return index


def sampling_sample(candidate_num):
    sampling_value = np.random.uniform()
    sample_index = int(sampling_value*candidate_num)
    return sample_index


def sampling_negative(negative_probability):
    probability = list(negative_probability)
    threshold = []
    top = 0.0
    for i in probability:
        top += i
        threshold.append(top)
    threshold = np.array(threshold)
    sampling_value = np.random.uniform()
    index = np.sum((threshold < sampling_value).astype(np.int), axis=-1)  # scalar
    return index


def softmax(x):
    """ softmax function """
    # assert(len(x.shape) > 1, "dimension must be larger than 1")
    # print(np.max(x, axis = 1, keepdims = True)) # axis = 1, 行
    x -= np.max(x, axis=-1, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    x = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    return x


def split_dataset_LawLabel(dataset, label_num):
    """
    :param dataset: the datalist, where:
            list[0]：input;
            list[1]: label of law;
            list[2]: label of charges;
            list[3]: label of times.
    :return: a dict of dataset, where the key is the law label index, the value is data set with the corresponding label.
    """
    data_dict = {i:[] for i in range(label_num)}
    for i in dataset:
        law_indexs = np.argmax(i[1], axis=-1)
        data_dict[law_indexs].append(i)
    return data_dict


def split_dataset_AccuLabel(dataset, accu_num):
    """
    :param dataset: the datalist, where:
            list[0]：input;
            list[1]: label of law;
            list[2]: label of charges;
            list[3]: label of times.
    :return: a dict of dataset, where the key is the law label index, the value is data set with the corresponding label.
    """
    data_dict = {i:[] for i in range(accu_num)}
    for i in dataset:
        accu_indexs = np.argmax(i[2], axis=-1)
        data_dict[accu_indexs].append(i)
    return data_dict


def split_dataset_TimeLabel(dataset, time_num):
    """
    :param dataset: the datalist, where:
            list[0]：input;
            list[1]: label of law;
            list[2]: label of charges;
            list[3]: label of times.
    :return: a dict of dataset, where the key is the law label index, the value is data set with the corresponding label.
    """
    data_dict = {i:[] for i in range(time_num)}
    for i in dataset:
        accu_indexs = np.argmax(i[3], axis=-1)
        data_dict[accu_indexs].append(i)
    return data_dict


def compute_probability(CosineMatric: np.ndarray, NumberList):
    """
    :param CosineMatric: a matrix records the relation between law labels
    :param NumberList: a list records the number of train samples with the corresponding label.
    :return: LabelProbability: the select probability of anchor law label
             NegativeProbability: the select probability of negative law label when the anchor is determined
    """
    numbers = np.array(NumberList, dtype=np.float)
    LabelProbability = softmax(np.max(CosineMatric, axis=-1)/np.power(numbers, (float(1.0)/float(3.0))) * 6.0)
    NegativeProbability = softmax(CosineMatric*5.0 * np.power(numbers, (float(1.0)/float(3.0))))

    return LabelProbability, NegativeProbability


def merge_data(anchor_list, positive_list, negative_list):
    fact_descriprions = []
    law_labels = []
    accu_labels = []
    time_labels = []

    def update_list(data_list, facts, laws, accus, times):
        data_facts, data_laws, data_accus, data_times = list(zip(*data_list))
        facts += list(data_facts)
        data_laws = list(data_laws)
        laws += data_laws
        accus += list(data_accus)
        times += list(data_times)
        return facts, laws, accus, times

    fact_descriprions, law_labels, accu_labels, time_labels \
        = update_list(anchor_list, fact_descriprions, law_labels, accu_labels, time_labels)
    fact_descriprions, law_labels, accu_labels, time_labels \
        = update_list(positive_list, fact_descriprions, law_labels, accu_labels, time_labels)
    fact_descriprions, law_labels, accu_labels, time_labels \
        = update_list(negative_list, fact_descriprions, law_labels, accu_labels, time_labels)

    return fact_descriprions, law_labels, accu_labels, time_labels


def sampling_one_batch(data_dict, cos_matrix, batch_size, group_indexes):
    """
    :param data_dict:
    :param cos_matrix:
    :param batch_size:
    :return:
    """
    sample_size = int(batch_size/3)
    number_list = [len(i) for i in list(data_dict.values())]
    class_num = list(cos_matrix.shape)[0]
    self_loop = np.eye(N=class_num)
    cos_matrix = cos_matrix - 10000 * self_loop

    LabelProbability, NegativeProbability = compute_probability(cos_matrix, number_list)

    anchor_indexes = list(sampling_label(LabelProbability, sample_num=sample_size))
    anchor_list = []
    pos_list = []
    neg_list = []
    neg_index = []

    for xa in anchor_indexes:
        candidate_num_anchor = number_list[xa]
        negative_probability = NegativeProbability[xa, :]
        xn = int(sampling_negative(negative_probability=negative_probability))
        if xn == xa:
            xn -= 1
        neg_index.append(xn)
        candidate_num_negative = number_list[xn]
        ya = sampling_sample(candidate_num_anchor)
        yp = sampling_sample(candidate_num_anchor)
        if yp == ya:
            yp -= 1
        yn = sampling_sample(candidate_num_negative)
        anchor_list.append(data_dict[xa][ya])
        pos_list.append(data_dict[xa][yp])
        neg_list.append(data_dict[xn][yn])

    # print(indexes)
    # print(neg_index)

    facts, law_labels, accu_labels, time_labels = merge_data(anchor_list, pos_list, neg_list)
    laws = np.array(law_labels)
    group_num = np.max(group_indexes)
    group_labels = list(to_categorical(np.sum(laws.astype(int) * group_indexes, axis=-1), num_classes=group_num+1))

    return facts, law_labels, accu_labels, time_labels, group_labels


def compute_cos(feature_matrix: np.ndarray):
    x_norm = np.linalg.norm(feature_matrix, axis=-1, keepdims=True)
    cos_matrix = np.dot(feature_matrix, feature_matrix.T)/np.dot(x_norm, x_norm.T)
    node_num = list(np.shape(feature_matrix))[0]
    self_loop = np.eye(node_num, dtype=np.float)
    cos_matrix = cos_matrix * (1.0 - self_loop)
    return cos_matrix


class DataGeneratorSampling:
    def __init__(self, dataset, group_indexes, batch_size, label_num, epoch_num, steps=794, mode='roll'):
        self.dataset = dataset.copy()
        # self.steps = steps
        self.data_dict_byLaw = split_dataset_LawLabel(self.dataset, label_num=label_num)
        # the training sample grouped by law_label
        self.group_indexes = group_indexes
        class_num = len(list(group_indexes))
        self.batch_size = batch_size
        self.cos_matrix = np.zeros(shape=[class_num, class_num], dtype=np.float)
        self.epoch_num = epoch_num

        self.mode = mode
        if self.mode=='roll':
            samples_per_batch = self.batch_size // 3
            if len(dataset) % samples_per_batch == 0:
                self.steps = len(dataset) // samples_per_batch
            else:
                self.steps = len(dataset) // samples_per_batch + 1
            padding_num = self.steps * samples_per_batch - len(dataset)
            self.dataset += self.dataset[:padding_num]
        else:
            self.steps = steps

    def __len__(self):
        return self.steps

    def set_matrix(self, cos_matrix):
        self.cos_matrix = cos_matrix

    def get_matrix(self):
        return self.cos_matrix

    def get_epoch(self):
        return self.epoch_num

    def shuffle_dataset(self):
        idxs = np.array(range(len(self.dataset)))
        np.random.shuffle(idxs)
        dataset = []
        for i in idxs:
            dataset.append(self.dataset[i])
        self.dataset = dataset

    def compute_probability(self, CosineMatric: np.ndarray, NumberList):
        """
        :param CosineMatric: a matrix records the relation between law labels
        :param NumberList: a list records the number of train samples with the corresponding label.
        :return: LabelProbability: the select probability of anchor law label
                 NegativeProbability: the select probability of negative law label when the anchor is determined
        """
        numbers = np.array(NumberList, dtype=np.float)
        LabelProbability = softmax(np.max(CosineMatric, axis=-1) / np.power(numbers, (float(1.0) / float(3.0))) * 6.0)
        NegativeProbability = softmax(CosineMatric * 5.0 * np.expand_dims(np.power(numbers, (float(1.0) / float(3.0))), axis=0))
        return LabelProbability, NegativeProbability

    def sampling_one_batch(self, sampling_num, start_index, cos_matrix):
        anchor_list = []
        pos_list = []
        neg_list = []
        number_list_law = [len(i) for i in list(self.data_dict_byLaw.values())]

        class_num = list(cos_matrix.shape)[0]
        self_loop = np.eye(N=class_num)
        cos_matrix = cos_matrix - 10000 * self_loop

        LabelProbability, NegativeProbability = self.compute_probability(cos_matrix, number_list_law)

        for i in range(start_index, start_index + sampling_num, 1):
            neg_data = self.dataset[i]
            neg_list.append(neg_data)
            neg_label = int(np.argmax(neg_data[1], axis=-1))

            negative_probability = NegativeProbability[neg_label, :]

            anchor_label = int(sampling_label(LabelProbability, sample_num=1))
            if anchor_label == neg_label:
                anchor_label = int(sampling_label(LabelProbability, sample_num=1))

            candidate_num_anchor = number_list_law[anchor_label]
            ya = sampling_sample(candidate_num_anchor)
            yp = sampling_sample(candidate_num_anchor)

            if (self.data_dict_byLaw[anchor_label][yp][0] == self.data_dict_byLaw[anchor_label][ya][0]).all():
                yp -= 1

            anchor_data = self.data_dict_byLaw[anchor_label][ya]
            anchor_list.append(anchor_data)
            pos_data = self.data_dict_byLaw[anchor_label][yp]
            pos_list.append(pos_data)

        # print(indexes)
        # print(neg_index)

        facts, law_labels, accu_labels, time_labels = merge_data(anchor_list, pos_list, neg_list)
        laws = np.array(law_labels)
        group_num = np.max(self.group_indexes)
        group_labels = list(
            to_categorical(np.sum(laws.astype(int) * self.group_indexes, axis=-1), num_classes=group_num + 1))

        return facts, law_labels, accu_labels, time_labels, group_labels

    def __iter__(self):
        sample_num = self.batch_size // 3
        start_index = 0
        step = 0
        self.shuffle_dataset()
        while True:
            if (start_index+sample_num) >= len(self.dataset):
                start_index = 0
            cos_matrix = self.get_matrix()

            if self.mode=='roll':
                facts, law_labels, accu_labels, time_labels, group_labels \
                    = self.sampling_one_batch(sampling_num=sample_num, start_index=start_index, cos_matrix=cos_matrix)
            else:
                facts, law_labels, accu_labels, time_labels, group_labels \
                    = sampling_one_batch(self.data_dict_byLaw, cos_matrix, self.batch_size, self.group_indexes)

            step += 1
            start_index += sample_num

            if len(facts) != self.batch_size:
                padding_num = self.batch_size - len(facts)
                facts += facts[-padding_num:]
                law_labels += law_labels[-padding_num:]
                accu_labels += accu_labels[-padding_num:]
                time_labels += time_labels[-padding_num:]
                group_labels += group_labels[-padding_num:]
            model_input = tf.convert_to_tensor(np.array(facts))
            model_output = {'law': tf.convert_to_tensor(np.array(law_labels)),
                            'accu': tf.convert_to_tensor(np.array(accu_labels)),
                            'time': tf.convert_to_tensor(np.array(time_labels)),
                            'group_prior': tf.convert_to_tensor(np.array(group_labels)),
                            'group_posterior': tf.convert_to_tensor(np.array(group_labels))
                            }
            if step >= (self.steps-1):
                step = 0
                self.shuffle_dataset()
            yield model_input, model_output


class DataGeneratorSampling_LawandAccu:
    def __init__(self, dataset, steps, group_indexes, batch_size, law_num, accu_num, time_num, epoch_num):
        self.dataset = dataset
        self.steps = steps
        self.data_dict_byLaw = split_dataset_LawLabel(self.dataset, label_num=law_num)
        self.data_dict_byAccu = split_dataset_AccuLabel(self.dataset, accu_num=accu_num)
        self.data_dict_byTime = split_dataset_TimeLabel(self.dataset, time_num=time_num)
        # the training sample grouped by law_label
        self.group_indexes = group_indexes
        class_num = len(list(group_indexes))
        self.batch_size = batch_size
        self.cos_matrix_law = np.zeros(shape=[law_num, law_num], dtype=np.float)
        self.cos_matrix_accu = np.zeros(shape=[accu_num, accu_num], dtype=np.float)
        self.cos_matrix_time = np.zeros(shape=[time_num, time_num], dtype=np.float)
        self.epoch_num = epoch_num

    def __len__(self):
        return self.steps

    def set_law_matrix(self, cos_matrix):
        self.cos_matrix_law = cos_matrix

    def set_accu_matrix(self, cos_matrix):
        self.cos_matrix_accu = cos_matrix

    def set_time_matrix(self, cos_matrix):
        self.cos_matrix_time = cos_matrix

    def get_law_matrix(self):
        return self.cos_matrix_law

    def get_accu_matrix(self):
        return self.cos_matrix_accu

    def get_time_matrix(self):
        return self.cos_matrix_time

    def get_epoch(self):
        return self.epoch_num

    def __iter__(self):
        task_size = (self.batch_size//6) * 3
        while True:
            cos_matrix_law = self.get_law_matrix()
            facts_l, law_labels_l, accu_labels_l, time_labels_l, group_labels_l \
                = sampling_one_batch(self.data_dict_byLaw, cos_matrix_law, task_size, self.group_indexes)

            cos_matrix_accu = self.get_accu_matrix()
            facts_a, law_labels_a, accu_labels_a, time_labels_a, group_labels_a \
                = sampling_one_batch(self.data_dict_byAccu, cos_matrix_accu, task_size, self.group_indexes)

            # cos_matrix_time = self.get_time_matrix()
            # facts_t, law_labels_t, accu_labels_t, time_labels_t, group_labels_t \
            #     = sampling_one_batch(self.data_dict_byTime, cos_matrix_time, task_size, self.group_indexes)

            # facts = facts_l + facts_a + facts_t
            # law_labels = law_labels_l + law_labels_a + law_labels_t
            # accu_labels = accu_labels_l + accu_labels_a + accu_labels_t
            # time_labels = time_labels_l + time_labels_a + time_labels_t
            # group_labels = group_labels_l + group_labels_a + group_labels_t

            facts = facts_l + facts_a
            law_labels = law_labels_l + law_labels_a
            accu_labels = accu_labels_l + accu_labels_a
            time_labels = time_labels_l + time_labels_a
            group_labels = group_labels_l + group_labels_a

            if len(facts) != self.batch_size:
                padding_num = self.batch_size - len(facts)
                facts += facts[-padding_num:]
                law_labels += law_labels[-padding_num:]
                accu_labels += accu_labels[-padding_num:]
                time_labels += time_labels[-padding_num:]
                group_labels += group_labels[-padding_num:]
            model_input = tf.convert_to_tensor(np.array(facts))
            model_output = {'law': tf.convert_to_tensor(np.array(law_labels)),
                            'accu': tf.convert_to_tensor(np.array(accu_labels)),
                            'time': tf.convert_to_tensor(np.array(time_labels)),
                            'group_prior': tf.convert_to_tensor(np.array(group_labels)),
                            'group_posterior': tf.convert_to_tensor(np.array(group_labels))
                            }
            yield model_input, model_output


class NormalSamplingDataGnenrator:
    def __init__(self, dataset, group_indexes, batch_size, law_num, accu_num, time_num, epoch_num):
        self.dataset = dataset
        self.data_dict_byLaw = split_dataset_LawLabel(self.dataset, label_num=law_num) # the training sample grouped by law_label
        self.data_dict_byAccu = split_dataset_AccuLabel(self.dataset, accu_num=accu_num)
        self.data_dict_byTime = split_dataset_TimeLabel(self.dataset, time_num=time_num)
        self.law_num = law_num
        self.accu_num = accu_num
        self.time_num = time_num
        self.group_indexes = group_indexes
        self.batch_size = batch_size
        self.cos_matrix_law = np.zeros(shape=[law_num, law_num], dtype=np.float)
        self.cos_matrix_accu = np.zeros(shape=[accu_num, accu_num], dtype=np.float)
        self.cos_matrix_time = np.zeros(shape=[time_num, time_num], dtype=np.float)
        self.epoch_num = epoch_num

        samples_per_batch = self.batch_size//3
        self.steps = len(dataset)//samples_per_batch + 1
        padding_num = self.steps * samples_per_batch - len(dataset)
        self.dataset += dataset[:padding_num]

    def __len__(self):
        return self.steps

    def set_law_matrix(self, cos_matrix):
        self.cos_matrix_law = cos_matrix

    def set_accu_matrix(self, cos_matrix):
        self.cos_matrix_accu = cos_matrix

    def set_time_matrix(self, cos_matrix):
        self.cos_matrix_time = cos_matrix

    def get_law_matrix(self):
        return self.cos_matrix_law

    def get_accu_matrix(self):
        return self.cos_matrix_accu

    def get_time_matrix(self):
        return self.cos_matrix_time

    def get_epoch(self):
        return self.epoch_num

    def sampling_one_batch(self, sampling_num, start_index):
        '''
        :param sampling_num: the sample number of anchor element in one triplet
        :return:
        '''
        anchor_list = []
        pos_list = []
        neg_list = []
        global_num = len(self.dataset)
        number_list_law = [len(i) for i in list(self.data_dict_byLaw.values())]
        for i in range(start_index, start_index+sampling_num, 1):
            anchor_date = self.dataset[i]
            anchor_list.append(anchor_date)
            anchor_label = np.argmax(anchor_date[1], axis=-1)
            candidate_num_anchor = number_list_law[int(anchor_label)]
            yp = sampling_sample(candidate_num_anchor)
            if (self.data_dict_byLaw[int(anchor_label)][yp][0] == anchor_date[0]).all():
                yp -= 1
            pos_data = self.data_dict_byLaw[int(anchor_label)][yp]
            pos_list.append(pos_data)

            yn = sampling_sample(global_num)
            if (self.dataset[yn][1] == anchor_date[1]).all():
                yn = sampling_sample(global_num)

            neg_data = self.dataset[yn]
            neg_list.append(neg_data)

        facts, law_labels, accu_labels, time_labels = merge_data(anchor_list, pos_list, neg_list)
        laws = np.array(law_labels)
        group_num = np.max(self.group_indexes)
        group_labels = list(
            to_categorical(np.sum(laws.astype(int) * self.group_indexes, axis=-1), num_classes=group_num + 1))

        return facts, law_labels, accu_labels, time_labels, group_labels

    def shuffle_dataset(self):
        idxs = np.array(range(len(self.dataset)))
        np.random.shuffle(idxs)
        dataset = []
        for i in idxs:
            dataset.append(self.dataset[i])
        self.dataset = dataset

    def __iter__(self):
        sample_num = self.batch_size//3
        start_index = 0
        step = 0
        self.shuffle_dataset()
        while True:
            if (start_index+sample_num) >= len(self.dataset):
                start_index = 0

            facts, law_labels, accu_labels, time_labels, group_labels \
                = self.sampling_one_batch(sampling_num=sample_num, start_index=start_index)

            step += 1
            start_index += sample_num

            if len(facts) != self.batch_size:
                padding_num = self.batch_size - len(facts)
                facts += facts[-padding_num:]
                law_labels += law_labels[-padding_num:]
                accu_labels += accu_labels[-padding_num:]
                time_labels += time_labels[-padding_num:]
                group_labels += group_labels[-padding_num:]
            model_input = tf.convert_to_tensor(np.array(facts))
            model_output = {'law': tf.convert_to_tensor(np.array(law_labels)),
                            'accu': tf.convert_to_tensor(np.array(accu_labels)),
                            'time': tf.convert_to_tensor(np.array(time_labels)),
                            'group_prior': tf.convert_to_tensor(np.array(group_labels)),
                            'group_posterior': tf.convert_to_tensor(np.array(group_labels))
                            }

            if step >= (self.steps-1):
                step = 0
                self.shuffle_dataset()
            yield model_input, model_output

import pickle as pkl
from utils.data_statistic import get_statistic
from data_preprocess.cail_reader import read_cails


def get_label_index(index2num_list, num_threshold_low, num_threshold_high):
    index_list = []
    pair_list = list(zip(index2num_list[0].tolist(), index2num_list[1].tolist()))
    for index, num in pair_list:
        if num_threshold_low <= num <= num_threshold_high:
            index_list.append(index)
    return index_list


if __name__ == '__main__':
    data_path = '/home/nxu/Ladan_tnnls/processed_dataset/CAIL'
    train_dir = data_path + '/normal/small/train_processed_thulac.pkl'
    dataset_fold = read_cails(dir_path=data_path,
                              data_format='normal',
                              version='small')

    train_set, valid_set, test_set = next(dataset_fold)

    law_statisitc, accu_statistic, time_statistic = get_statistic(train_set, law_num=103, accu_num=119,
                                                                  time_num=11)

    print(law_statisitc)

    print(accu_statistic)

    print(time_statistic)

    print(get_label_index(law_statisitc, 100, 200))

    print(get_label_index(accu_statistic, 100, 200))

    tail_law_list = [37, 89, 74, 66, 25, 82, 32, 36, 5, 81, 7, 43, 17, 14, 48, 76, 96, 15, 72, 70, 54, 69, 49, 53, 85, 56, 44]
    tail_charge_list = [37, 85, 96, 114, 99, 86, 106, 36, 71, 104, 72, 95, 34, 39, 51, 23, 109, 62, 54, 31, 59, 60, 81, 110, 76, 26, 19, 64]

    tail_100to200_law = [81, 7, 43, 17, 14, 48, 76, 96, 15, 72, 70, 54, 69, 49, 53, 85, 56, 44]
    tail_100to200_charge = [114, 99, 86, 106, 36, 71, 104, 72, 95, 34, 39, 51, 23, 109, 62, 54, 31, 59, 60, 81, 110, 76, 26, 19, 64]
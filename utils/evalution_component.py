from sklearn import metrics
import numpy as np
from tensorflow.keras.callbacks import *


# class AveAccuracyCallback(Callback):
#     def __init__(self, predict_batch_size=32, include_on_batch=False):
#         super(AveAccuracyCallback, self).__init__()
#         self.predict_batch_size = predict_batch_size
#         self.include_on_batch = include_on_batch
#
#     def on_batch_begin(self, batch, logs={}):
#         pass
#
#     def on_train_begin(self, logs={}):
#         if not ('avg_f1_score_val' in self.params['metrics']):
#             self.params['metrics'].append('avg_f1_score_val')
#
#     def on_batch_end(self, batch, logs={}):
#         if (self.include_on_batch):
#             logs['avg_f1_score_val'] = float('-inf')
#
#     def on_epoch_end(self, epoch, logs={}):
#         logs['avg_f1_score_val'] = float('-inf')
#         if (self.validation_data):
#             law_pred, accu_pred, time_pred = self.model.predict(self.validation_data[0],
#                                            batch_size=self.predict_batch_size)
#             law_label, accu_label, time_label = self.validation_data[1]
#
#             law_accuracy = metrics.accuracy_score(law_label, law_pred)
#             accu_accuracy = metrics.accuracy_score(accu_label, accu_pred)
#             time_accuracy = metrics.accuracy_score(time_label, time_pred)
#
#             avgacc=(law_accuracy + accu_accuracy + time_accuracy) / 3
#             # print("avg_f1_score %.4f " % (avgf1))
#             logs['avg_acc_val'] =avgacc


def evaluation_multitask(y, prediction, task_num):
    metrics_acc = []
    for x in range(task_num):
        y_pred = np.argmax(prediction[x], axis=1)
        y_true = np.argmax(y[x], axis=1)
        accuracy_metric = metrics.accuracy_score(y_true, y_pred)
        macro_recall = metrics.recall_score(y_true, y_pred, average='macro')
        micro_recall = metrics.recall_score(y_true, y_pred, average='micro')
        macro_precision = metrics.precision_score(y_true, y_pred, average='macro')
        micro_precision = metrics.precision_score(y_true, y_pred, average='micro')
        macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')
        micro_f1 = metrics.f1_score(y_true, y_pred, average='micro')
        metrics_acc.append(
            (accuracy_metric, macro_recall, micro_recall, macro_precision, micro_precision, macro_f1, micro_f1))
    return metrics_acc


def evaluation(y_pred, y_true):
    accuracy_metric = metrics.accuracy_score(y_true, y_pred)
    macro_recall = metrics.recall_score(y_true, y_pred, average='macro')
    macro_precision = metrics.precision_score(y_true, y_pred, average='macro')
    macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')
    return accuracy_metric, macro_recall, macro_precision, macro_f1


def evaluation_label_list(y_pred, y_true, label_list):
    y_pred_list = np.argmax(y_pred, axis=1).tolist()
    y_true_list = np.argmax(y_true, axis=1).tolist()
    y_true_filtered = []
    y_pred_filtered = []
    for i in range(len(y_pred_list)):
        if y_true_list[i] in label_list:
            y_true_filtered.append(y_true_list[i])
            y_pred_filtered.append(y_pred_list[i])
    y_true = np.array(y_true_filtered)
    y_pred = np.array(y_pred_filtered)
    return evaluation(y_pred, y_true)


def get_value(res):
    if res["TP"] == 0:
        if res["FP"] == 0 and res["FN"] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
        recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def gen_result(res, label_list=None, total_num=None):
    precision = []
    recall = []
    f1 = []
    # print('sample num:', total_num)
    total = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    if label_list:
        print('compute the label list')
        for a in label_list:
            total["TP"] += res[a]["TP"]
            total["FP"] += res[a]["FP"]
            total["FN"] += res[a]["FN"]
            total["TN"] += res[a]["TN"]

            p, r, f = get_value(res[a])
            precision.append(p)
            recall.append(r)
            f1.append(f)
    else:
        for a in range(0, len(res)):
            total["TP"] += res[a]["TP"]
            total["FP"] += res[a]["FP"]
            total["FN"] += res[a]["FN"]
            total["TN"] += res[a]["TN"]

            p, r, f = get_value(res[a])
            precision.append(p)
            recall.append(r)
            f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_value(total)
    # print(total)
    acc = 1.0 * total['TP']/total_num

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    for a in range(0, len(f1)):
        macro_precision += precision[a]
        macro_recall += recall[a]
        macro_f1 += f1[a]

    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    print("Accuracy\t%.4f" % acc)
    print("Micro precision\t%.4f" % micro_precision)
    print("Micro recall\t%.4f" % micro_recall)
    print("Micro f1\t%.4f" % micro_f1)
    print("Macro precision\t%.4f" % macro_precision)
    print("Macro recall\t%.4f" % macro_recall)
    print("Macro f1\t%.4f" % macro_f1)

    return


def eval_data_types(target, prediction, num_labels, label_list=None):
    ground_truth_v2 = []
    predictions_v2 = []
    total_num = 0
    for i in target:
        total_num += 1
        v = [0 for j in range(num_labels)]
        v[i] = 1
        ground_truth_v2.append(v)
    for i in prediction:
        v = [0 for j in range(num_labels)]
        v[i] = 1
        predictions_v2.append(v)

    res = []
    for i in range(num_labels):
        res.append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})
    y_true = np.array(ground_truth_v2)
    y_pred = np.array(predictions_v2)
    for i in range(num_labels):
        outputs1 = y_pred[:, i]
        labels1 = y_true[:, i]
        res[i]["TP"] += int((labels1 * outputs1).sum())
        res[i]["FN"] += int((labels1 * (1 - outputs1)).sum())
        res[i]["FP"] += int(((1 - labels1) * outputs1).sum())
        res[i]["TN"] += int(((1 - labels1) * (1 - outputs1)).sum())
    gen_result(res, label_list, total_num)

    return 0


def filter_samples(y_pred, y_true, label_list):
    y_pred_list = y_pred.tolist()
    y_true_list = y_true.tolist()
    y_true_filtered = []
    y_pred_filtered = []
    for i in range(len(y_pred_list)):
        if y_true_list[i] in label_list:
            y_true_filtered.append(y_true_list[i])
            y_pred_filtered.append(y_pred_list[i])

    return y_true_filtered, y_pred_filtered


if __name__ == '__main__':
    label_list = [0, 2]

    y_pred = np.argmax(np.array([[1, 0, 0], [0, 0, 1], [0, 0, 1]]), axis=1)
    print(type(y_pred))
    y_pred_list = []
    for i in range(len(y_pred.tolist())):
        if y_pred.tolist()[i] in label_list:
            y_pred_list.append(y_pred.tolist()[i])
    print(y_pred_list)

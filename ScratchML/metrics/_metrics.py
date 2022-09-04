import numpy as np
import math


def accuracy_score(y_true, y_pred):
    total = len(y_true)
    if len(y_pred) != len(y_true):
        raise Exception("Mismatch no. of elements")
    count = sum([1 for x in range(len(y_true)) if y_true[x] == y_pred[x]])
    return round(count / total, 3)


def confusion_matrix(y_true, y_pred):
    if len(y_pred) != len(y_true):
        raise Exception("Mismatch no. of elements")
    for i in list(set(y_pred)):
        if i not in list(set(y_true)):
            raise Exception("Unknown element in y_pred")
    conf_mat = np.zeros((len(set(y_true)), len(set(y_true))))
    map_value = dict(
        zip(list(set(y_true)), [x for x in range(len(set(y_true)))]))
    for r, v in zip(y_true, y_pred):
        conf_mat[map_value[r]][map_value[v]] += 1
    return conf_mat


def precision_score(y_true, y_pred, average='binary'):
    if len(y_pred) != len(y_true):
        raise Exception("Mismatch no. of elements")
    if average == 'binary' and len(set(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fp)
    else:
        raise Exception('Yet to be implemented')


def recall_score(y_true, y_pred, average='binary'):
    if len(y_pred) != len(y_true):
        raise Exception("Mismatch no. of elements")
    if average == 'binary' and len(set(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn)
    else:
        raise Exception('Yet to be implemented')


def f1_score(y_true, y_pred, average='binary'):
    if len(y_pred) != len(y_true):
        raise Exception("Mismatch no. of elements")
    if average == 'binary' and len(set(y_true)) == 2:
        pr = precision_score(y_true, y_pred, average='binary')
        re = recall_score(y_true, y_pred, average='binary')
        f1_score = 2 * ((pr * re) / (pr + re))
        return f1_score
    else:
        raise Exception('Yet to be implemented')


def log_loss(y_true, y_prob):
    if len(y_true) != len(y_prob):
        raise Exception("Mismatch no. of elements")
    log_score = 0
    for i in range(len(y_true)):
        for j in range(2):
            log_score += -((y_true[i] * np.log(y_prob[i][j])) +
                           (1 - y_true[i]) * np.log(1 - y_prob[i][j]))
    return log_score


def r2_score(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise Exception("Mismatch no. of elements")
    mean_val = np.mean(y_true)
    SStotal = np.mean([(x - mean_val)**2 for x in y_true])
    SSres = []
    for i in range(len(y_true)):
        SSres.append((y_true[i] - y_pred[i])**2)
    SSres = np.mean(SSres)
    return 1 - (SSres / SStotal)

def mean_absolute_error(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise Exception("Mismatch no. of elements")
    error_val = []
    for i in range(len(y_true)):
        error_val.append(np.abs(y_true[i] - y_pred[i]))
    return np.mean(error_val)

def rmse(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise Exception("Mismatch no. of elements")
    error_val = []
    for i in range(len(y_true)):
        error_val.append((y_true[i] - y_pred[i])**2)
    return math.sqrt(np.mean(error_val))
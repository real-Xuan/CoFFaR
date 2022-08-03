# coding=utf-8
import numpy as np
from sklearn.model_selection import KFold
from scipy.optimize import brentq
from scipy import interpolate


def evaluate(distances, labels):
    '''
            : Evaluate

            :parameter distances: Distances of two face images.
            :parameter labels:  The true labels which whether it same people or not.

            :return: None
    '''

    # Find the maximum distance in list.
    distMax = np.max(distances)
    # print("distMax: ", distMax)

    # Find the minimum distance in list.
    distMin = np.min(distances)
    # print("distMin: ", distMin)

    # Mapping distances to [0,1]
    distances = [(x - distMin) / (distMax - distMin) for x in distances]
    #print("distances: ", distances)

    # Value threshold by specific step
    #thresholds = np.arange(distMin, distMax, 0.01)
    thresholds = np.arange(0, 1, 0.0001)
    #print("thresholds: ", thresholds)

    tpr, fpr, accuracy, best_threshold = calculate_roc(thresholds, distances, labels)
    print("tpr: ", tpr)
    print("fpr: ", fpr)

    # Find the value of TAR for FAR is 0.001, 0.01 and 0.1.
    # TAR @ FAR = 0.1 / 0.01 / 0.001, FAR = FPR, TAR = TPR
    # TAR_FAR_E1 = brentq(lambda x: 0.1 - interpolate.interp1d(tpr, fpr, fill_value="extrapolate")(x), 0., 1.)
    # TAR_FAR_E2 = brentq(lambda x: 0.01 - interpolate.interp1d(tpr, fpr, fill_value="extrapolate")(x), 0., 1.)
    # TAR_FAR_E3 = brentq(lambda x: 0.001 - interpolate.interp1d(tpr, fpr, fill_value="extrapolate")(x), 0., 1.)

    # print("TAR_FAR_E1: ", TAR_FAR_E1)
    # print("TAR_FAR_E2: ", TAR_FAR_E2)
    # print("TAR_FAR_E3: ", TAR_FAR_E3)

    return tpr, fpr, accuracy, best_threshold


def calculate_roc(thresholds, distances, labels):
    assert len(labels) == len(distances)
    # Face pairs count 5320
    nrof_pairs = min(len(labels), len(distances))

    nrof_thresholds = len(thresholds)  # The number of thresholds

    tprs = np.zeros(nrof_thresholds)
    fprs = np.zeros(nrof_thresholds)

    # print("thresholds: ", thresholds)

    # Find the best threshold for the fold
    acc_train = np.array(np.zeros(nrof_thresholds))

    for threshold_idx, threshold in enumerate(thresholds):
        tprs[threshold_idx], fprs[threshold_idx], acc_train[threshold_idx] = calculate_accuracy(
            threshold, distances, labels)

    # print("acc_train", acc_train)
    #
    best_threshold_index = np.argmax(acc_train)
    # print("acc_train: ", acc_train[best_threshold_index])
    # print("best_threshold_index: ", best_threshold_index)
    best_threshold = thresholds[best_threshold_index]

    if np.mean(acc_train[best_threshold_index]) < 0.49:
        print("accuracy： ", acc_train[best_threshold_index])
        raise Exception

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)

    return tpr, fpr, acc_train[best_threshold_index], best_threshold
    # return tpr, fpr, np.mean(accuracy), best_threshold


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    # predict_issame = np.greater(dist, threshold)
    # print("predict_issame: ", predict_issame)

    assert len(predict_issame) == len(actual_issame)

    tp = np.sum(np.logical_and(predict_issame, actual_issame) != 0)
    # print("tp: ", tp)
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)) != 0)
    # print("fp: ", fp)
    tn = np.sum(np.logical_and(np.logical_not(predict_issame),
                               np.logical_not(actual_issame)) != 0)
    # print("tn: ", tn)
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame) != 0)
    # print("fn: ", fn)

    # print("tp + fp + tn + fn: ", tp + fp + tn + fn)
    # print("len(actual_issame): ", len(actual_issame))
    assert tp + fp + tn + fn == len(actual_issame)

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    # print("tp + tn: ", tp+tn)
    # print("dist.size: ", dist.size)

    # err = float(fp + fn) / dist.size
    # acc = 1 - err

    # acc = (tp + fp + tn + fn) / len(actual_issame)
    acc = (tp + tn) / len(actual_issame)
    # print("acc: ", acc)

    # if acc < 0.48: raise Exception

    return tpr, fpr, acc

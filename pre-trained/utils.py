import os
import sys
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


from matplotlib import pyplot as plt
from sklearn import metrics
from scipy import interpolate
from scipy.optimize import brentq



def plot_features(features, labels, num_classes, epoch=1, isTest=False):
    """Plot features on 2D plane.
    Args:
        num_classes: 类别数量
        features: (num_instances, num_features).
        labels: (num_instances).
    """
    colors = ['green', 'blue']

    for label_idx in range(num_classes):
        plt.scatter(
            features[labels == label_idx, 0],
            features[labels == label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.legend(['Diffrent ID', 'Same ID'], loc='upper right')
    dirname = './plotsave'
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    if not isTest:
        save_name = os.path.join(dirname, 'epoch_' + str(epoch + 1) + '.png')
    else: save_name = os.path.join(dirname, 'epoch_test' + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        # 这个函数是用来求output中的最大值或最小值，返回两个参数：其一返回output中的最大值（或最小值），其二返回该值的索引。
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class Logger(object):
    """
    Write console output to external text file.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        # if fpath is not None:
        #     mkdir_if_missing(os.path.dirname(fpath))
        #     self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def calculate_performance_split(y_test, y_pred_class, y_pred_prob, split_id):
    # confusion matrix
    confusion = metrics.confusion_matrix(y_test, y_pred_class)
    # TP TN FP FN
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    # accuracy
    ACC = (TP + TN) / float(TP + TN + FN + FP)
    # ACC = metrics.accuracy_score(y_test, y_pred_class) another way

    # precision
    PPV = TP / float(TP + FP)
    # PPV = metrics.precision_score(y_test, y_pred_class)

    # TPR, sensitivity, recall
    TPR = TP / float(TP + FN)
    # TPR = metrics.recall_score(y_test, y_pred_class)

    # TNR, specificity
    TNR = TN / float(TN + FP)

    # FPR
    FPR = FP / float(TN + FP)
    # FPR = 1 - TNR

    # F1 score
    F1_score = (2 * PPV * TPR) / (PPV + TPR)
    # F1_score = metrics.f1_score(y_test, y_pred_class)

    # ROC
    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    #print("y_pred_prob: ", y_pred_prob)
    #print("fpr: ", fpr)
    fig = plt.figure(facecolor='white')
    plt.plot(fpr, tpr, 'r')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve')
    plt.xlabel('FPR (False Positive Rate)')
    plt.ylabel('TPR (True Positive Rate)')
    plt.grid(True)
    plt.draw()
    plt.pause(4)
    plt.savefig('ROC_' + str(split_id) + '.png')
    plt.close(fig)

    # AUC
    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    # AUC = metrics.roc_auc_score(y_test, y_pred_prob)
    AUC = metrics.auc(fpr, tpr)
    # # calculate cross-validated AUC
    # from sklearn.cross_validation import cross_val_score
    # mean_socre = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()
    # print(mean_socre)

    # EER
    EER = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)

    # TAR @ FAR = 0.1 / 0.01 / 0.001, FAR = FPR, TAR = TPR
    TAR_FAR_E0_3 = brentq(lambda x: 0.3 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    TAR_FAR_E1 = brentq(lambda x: 0.1 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    TAR_FAR_E2 = brentq(lambda x: 0.01 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    TAR_FAR_E3 = brentq(lambda x: 0.001 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    return ACC, AUC, TAR_FAR_E1, TAR_FAR_E2, TAR_FAR_E3, TAR_FAR_E0_3, fpr, tpr, thresholds


def prob_to_predictlabel(test_probs, labels):
    assert len(test_probs) == len(labels)
    y_pred = []
    y_probs = []
    # print("labels: ", labels)
    # print(test_probs)

    for idx in range(len(test_probs)):
        y_pred.append(0) if test_probs[idx][0] > test_probs[idx][1] else y_pred.append(1)
        y_probs.append(test_probs[idx][1].tolist()) if labels[idx] == 0 else y_probs.append(test_probs[idx][1].tolist())
    assert len(y_probs) == len(y_pred)

    return y_pred, y_probs

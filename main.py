import os
import sys
import argparse
import datetime
import time
import os.path as osp
import matplotlib

import eval_metrics

from matplotlib import pyplot as plt
import numpy as np
from tqdm import *

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import datasets
import models
from centerloss import CenterLoss
import utils
from eval_metrics import evaluate

parser = argparse.ArgumentParser("Center Loss Experiment")

# dataset
parser.add_argument('-d', '--dataset', type=str, default='LR-data', choices=['LR-data'])
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
# optimization
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--lr-model', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.1, help="learning rate for center loss")
parser.add_argument('--weight-cent', type=float, default=1, help="weight for center loss")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")

# model
parser.add_argument('--model', type=str, default='LeNet')

# misc
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--plot', action='store_true', help="whether to plot features for every epoch")

args = parser.parse_args()

def main():
    '''
        : Main Execute Transcript

        :return: None
    '''

    # torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    # sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset + '.txt'))

    if use_gpu:
        print("Using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Using CPU")

    # Load data.
    print("Creating dataset: {}".format(args.dataset))
    dataset = datasets.create(
        name=args.dataset, batch_size=args.batch_size, use_gpu=use_gpu,
        num_workers=args.workers,
    )

    trainloader, valloader, testloader = dataset.trainloader, dataset.valloader, dataset.testloader

    # Creat model.
    print("Creating model: {}".format(args.model))
    #model = models.create(name=args.model, num_classes=dataset.num_classes)
    from net import resnet18, resnet152
    model = resnet152(num_classes=dataset.num_classes)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    # Creat loss.
    criterion_xent = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=dataset.num_classes, feat_dim=2, use_gpu=use_gpu)
    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
    optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)

    # Adjust learning rate.
    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)

    test_acc = []

    for epoch in tqdm(range(args.max_epoch), colour='red'):
        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        train(model, criterion_xent, criterion_cent,
              optimizer_model, optimizer_centloss,
              trainloader, use_gpu, dataset.num_classes, epoch)

        if args.stepsize > 0: scheduler.step()

        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")
            # If validate is True
            # accVal, err = validate(model, valloader, num_classes=dataset.num_classes, epoch=epoch)
            acc = test(model, testloader)
            test_acc.append(acc)
            # acc, err = validate(model, valloader, dataset.num_classes, epoch)
            # MARK
            # print("Val Accuracy (%): {}\t ".format(acc))
            print("Accuracy (%): {}\t ".format(max(test_acc)))


def train(model, criterion_xent, criterion_cent,
          optimizer_model, optimizer_centloss,
          trainloader, use_gpu, num_classes, epoch):
    '''
        : Train Transcript

        :return: None
    '''

    model.train()
    xent_losses = AverageMeter()
    cent_losses = AverageMeter()
    losses = AverageMeter()

    # pbar = tqdm(enumerate(trainloader))

    if args.plot:
        all_features, all_labels = [], []

    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        features, outputs = model(data)
        loss_xent = criterion_xent(outputs, labels)
        loss_cent = criterion_cent(features, labels)
        loss_cent *= args.weight_cent
        loss = loss_xent + loss_cent
        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward()
        optimizer_model.step()

        # by doing so, weight_cent would not impact on the learning of centers
        for param in criterion_cent.parameters():
            param.grad.data *= (1. / args.weight_cent)
        optimizer_centloss.step()

        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))
        cent_losses.update(loss_cent.item(), labels.size(0))

        if args.plot:
            if use_gpu:
                all_features.append(features.data.cpu().numpy())
                all_labels.append(labels.data.cpu().numpy())
            else:
                all_features.append(features.data.numpy())
                all_labels.append(labels.data.numpy())

        if (batch_idx + 1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) XentLoss {:.6f} ({:.6f}) CenterLoss {:.6f} ({:.6f})" \
                  .format(batch_idx + 1, len(trainloader), losses.val, losses.avg, xent_losses.val, xent_losses.avg,
                          cent_losses.val, cent_losses.avg))

    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, all_labels, num_classes, epoch, prefix='train')


def validate(model, valloader, num_classes, epoch):
    '''
        : Validate Transcript

        :return: None
    '''

    model.eval()
    correct, total = 0, 0
    if args.plot:
        all_features, all_labels = [], []

    with torch.no_grad():
        for data, labels in valloader:
            data, labels = data.cuda(), labels.cuda()
            features, outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

            if args.plot:
                if use_gpu:
                    all_features.append(features.data.cpu().numpy())
                    all_labels.append(labels.data.cpu().numpy())
                else:
                    all_features.append(features.data.numpy())
                    all_labels.append(labels.data.numpy())

    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, all_labels, num_classes, epoch, prefix='test')

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err


def test(model, test_loader):
    '''
        : Test Transcript

        :parameter model:
        :parameter test_loader:

        :return: None
    '''

    model.eval()

    l2_dist = utils.PairwiseDistance(2)

    labels, distances = [], []

    for batch_idx, data in enumerate(test_loader):
        # if args.cuda:
        #     data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = data[0].cuda(), data[1].cuda(), data[2]

        # data_a, data_p, label = Variable(data_a), Variable(data_p), Variable(label)

        out_a, _ = model(data_a.float())
        # print("out_a: ", out_a)
        out_p, _ = model(data_p.float())
        # print("out_p: ", out_p)

        dists = l2_dist.forward(out_a, out_p)  # torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance

        distances.append(dists.data.cpu().numpy())

        labels.append(label.data.cpu().numpy())

        # if batch_idx % args.log_interval == 0:
        #     pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
        #         epoch, batch_idx * len(data_a), len(test_loader.dataset),
        #                100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])

    distances = np.array([subdist for dist in distances for subdist in dist])

    tpr, fpr, accuracy, best_threshold = evaluate(distances, labels)
    print('\n\33[91mTest set: Accuracy: {:.8f} best_threshold: {:.2f}\33[0m'.format(accuracy, best_threshold))
    # logger.log_value('Test Accuracy', np.mean(accuracy))
    # plot_roc(fpr, tpr, args.log_dir, figure_name="roc_test_epoch_{}.png".format(epoch))

    # print("accuracy: ", accuracy)

    return accuracy


def plot_features(features, labels, num_classes, epoch, prefix):
    """Plot features on 2D plane.

    Args:
        features: (num_instances, num_features).
        labels: (num_instances). 
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels == label_idx, 0],
            features[labels == label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    dirname = osp.join(args.save_dir, prefix)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, 'epoch_' + str(epoch + 1) + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()

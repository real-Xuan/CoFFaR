import os
import cv2
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch


class CustomDataset(Dataset):
    '''
        : Read Image using OpenCV.

        :return: Return img(20,40), y_label
    '''

    def __init__(self, resize_width=20, resize_height=20, repeat=None):

        '''
        : 自定义数据集

        :parameter image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :parameter resize_height 为None时，不进行缩放
        :parameter resize_width  为None时，不进行缩放，
                              PS：当参数resize_height或resize_width其中一个为None时，可实现等比例缩放
        :parameter repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize

        :return
        '''

        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.count = 0
        self.dataRoot = "/home/xuan/xuan/LR_Experiment/CenterLossExprimentsForLR/TestDataList/verification_images/"
        self.imglist, self.label = self.readCsvData()
        self.iterLen = len(self.label)
        self.idx = 0

    def __getitem__(self, idx):
        pairImg = self.imglist[idx]
        x1, x2 = self.generator(pairImg)
        return x1, x2, self.label[idx]

    def __len__(self):
        if self.repeat == None:
            data_len = self.iterLen
        else:
            data_len = self.iterLen * self.repeat

        return data_len

    def readCsvData(self):
        '''
            : Read Image list from CSV file and creat label list.

            :parameter path_1: The path of first image.
            :parameter path_2: The path of second image.

            :return: Return img(20,40), y_label
        '''

        plist = []
        nlist = []
        for line in open(
                '/home/xuan/xuan/LR_Experiment/CenterLossExprimentsForLR/TestDataList/positive_pairs_names.csv', 'r',
                encoding='utf-8-sig'):
            plist.append(line.strip())
        groupNumP = int(len(plist) / 2)
        plabel = [0] * groupNumP  # label
        plist = np.array(plist)
        plist.reshape((groupNumP, 2))

        for line in open(
                '/home/xuan/xuan/LR_Experiment/CenterLossExprimentsForLR/TestDataList/negative_pairs_names.csv', 'r',
                encoding='utf-8-sig'):
            nlist.append(line.strip())
        groupNumN = int(len(nlist) / 2)
        nlabel = [1] * groupNumN  # label
        nlist = np.array(nlist)
        nlist.reshape((groupNumN, 2))
        pnlist = np.concatenate((plist, nlist), axis=0)

        return pnlist, plabel + nlabel

    def generator(self, pairImg):
        '''
            : Read Image using OpenCV.

            :parameter path_1: The path of first image.
            :parameter path_2: The path of second image.

            :return: Return img(20,40), y_label
        '''

        '''
            ".split()"
            Split by ","
            '9713_cam2_58.jpg','9713_cam3_86.jpg' -> ["'9713_cam2_58.jpg'", "'9713_cam3_86.jpg'"]
            return 返回分割后的字符串列表
            pairImg = str(pairImg)
        
        '''
        pairImg = pairImg.split(",")

        img1 = pairImg[0].strip(
            "'")  # Remove the "'", pairImg[0]: ["'9713_cam2_58.jpg'", "'9713_cam3_86.jpg'"] -> 9713_cam2_58.jpg

        img1 = self.dataRoot + img1  # Got the compelet directory

        img2 = pairImg[1].strip("'")
        img2 = self.dataRoot + img2  # Got the compelet directory

        x1, x2 = self.readImage(img1, img2)

        # x1 = x1 * 1. / 255
        # x2 = x2 * 1. / 255

        x1 = transforms.ToTensor()(x1)
        x2 = transforms.ToTensor()(x2)

        x1 = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(x1)
        x2 = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(x2)

        # y_label = self.label
        # y_label = np.array(self.label)
        # y_label = torch.tensor(y_label, dtype=torch.int)

        return x1, x2

    def readImage(self, path_1, path_2):
        '''
            : Read Image using OpenCV.

            :parameter path_1: The path of first image.
            :parameter path_2: The path of second image.

            :return: Return two (3, 20, 20) images.
        '''

        img1_ = cv2.imread(path_1)
        img2_ = cv2.imread(path_2)

        # Determine whether the image is empty or not.
        if img1_ is None or img2_ is None:
            print(path_1 + '\n', path_2)
            return 1

        img1_ = cv2.cvtColor(img1_, cv2.COLOR_BGR2RGB)  # cv2 load images as BGR, convert it to RGB
        img2_ = cv2.cvtColor(img2_, cv2.COLOR_BGR2RGB)  # cv2 load images as BGR, convert it to RGB

        img1_ = cv2.resize(img1_, (self.resize_width, self.resize_height), interpolation=cv2.INTER_AREA)
        img2_ = cv2.resize(img2_, (self.resize_width, self.resize_height), interpolation=cv2.INTER_AREA)

        return img1_, img2_

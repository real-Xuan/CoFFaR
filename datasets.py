import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from customDataset import *


class DATASET(object):
    '''
        : Read Image using OpenCV.

        :return: Return img(20,40), y_label
    '''

    def __init__(self, batch_size, use_gpu, num_workers):
        pin_memory = True if use_gpu else False

        transform = transforms.Compose([
            transforms.Resize([20, 20]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        trainset = ImageFolder(root="/home/xuan/exdisk/LR-Experiment-Data/QMUL-SurvFace/training_set/",
                               transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        valset = ImageFolder(root="/home/xuan/exdisk/LR-Experiment-Data/QMUL-SurvFace/training_set/",
                             transform=transform)
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        test_data = CustomDataset()
        testloader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.num_classes = len(trainset.class_to_idx)


__factory = {
    'LR-data': DATASET,
}


def create(name, batch_size, use_gpu, num_workers):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](batch_size, use_gpu, num_workers)

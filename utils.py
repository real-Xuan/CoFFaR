import os
import sys
import errno
import shutil

import torch


class PairwiseDistance(torch.nn.Module):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = torch.tensor(p, device='cuda', dtype=torch.float)

    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        out = torch.sqrt(out)
        # print(out)
        # return out
        return torch.pow(out + eps, 1. / self.norm)


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.path.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
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


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(os.path.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, os.path.join(os.path.dirname(fpath), 'best_model.pth.tar'))

    """
class Logger(object):
    
    #Write console output to external text file.
    
    #Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

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
            
    """

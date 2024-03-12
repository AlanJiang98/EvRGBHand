"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Basic logger. It Computes and stores the average and current value
"""
import datetime
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist
import torch.nn.functional as F

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=100, fmt="{median:.4f} ({global_avg:.4f})"):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )
    

def reduce_across_processes(val):
    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t

class AverageMeter(object):
    
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



class EvalMetricsLogger(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        # define a upper-bound performance (worst case) 
        # numbers are in unit millimeter
        self.PAmPJPE = 100.0/1000.0
        self.mPJPE = 100.0/1000.0
        self.mPVE = 100.0/1000.0

        self.epoch = 0

    def update(self, mPVE, mPJPE, PAmPJPE, epoch):
        self.PAmPJPE = PAmPJPE
        self.mPJPE = mPJPE
        self.mPVE = mPVE
        self.epoch = epoch

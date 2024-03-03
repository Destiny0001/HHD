import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, initial_lr):
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.initial_lr * (self.last_epoch + 1) / self.warmup_epochs
        else:
            lr = self.initial_lr
        return [lr for _ in self.optimizer.param_groups]

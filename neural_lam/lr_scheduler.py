# Standard library
import math

# Third-party
import matplotlib.pyplot as plt
import torch


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer,
        total_steps,
        warmup_steps=1000,
        max_lr=0.001,
        min_lr=0.00001,
    ):
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        schedule = MattsSchedule(
            total_steps=100, warmup_steps=55, min_lr=0, max_lr=1
        )
        super().__init__(optimizer)

    def get_lr(self):
        self.base_lrs
        lrs = [1 for group in self.optimizer.param_groups]
        return lrs

    def warmup(self, step):
        return step / self.warmup_steps * self.max

    def cosine_annealing(self, step):
        if step > self.max_steps:
            return self.min

        return self.min + 0.5 * (self.max_lr - self.min_lr) * (
            1 + math.cos(math.pi * step / self.max_steps)
        )


class MattsSchedule:
    def __init__(
        self,
        total_steps,
        warmup_steps=1000,
        min_lr=0.00001,
        max_lr=0.001,
    ):
        self.max_steps = total_steps
        self.warmup_steps = warmup_steps
        self.annealing_steps = total_steps - warmup_steps

        self.max_lr = max_lr
        self.min_lr = min_lr

    def warmup(self, step):
        return step / self.warmup_steps * self.max_lr

    def cosine_annealing(self, step):
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + math.cos(math.pi * step / self.annealing_steps)
        )

    def calculate_lr(self, step):
        if step < self.warmup_steps:
            lr = self.warmup(step)
        elif step < self.max_steps:
            lr = self.cosine_annealing(step - self.warmup_steps)
        else:
            lr = self.min_lr
        return lr

    def get_lr(self, step):
        __import__("pdb").set_trace()  # TODO delme kj:w

        return [self.calculate_lr(step) for _ in optimizer.param_groups]

    def __len__(self):
        return self.max_steps

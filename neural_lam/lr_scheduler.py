# Third-party
import torch


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps=1000,
        annealing_steps=100000,
        max_factor=1.0,
        min_factor=0.001,
    ):
        self.warmup_steps = warmup_steps
        self.annealing_steps = annealing_steps
        initial_learning_rate = optimizer.param_groups[0]["lr"]

        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=min_factor,
            end_factor=max_factor,
            total_iters=warmup_steps,
        )

        self.annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=annealing_steps,
            eta_min=min_factor * initial_learning_rate,
        )

        super().__init__(optimizer)

    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            return self.warmup_scheduler.get_last_lr()
        elif self._step_count <= self.warmup_steps + self.annealing_steps:
            return self.annealing_scheduler.get_last_lr()

        return True

    def step(self):
        if self._step_count == 0:
            pass
        elif self._step_count <= self.warmup_steps:
            self.warmup_scheduler.step()
        elif self._step_count <= self.warmup_steps + self.annealing_steps:
            self.annealing_scheduler.step()
        self._step_count += 1

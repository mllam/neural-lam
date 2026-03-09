# Third-party
import pytorch_lightning as pl
import torch


class EMACallback(pl.Callback):
    """Exponential Moving Average (EMA) callback for model weights.

    Maintains a shadow copy of model parameters as a running average:
        θ_ema ← decay * θ_ema + (1 - decay) * θ_current

    EMA weights are swapped in for validation/test/inference, while training
    continues on the raw optimizer-updated weights. This reduces per-step
    noise compounding during autoregressive rollouts and produces more stable
    checkpoints.

    Parameters
    ----------
    decay : float
        EMA decay factor in [0, 1). Higher values (e.g. 0.999) give a
        slower-moving average. Typical values: 0.999 or 0.9999.
    """

    def __init__(self, decay: float = 0.999):
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.ema_weights: list[torch.Tensor] = []
        self.original_weights: list[torch.Tensor] = []

    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Clone current model parameters as the initial EMA weights."""
        if not self.ema_weights:
            # Only initialize if not already loaded from checkpoint
            self.ema_weights = [p.data.clone() for p in pl_module.parameters()]

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        """Update EMA weights using in-place lerp for efficiency.

        torch.lerp_(end, weight) computes:
            self = self + weight * (end - self)
            self = (1 - weight) * self + weight * end

        With weight = (1 - decay):
            ema = decay * ema + (1 - decay) * current
        """
        for ema_w, param in zip(self.ema_weights, pl_module.parameters()):
            ema_w.lerp_(param.data, 1.0 - self.decay)

    def _swap_to_ema(self, pl_module: pl.LightningModule) -> None:
        """Swap model weights to EMA weights for evaluation."""
        self.original_weights = [p.data.clone() for p in pl_module.parameters()]
        for param, ema_w in zip(pl_module.parameters(), self.ema_weights):
            param.data.copy_(ema_w)

    def _swap_to_original(self, pl_module: pl.LightningModule) -> None:
        """Restore original training weights after evaluation."""
        for param, orig_w in zip(pl_module.parameters(), self.original_weights):
            param.data.copy_(orig_w)
        self.original_weights = []

    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Swap in EMA weights before validation."""
        if self.ema_weights:
            self._swap_to_ema(pl_module)

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Restore original weights after validation."""
        if self.original_weights:
            self._swap_to_original(pl_module)

    def on_test_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Swap in EMA weights before testing."""
        if self.ema_weights:
            self._swap_to_ema(pl_module)

    def on_test_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Restore original weights after testing."""
        if self.original_weights:
            self._swap_to_original(pl_module)

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict,
    ) -> None:
        """Persist EMA weights in the checkpoint."""
        checkpoint["ema_weights"] = [w.cpu() for w in self.ema_weights]

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict,
    ) -> None:
        """Restore EMA weights from a checkpoint.

        Handles device transfer so that EMA weights are placed on the same
        device as the model parameters.
        """
        if "ema_weights" in checkpoint:
            device = next(pl_module.parameters()).device
            self.ema_weights = [w.to(device) for w in checkpoint["ema_weights"]]

"""PyTorch Lightning callbacks used during training and evaluation."""

# Standard library
import warnings
from typing import Any

# Third-party
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT


class EMACallback(pl.Callback):
    """
    Keep an exponential moving average (EMA) of the model weights.

    A shadow copy of every model parameter is maintained as the running
    average ``ema <- decay * ema + (1 - decay) * param``, updated after each
    training batch. The averaged weights are swapped into the model for
    validation and testing and swapped back out again afterwards, so that
    training always continues from the raw optimizer-updated weights.
    Averaging damps the per-step parameter noise that would otherwise
    compound along an autoregressive rollout.

    Only parameters are averaged. Buffers, such as the standardization
    statistics and the boundary mask, are constants rather than learned
    quantities and are left untouched.

    The shadow weights are persisted under ``"callbacks"`` in the checkpoint,
    so a checkpoint holds the raw weights in its ``"state_dict"`` and the
    averaged weights alongside it. Resuming therefore restores both, while
    the validation loss that ``ModelCheckpoint`` monitors is the one measured
    on the averaged weights. Evaluating the averaged weights of a finished
    run requires passing the same ``--ema_decay`` flag as during training, so
    that this callback is present to swap them in.

    Parameters
    ----------
    decay : float
        Decay factor of the running average, in the interval [0, 1). Larger
        values average over a longer window; 0.999 is a common choice.
    """

    def __init__(self, decay: float) -> None:
        """
        Initialize the callback.

        Parameters
        ----------
        decay : float
            Decay factor of the running average, in the interval [0, 1).

        Raises
        ------
        ValueError
            If `decay` lies outside [0, 1).
        """
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError(
                f"EMA decay must be in the interval [0, 1), got {decay}"
            )
        self.decay = decay
        self.ema_state: dict[str, torch.Tensor] = {}
        self._raw_state: dict[str, torch.Tensor] = {}

    def _shadow_for(self, name: str, param: torch.Tensor) -> torch.Tensor:
        """
        Return the shadow weights of a parameter, on that parameter's device.

        Device placement is resolved here rather than when the state is
        restored. Lightning restores callback state before the strategy has
        moved the module onto its accelerator, so the device the model will
        end up on is not yet known at that point. Resolving it per parameter
        also keeps the shadow weights spread the same way as the model itself
        when parameters do not all share one device, as under DDP or model
        parallelism.

        Parameters
        ----------
        name : str
            Name of the parameter, as returned by `named_parameters`.
        param : torch.Tensor
            The parameter itself, whose device the shadow weights follow.

        Returns
        -------
        torch.Tensor
            Shadow weights for `name`, on the same device as `param`.

        Raises
        ------
        ValueError
            If no shadow weights are held for `name`.
        """
        shadow = self.ema_state.get(name)
        if shadow is None:
            raise ValueError(
                f"No EMA weights for parameter '{name}'. The EMA state "
                "restored from the checkpoint does not match the current "
                "model."
            )
        if shadow.device != param.device:
            shadow = shadow.to(device=param.device)
            self.ema_state[name] = shadow
        return shadow

    def _swap_in(self, pl_module: pl.LightningModule) -> None:
        """
        Park the raw weights and copy the averaged weights into the model.

        Parameters
        ----------
        pl_module : pl.LightningModule
            Module whose parameters are replaced by the averaged weights.
        """
        self._raw_state = {}
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                shadow = self._shadow_for(name, param)
                self._raw_state[name] = param.detach().clone()
                param.copy_(shadow)

    def _swap_out(self, pl_module: pl.LightningModule) -> None:
        """
        Copy the parked raw weights back into the model.

        Parameters
        ----------
        pl_module : pl.LightningModule
            Module whose parameters are restored to their raw values.
        """
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                param.copy_(self._raw_state[name])
        self._raw_state = {}

    def _start_eval(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """
        Swap the averaged weights in for an evaluation loop.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer running the evaluation loop.
        pl_module : pl.LightningModule
            Module to evaluate.
        stage : str
            Name of the loop, used in the warning about missing weights.
        """
        if not self.ema_state:
            # Sanity checking runs before the shadow weights are seeded,
            # which is expected. Anywhere else this means the averaged
            # weights the user asked to evaluate are simply not there.
            if not trainer.sanity_checking:
                warnings.warn(
                    f"No EMA weights available, running {stage} on the raw "
                    "model weights instead. Load a checkpoint written with "
                    "EMA enabled to evaluate the averaged weights.",
                    UserWarning,
                    stacklevel=2,
                )
            return
        self._swap_in(pl_module)

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Seed the shadow weights from the model parameters.

        Runs after every checkpoint-restore path, including the late restore
        used by sharded strategies, so restored shadow weights are kept.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer about to start training.
        pl_module : pl.LightningModule
            Module whose parameters seed the average.
        """
        if not self.ema_state:
            self.ema_state = {
                name: param.detach().clone()
                for name, param in pl_module.named_parameters()
            }

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        Fold the current parameters into the running average.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer running the training loop.
        pl_module : pl.LightningModule
            Module whose parameters are folded into the average.
        outputs : STEP_OUTPUT
            Output of the training step, unused.
        batch : Any
            The batch that was just processed, unused.
        batch_idx : int
            Index of the batch that was just processed, unused.
        """
        with torch.no_grad():
            for name, param in pl_module.named_parameters():
                self._shadow_for(name, param).lerp_(
                    param.detach(), 1.0 - self.decay
                )

    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Swap the averaged weights in before validation.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer about to start validation.
        pl_module : pl.LightningModule
            Module to validate.
        """
        self._start_eval(trainer, pl_module, "validation")

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Swap the raw weights back in after validation.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer that has finished validation.
        pl_module : pl.LightningModule
            Module that was validated.
        """
        if self._raw_state:
            self._swap_out(pl_module)

    def on_test_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Swap the averaged weights in before testing.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer about to start testing.
        pl_module : pl.LightningModule
            Module to test.
        """
        self._start_eval(trainer, pl_module, "testing")

    def on_test_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Swap the raw weights back in after testing.

        Parameters
        ----------
        trainer : pl.Trainer
            The trainer that has finished testing.
        pl_module : pl.LightningModule
            Module that was tested.
        """
        if self._raw_state:
            self._swap_out(pl_module)

    def state_dict(self) -> dict[str, Any]:
        """
        Return the averaged weights for storage in a checkpoint.

        The shadow weights are moved to the CPU so that a checkpoint written
        on one accelerator can be restored on another.

        Returns
        -------
        dict of str to Any
            The shadow weights, keyed by parameter name.
        """
        return {
            "ema_state": {
                name: shadow.detach().cpu()
                for name, shadow in self.ema_state.items()
            }
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Restore the averaged weights from a checkpoint.

        The shadow weights are left on whichever device they were loaded
        onto; see `_shadow_for` for why placement is deferred.

        Parameters
        ----------
        state_dict : dict of str to Any
            State previously produced by `state_dict`.
        """
        self.ema_state = dict(state_dict["ema_state"])

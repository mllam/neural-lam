# Third-party
import torch

# Local
from .ar_forecaster import ARForecaster
from .step_predictor import StepPredictor


class ARForecastSampler(ARForecaster):
    """
    Ensemble sampler that perturbs initial states and delegates AR unroll.

    This class creates stochastic ensemble members by adding Gaussian noise
    to the expanded initial states and then calls ARForecaster on the expanded
    batch as if it were a deterministic batch.
    """

    def __init__(
        self,
        step_predictor: StepPredictor,
        args,
        noise_std: float = 0.05,
    ):
        """
        Parameters
        ----------
        step_predictor : StepPredictor
            One-step predictor used by the AR unroll.
        args : Any
            Runtime args/config namespace.
        noise_std : float
            Standard deviation of Gaussian perturbations added to
            `init_states` after ensemble expansion.
        """
        super().__init__(step_predictor, args)
        self.noise_std = float(noise_std)

    def forward(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        true_states: torch.Tensor = None,
        pred_steps: int = None,
        ensemble_size: int = 1,
    ):
        """
        Generate ensemble forecasts by batch expansion and AR unroll.

        Parameters
        ----------
        init_states : torch.Tensor
            Shape (B, 2, N, F), where B=batch, 2=conditioning states,
            N=grid nodes, F=state features.
        forcing_features : torch.Tensor
            Shape (B, T, N, F_forcing), where T=forecast horizon length.
        true_states : torch.Tensor, optional
            Boundary-condition source with shape (B, T, N, F) or
            (B, S, T, N, F) when ensemble-specific boundaries are provided.
        pred_steps : int, optional
            Number of rollout steps; defaults to T from forcing_features.
        ensemble_size : int
            Number of ensemble members S.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            prediction shape (B, S, T, N, F), and pred_std either
            (B, S, T, N, F) when predicted by the model or a constant
            per-variable tensor (F,) for deterministic predictors.
        """
        if ensemble_size < 2:
            return super().forward(
                init_states=init_states,
                forcing_features=forcing_features,
                true_states=true_states,
                pred_steps=pred_steps,
                ensemble_size=ensemble_size,
            )

        batch_size = init_states.shape[0]
        max_steps = forcing_features.shape[1]
        if pred_steps is None:
            pred_steps = max_steps
        if pred_steps > max_steps:
            raise ValueError("pred_steps cannot exceed forcing time dimension")

        # Expand batch to (B*S, ...)
        expanded_init_states = init_states.repeat_interleave(
            ensemble_size, dim=0
        )
        expanded_forcing = forcing_features[:, :pred_steps].repeat_interleave(
            ensemble_size, dim=0
        )

        # Inject Gaussian perturbations to create ensemble diversity.
        noise = torch.randn_like(expanded_init_states) * self.noise_std
        expanded_init_states = expanded_init_states + noise

        # Prepare boundary states to match the expanded batch if provided.
        expanded_true_states = None
        if true_states is not None:
            if hasattr(self, "_prepare_true_states"):
                expanded_true_states = self._prepare_true_states(
                    true_states=true_states,
                    batch_size=batch_size,
                    pred_steps=pred_steps,
                    ensemble_size=ensemble_size,
                )
            else:
                expanded_true_states = true_states.repeat_interleave(
                    ensemble_size, dim=0
                )

        # Trick: run parent forecaster with ensemble_size=1 on (B*S, ...).
        prediction, pred_std = super().forward(
            init_states=expanded_init_states,
            forcing_features=expanded_forcing,
            true_states=expanded_true_states,
            pred_steps=pred_steps,
            ensemble_size=1,
        )  # prediction: (B*S, T, N, F)

        # Reshape back to ensemble representation (B, S, T, N, F).
        prediction = prediction.reshape(
            batch_size, ensemble_size, pred_steps, *prediction.shape[2:]
        )

        if pred_std is not None and pred_std.ndim >= 4:
            pred_std = pred_std.reshape(
                batch_size, ensemble_size, pred_steps, *pred_std.shape[2:]
            )

        return prediction, pred_std

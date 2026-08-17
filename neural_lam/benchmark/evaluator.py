"""Forecast benchmarking and evaluation engine for Limited Area Models."""

# Standard library
import time
from dataclasses import dataclass, field

# Third-party
import numpy as np
import torch
from torch.utils.data import DataLoader

# Local
from .. import metrics
from ..datastore.base import BaseDatastore, BaseRegularGridDatastore
from ..models.module import ForecasterModule
from ..weather_dataset import WeatherDataset


@dataclass
class BenchmarkScorecard:
    """Scorecard holding forecast accuracy and spatial/spectral diagnostics."""

    model_name: str
    num_rollout_steps: int
    variables: list[str]
    lead_time_hours: list[float]
    mse_per_lead_time: dict[str, list[float]]  # Standardized MSE
    rmse_per_lead_time: dict[str, list[float]]  # Physical RMSE
    mbe_per_lead_time: dict[str, list[float]]  # Physical Bias
    radial_psd: dict[str, list[float]]  # 2D DCT-II power spectral density
    wavenumbers: list[float]
    fss_scores: dict[str, dict[int, list[float]]] = field(
        default_factory=dict
    )  # scale -> scores
    spectral_collapse_ratios: dict[str, list[float]] = field(
        default_factory=dict
    )
    hallucination_indices: dict[str, list[float]] = field(default_factory=dict)
    mean_step_latency_ms: float = 0.0
    is_stable: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a formatted string summary of the benchmark scorecard."""
        lines = [
            f"=== Forecast Benchmark Summary: {self.model_name} ===",
            (
                f"Rollout Steps: {self.num_rollout_steps} | "
                f"Stable: {self.is_stable} | "
                f"Avg Step Latency: {self.mean_step_latency_ms:.2f} ms"
            ),
            "",
            "Lead-Time Physical RMSE & Bias:",
        ]
        for var in self.variables:
            rmse_str = ", ".join(
                f"t{i + 1}: {rmse:.3f}"
                for i, rmse in enumerate(self.rmse_per_lead_time.get(var, []))
            )
            bias_str = ", ".join(
                f"t{i + 1}: {bias:+.3f}"
                for i, bias in enumerate(self.mbe_per_lead_time.get(var, []))
            )
            lines.append(f"  [{var}]")
            lines.append(f"    RMSE -> {rmse_str}")
            lines.append(f"    Bias -> {bias_str}")

            if var in self.spectral_collapse_ratios:
                scr_val = np.mean(self.spectral_collapse_ratios[var])
                lines.append(
                    f"    Mean Spectral Collapse Ratio (SCR) -> {scr_val:.2f}"
                )

            if var in self.hallucination_indices:
                hi_val = np.mean(self.hallucination_indices[var])
                lines.append(
                    f"    Mean Hallucination Index (HI) -> {hi_val:.2f}"
                )

        return "\n".join(lines)


class ForecastBenchmark:
    """
    Standard evaluation benchmark for weather models on Neural-LAM datastores.

    Supports interior domain masking, 2D DCT-II spectral decomposition, and
    scale-dependent Fractions Skill Score (FSS).
    """

    def __init__(
        self,
        datastore: BaseDatastore,
        split: str = "test",
        eval_steps: int = 4,
        batch_size: int = 1,
        buffer_width: int = 10,
        fss_kernel_sizes: list[int] | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Initialize the benchmark evaluator.

        Parameters
        ----------
        datastore : BaseDatastore
            The datastore providing grid metadata, data, and standardization.
        split : str, default "test"
            The dataset split to evaluate on.
        eval_steps : int, default 4
            The number of future autoregressive rollout steps to evaluate.
        batch_size : int, default 1
            Batch size for evaluation.
        buffer_width : int, default 10
            Lateral boundary buffer zone width in grid units.
        fss_kernel_sizes : list of int or None, optional
            Kernel neighborhood sizes for Fractions Skill Score.
            Defaults to [1, 3, 7].
        device : str or torch.device, default "cpu"
            Computation device.
        """
        self.datastore = datastore
        self.split = split
        self.eval_steps = eval_steps
        self.batch_size = batch_size
        self.buffer_width = buffer_width
        self.fss_kernel_sizes = fss_kernel_sizes or [1, 3, 7]
        self.device = torch.device(device)

        self.dataset = WeatherDataset(
            datastore=self.datastore,
            split=self.split,
            ar_steps=self.eval_steps,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Retrieve standardization stats
        da_state_stats = self.datastore.get_standardization_dataarray("state")
        self.state_mean = torch.tensor(
            da_state_stats.state_mean.values,
            dtype=torch.float32,
            device=self.device,
        )
        self.state_std = torch.tensor(
            da_state_stats.state_std.values,
            dtype=torch.float32,
            device=self.device,
        )
        self.var_names = list(self.datastore.get_vars_names("state"))
        self.step_length_hours = float(
            self.datastore.step_length.total_seconds() / 3600.0
        )

        # Build interior evaluation domain mask
        self.interior_mask_2d: torch.Tensor | None = None
        if isinstance(self.datastore, BaseRegularGridDatastore):
            ny, nx = (
                self.datastore.grid_shape_state.y,
                self.datastore.grid_shape_state.x,
            )
            mask = torch.ones((ny, nx), device=self.device)
            bw = min(self.buffer_width, ny // 4, nx // 4)
            if bw > 0:
                mask[:bw, :] = 0.0
                mask[-bw:, :] = 0.0
                mask[:, :bw] = 0.0
                mask[:, -bw:] = 0.0
            self.interior_mask_2d = mask

    def evaluate(
        self,
        model: ForecasterModule,
        max_batches: int | None = 2,
    ) -> BenchmarkScorecard:
        """
        Execute the evaluation benchmark on the provided model.

        Parameters
        ----------
        model : ForecasterModule
            The trained model to evaluate.
        max_batches : int or None, optional
            Maximum number of batches to evaluate. Defaults to 2 for speed.

        Returns
        -------
        BenchmarkScorecard
            The computed metrics, FSS, and 2D DCT-II spectral scorecard.
        """
        model = model.to(self.device)
        model.eval()

        var_mse_list: dict[str, list[list[float]]] = {
            v: [] for v in self.var_names
        }
        var_rmse_list: dict[str, list[list[float]]] = {
            v: [] for v in self.var_names
        }
        var_bias_list: dict[str, list[list[float]]] = {
            v: [] for v in self.var_names
        }
        psd_model_list: dict[str, list[torch.Tensor]] = {
            v: [] for v in self.var_names
        }
        psd_targ_list: dict[str, list[torch.Tensor]] = {
            v: [] for v in self.var_names
        }
        fss_map: dict[str, dict[int, list[list[float]]]] = {
            v: {k: [] for k in self.fss_kernel_sizes} for v in self.var_names
        }
        step_latencies_ms: list[float] = []
        is_stable = True
        k_bins_out: list[float] = []

        with torch.no_grad():
            for b_idx, batch in enumerate(self.dataloader):
                if max_batches is not None and b_idx >= max_batches:
                    break

                batch = model.on_after_batch_transfer(batch, dataloader_idx=0)
                init_states, target_states, forcing, _ = batch

                t0 = time.perf_counter()
                prediction, target_states, _, _ = model.common_step(batch)
                t1 = time.perf_counter()
                step_latencies_ms.append(
                    ((t1 - t0) * 1000.0) / max(1, self.eval_steps)
                )

                if (
                    torch.isnan(prediction).any()
                    or torch.isinf(prediction).any()
                ):
                    is_stable = False
                    break

                # Denormalize predictions and targets to physical space
                pred_physical = prediction * self.state_std + self.state_mean
                target_physical = (
                    target_states * self.state_std + self.state_mean
                )

                # Compute per-lead-time errors for each variable
                for v_i, var in enumerate(self.var_names):
                    v_pred_std = prediction[..., v_i]
                    v_targ_std = target_states[..., v_i]
                    v_pred_phys = pred_physical[..., v_i]
                    v_targ_phys = target_physical[..., v_i]

                    # Per step metrics: shape (B, T, N) -> average over B and N
                    mse_steps = (
                        ((v_pred_std - v_targ_std) ** 2)
                        .mean(dim=(0, 2))
                        .cpu()
                        .tolist()
                    )
                    rmse_steps = (
                        torch.sqrt(
                            ((v_pred_phys - v_targ_phys) ** 2).mean(dim=(0, 2))
                        )
                        .cpu()
                        .tolist()
                    )
                    bias_steps = (
                        (v_pred_phys - v_targ_phys)
                        .mean(dim=(0, 2))
                        .cpu()
                        .tolist()
                    )

                    var_mse_list[var].append(mse_steps)
                    var_rmse_list[var].append(rmse_steps)
                    var_bias_list[var].append(bias_steps)

                    # Compute 2D DCT-II power spectral density and FSS
                    # if regular grid
                    if (
                        isinstance(self.datastore, BaseRegularGridDatastore)
                        and self.interior_mask_2d is not None
                    ):
                        ny = self.datastore.grid_shape_state.y
                        nx = self.datastore.grid_shape_state.x
                        field_pred_2d = v_pred_phys.reshape(-1, ny, nx)
                        field_targ_2d = v_targ_phys.reshape(-1, ny, nx)

                        k_centers, rad_psd_pred = metrics.dct_power_spectrum_2d(
                            field_pred_2d, mask=self.interior_mask_2d
                        )
                        _, rad_psd_targ = metrics.dct_power_spectrum_2d(
                            field_targ_2d, mask=self.interior_mask_2d
                        )

                        psd_model_list[var].append(
                            rad_psd_pred.mean(dim=0).cpu()
                        )
                        psd_targ_list[var].append(
                            rad_psd_targ.mean(dim=0).cpu()
                        )
                        if not k_bins_out:
                            k_bins_out = k_centers.cpu().tolist()

                        # Compute FSS at 75th percentile threshold
                        thresh = float(
                            torch.quantile(field_targ_2d, 0.75).item()
                        )
                        for k_sz in self.fss_kernel_sizes:
                            fss_val = metrics.fractions_skill_score_2d(
                                field_pred_2d,
                                field_targ_2d,
                                threshold=thresh,
                                kernel_size=k_sz,
                                mask=self.interior_mask_2d,
                            )
                            fss_map[var][k_sz].append([float(fss_val.item())])

        # Aggregate across batches
        avg_mse = {
            v: (
                np.mean(var_mse_list[v], axis=0).tolist()
                if var_mse_list[v]
                else []
            )
            for v in self.var_names
        }
        avg_rmse = {
            v: (
                np.mean(var_rmse_list[v], axis=0).tolist()
                if var_rmse_list[v]
                else []
            )
            for v in self.var_names
        }
        avg_bias = {
            v: (
                np.mean(var_bias_list[v], axis=0).tolist()
                if var_bias_list[v]
                else []
            )
            for v in self.var_names
        }
        avg_psd = {
            v: (
                torch.stack(psd_model_list[v]).mean(dim=0).tolist()
                if psd_model_list[v]
                else []
            )
            for v in self.var_names
        }
        scr_map: dict[str, list[float]] = {}
        hi_map: dict[str, list[float]] = {}
        fss_avg: dict[str, dict[int, list[float]]] = {}

        for var in self.var_names:
            if psd_model_list[var] and psd_targ_list[var]:
                m_psd = torch.stack(psd_model_list[var]).mean(dim=0)
                t_psd = torch.stack(psd_targ_list[var]).mean(dim=0)
                scr_tensor = metrics.spectral_collapse_ratio(m_psd, t_psd)
                scr_map[var] = scr_tensor.tolist()

                # Fine-scale SCR (last quarter of spectrum)
                scr_fine = float(scr_tensor[-len(scr_tensor) // 4 :].mean())
                fss_fine = (
                    float(np.mean(fss_map[var][self.fss_kernel_sizes[0]]))
                    if fss_map[var][self.fss_kernel_sizes[0]]
                    else 0.5
                )
                hi_val = float(metrics.hallucination_index(scr_fine, fss_fine))
                hi_map[var] = [hi_val]

            fss_avg[var] = {
                k_sz: [float(np.mean(fss_map[var][k_sz]))]
                for k_sz in self.fss_kernel_sizes
                if fss_map[var][k_sz]
            }

        lead_hours = [
            (t + 1) * self.step_length_hours for t in range(self.eval_steps)
        ]
        mean_latency = (
            float(np.mean(step_latencies_ms)) if step_latencies_ms else 0.0
        )

        return BenchmarkScorecard(
            model_name=model.forecaster.__class__.__name__,
            num_rollout_steps=self.eval_steps,
            variables=self.var_names,
            lead_time_hours=lead_hours,
            mse_per_lead_time=avg_mse,
            rmse_per_lead_time=avg_rmse,
            mbe_per_lead_time=avg_bias,
            radial_psd=avg_psd,
            wavenumbers=k_bins_out,
            fss_scores=fss_avg,
            spectral_collapse_ratios=scr_map,
            hallucination_indices=hi_map,
            mean_step_latency_ms=mean_latency,
            is_stable=is_stable,
        )

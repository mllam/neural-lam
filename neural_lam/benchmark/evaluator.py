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
from ..datastore.base import BaseDatastore
from ..models.module import ForecasterModule
from ..weather_dataset import WeatherDataset


@dataclass
class BenchmarkScorecard:
    """Scorecard holding forecast accuracy and spatial/spectral diagnostics."""

    model_name: str
    num_rollout_steps: int
    variables: list[str]
    lead_time_hours: list[float]
    mse_per_lead_time: dict[str, list[float]]
    rmse_per_lead_time: dict[str, list[float]]
    mbe_per_lead_time: dict[str, list[float]]
    stde_per_lead_time: dict[str, list[float]]
    vector_wind_rmse: list[float] | None = None
    radial_psd_model: dict[str, list[float]] = field(default_factory=dict)
    radial_psd_target: dict[str, list[float]] = field(default_factory=dict)
    wavenumbers: list[float] = field(default_factory=list)
    log_spectral_distance: dict[str, float] = field(default_factory=dict)
    fss_scores: dict[str, dict[int, list[float]]] = field(default_factory=dict)
    wind_threshold_metrics: dict[float, dict[str, float]] = field(
        default_factory=dict
    )
    mean_step_latency_ms: float = 0.0
    is_stable: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def evaluate_verdict(
        self,
        reference_scorecard: "BenchmarkScorecard | None" = None,
        mode: str = "operational",
    ) -> tuple[bool, list[str]]:
        """
        Evaluate objective pass/fail or advisory verdict for candidate model.

        Parameters
        ----------
        reference_scorecard : BenchmarkScorecard or None, optional
            Reference model scorecard (e.g. baseline GraphLAM or persistence).
        mode : str, default "operational"
            Verdict mode: "operational" (strict gating) or "research"
            (advisory warnings).

        Returns
        -------
        tuple of (bool, list of str)
            - ``passed``: True if all criteria are met (or if mode="research").
            - ``messages``: List of failure reasons or advisory warnings.
        """
        messages: list[str] = []
        is_pass = True

        if not self.is_stable:
            is_pass = False
            messages.append(
                "FAIL: Model produced NaN or Inf values in rollout."
            )

        if reference_scorecard is not None:
            # Check RMSE skill: candidate must not be > 5% worse than reference
            for var in self.variables:
                if (
                    var in self.rmse_per_lead_time
                    and var in reference_scorecard.rmse_per_lead_time
                ):
                    cand_rmse = self.rmse_per_lead_time[var]
                    ref_rmse = reference_scorecard.rmse_per_lead_time[var]
                    if cand_rmse and ref_rmse:
                        avg_cand = float(np.mean(cand_rmse))
                        avg_ref = float(np.mean(ref_rmse))
                        if avg_ref > 0 and avg_cand > 1.05 * avg_ref:
                            msg = (
                                f"RMSE degraded on {var}: candidate "
                                f"({avg_cand:.3f}) > 1.05 * ref ({avg_ref:.3f})"
                            )
                            if mode == "operational":
                                is_pass = False
                                messages.append(f"FAIL: {msg}")
                            else:
                                messages.append(f"WARNING: {msg}")

            # Check Log-Spectral Distance: candidate must not be > 10% worse
            for var in self.variables:
                if (
                    var in self.log_spectral_distance
                    and var in reference_scorecard.log_spectral_distance
                ):
                    cand_lsd = self.log_spectral_distance[var]
                    ref_lsd = reference_scorecard.log_spectral_distance[var]
                    if ref_lsd > 0 and cand_lsd > 1.10 * ref_lsd:
                        msg = (
                            f"LSD degraded on {var}: candidate "
                            f"{cand_lsd:.2f} dB > 1.10 * ref "
                            f"({ref_lsd:.2f} dB)"
                        )
                        if mode == "operational":
                            is_pass = False
                            messages.append(f"FAIL: {msg}")
                        else:
                            messages.append(f"WARNING: {msg}")

        # Check wind frequency bias bounds: [0.8, 1.25]
        for thresh, metrics_dict in self.wind_threshold_metrics.items():
            if "fbias" in metrics_dict:
                fb = metrics_dict["fbias"]
                if fb < 0.8 or fb > 1.25:
                    msg = (
                        f"Wind Frequency Bias at {thresh} m/s out of bounds: "
                        f"{fb:.2f} not in [0.8, 1.25]"
                    )
                    if mode == "operational":
                        is_pass = False
                        messages.append(f"FAIL: {msg}")
                    else:
                        messages.append(f"WARNING: {msg}")

        if not messages:
            messages.append(
                "PASS: All verification scorecard checks satisfied."
            )

        if mode == "research" and not is_pass:
            is_pass = True

        return is_pass, messages

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

            if var in self.log_spectral_distance:
                lsd_val = self.log_spectral_distance[var]
                lines.append(
                    f"    Log-Spectral Distance (LSD) -> {lsd_val:.2f} dB"
                )

            if var in self.fss_scores and self.fss_scores[var]:
                fss_strs = [
                    f"{scale}px: {np.mean(scores):.2f}"
                    for scale, scores in sorted(self.fss_scores[var].items())
                ]
                lines.append(f"    FSS -> {', '.join(fss_strs)}")

        if self.vector_wind_rmse:
            wind_str = ", ".join(
                f"t{i + 1}: {w:.3f}"
                for i, w in enumerate(self.vector_wind_rmse)
            )
            lines.append(f"  [Vector Wind RMSE] -> {wind_str}")

        return "\n".join(lines)


class ForecastBenchmark:
    """
    Standard evaluation benchmark for weather models on Neural-LAM datastores.

    Supports interior domain masking, 2D DCT-II spectral decomposition,
    Log-Spectral Distance (LSD), and Fractions Skill Score (FSS).
    """

    def __init__(
        self,
        datastore: BaseDatastore,
        split: str = "test",
        eval_steps: int = 4,
        batch_size: int = 1,
        buffer_width: int = 0,
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
        buffer_width : int, default 0
            Additional lateral boundary buffer zone width in grid units.
        fss_kernel_sizes : list of int or None, optional
            Kernel neighborhood sizes for Fractions Skill Score. Must be odd.
            Defaults to [1, 3, 7, 15, 31].
        device : str or torch.device, default "cpu"
            Computation device.
        """
        self.datastore = datastore
        self.split = split
        self.eval_steps = eval_steps
        self.batch_size = batch_size
        self.buffer_width = buffer_width
        self.fss_kernel_sizes = fss_kernel_sizes or [1, 3, 7, 15, 31]
        for k_sz in self.fss_kernel_sizes:
            if k_sz % 2 == 0:
                raise ValueError(f"FSS kernel size must be odd, got {k_sz}.")
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
        self.interior_mask_1d: torch.Tensor | None = None

        if self.datastore.is_on_rectilinear_grid:
            nx = self.datastore.grid_shape_state.x
            ny = self.datastore.grid_shape_state.y

            # 1 in boundary_mask indicates boundary node, 0 indicates interior
            b_mask_arr = self.datastore.boundary_mask.values
            interior_1d_bool = torch.tensor(
                b_mask_arr == 0, device=self.device, dtype=torch.bool
            )
            # Reshape (nx, ny) then transpose to (ny, nx)
            interior_2d_bool = interior_1d_bool.reshape(nx, ny).transpose(0, 1)

            if self.buffer_width > 0:
                bw = min(self.buffer_width, ny // 4, nx // 4)
                interior_2d_bool[:bw, :] = False
                interior_2d_bool[-bw:, :] = False
                interior_2d_bool[:, :bw] = False
                interior_2d_bool[:, -bw:] = False
                # Transpose back to (nx, ny) to flatten into 1D
                interior_1d_bool = interior_2d_bool.transpose(0, 1).flatten()

            self.interior_mask_2d = interior_2d_bool.float()
            self.interior_mask_1d = interior_1d_bool

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
            The computed metrics, FSS, LSD, and spectral scorecard.
        """
        model = model.to(self.device)
        model.eval()

        # Exact dataset-level error accumulation
        sum_sq_error: dict[str, torch.Tensor] = {
            v: torch.zeros(self.eval_steps, device=self.device)
            for v in self.var_names
        }
        sum_error: dict[str, torch.Tensor] = {
            v: torch.zeros(self.eval_steps, device=self.device)
            for v in self.var_names
        }
        count_cells: dict[str, int] = {v: 0 for v in self.var_names}

        # Vector wind evaluation
        has_vector_wind = "u10m" in self.var_names and "v10m" in self.var_names
        sum_sq_wind_error = torch.zeros(self.eval_steps, device=self.device)
        count_wind_cells = 0

        psd_model_list: dict[str, list[torch.Tensor]] = {
            v: [] for v in self.var_names
        }
        psd_targ_list: dict[str, list[torch.Tensor]] = {
            v: [] for v in self.var_names
        }
        fss_map: dict[str, dict[int, list[float]]] = {
            v: {k: [] for k in self.fss_kernel_sizes} for v in self.var_names
        }
        wind_contingency_counts: dict[float, dict[str, float]] = {
            thresh: {"h": 0.0, "f": 0.0, "m": 0.0, "cn": 0.0}
            for thresh in [5.0, 10.0, 15.0]
        }

        step_latencies_ms: list[float] = []
        is_stable = True
        k_bins_out: list[float] = []

        with torch.no_grad():
            for b_idx, batch in enumerate(self.dataloader):
                if max_batches is not None and b_idx >= max_batches:
                    break

                batch = model.on_after_batch_transfer(batch, dataloader_idx=0)

                if torch.cuda.is_available() and self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                prediction, target_states, _, _ = model.common_step(batch)
                if torch.cuda.is_available() and self.device.type == "cuda":
                    torch.cuda.synchronize()
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

                b_size = prediction.shape[0]

                # Accumulate dataset-level error statistics
                for v_i, var in enumerate(self.var_names):
                    v_p_phys = pred_physical[..., v_i]  # (B, T, N)
                    v_t_phys = target_physical[..., v_i]

                    if self.interior_mask_1d is not None:
                        v_p_interior = v_p_phys[:, :, self.interior_mask_1d]
                        v_t_interior = v_t_phys[:, :, self.interior_mask_1d]
                    else:
                        v_p_interior = v_p_phys
                        v_t_interior = v_t_phys

                    diff = v_p_interior - v_t_interior
                    sum_sq_error[var] += (diff**2).sum(dim=(0, 2))
                    sum_error[var] += diff.sum(dim=(0, 2))
                    count_cells[var] += b_size * v_p_interior.shape[2]

                    # 2D DCT-II spectral decomposition and FSS on regular grids
                    if (
                        self.datastore.is_on_rectilinear_grid
                        and self.interior_mask_2d is not None
                    ):
                        nx = self.datastore.grid_shape_state.x
                        ny = self.datastore.grid_shape_state.y

                        # Reshape (..., nx, ny) and transpose to (..., ny, nx)
                        p_2d = v_p_phys.reshape(-1, nx, ny).transpose(-2, -1)
                        t_2d = v_t_phys.reshape(-1, nx, ny).transpose(-2, -1)

                        k_centers, rad_psd_p = metrics.dct_power_spectrum_2d(
                            p_2d, mask=self.interior_mask_2d
                        )
                        _, rad_psd_t = metrics.dct_power_spectrum_2d(
                            t_2d, mask=self.interior_mask_2d
                        )

                        psd_model_list[var].append(rad_psd_p.mean(dim=0).cpu())
                        psd_targ_list[var].append(rad_psd_t.mean(dim=0).cpu())
                        if not k_bins_out:
                            k_bins_out = k_centers.cpu().tolist()

                        # FSS at 90th percentile threshold
                        thresh = float(torch.quantile(t_2d, 0.90).item())
                        for k_sz in self.fss_kernel_sizes:
                            fss_val = metrics.fractions_skill_score_2d(
                                p_2d,
                                t_2d,
                                threshold=thresh,
                                kernel_size=k_sz,
                                mask=self.interior_mask_2d,
                            )
                            fss_map[var][k_sz].append(float(fss_val.item()))

                # Vector wind speed and threshold contingency
                if has_vector_wind:
                    u_idx = self.var_names.index("u10m")
                    v_idx = self.var_names.index("v10m")
                    u_p = pred_physical[..., u_idx]
                    u_t = target_physical[..., u_idx]
                    v_p = pred_physical[..., v_idx]
                    v_t = target_physical[..., v_idx]

                    if self.interior_mask_1d is not None:
                        u_p_int = u_p[:, :, self.interior_mask_1d]
                        u_t_int = u_t[:, :, self.interior_mask_1d]
                        v_p_int = v_p[:, :, self.interior_mask_1d]
                        v_t_int = v_t[:, :, self.interior_mask_1d]
                    else:
                        u_p_int = u_p
                        u_t_int = u_t
                        v_p_int = v_p
                        v_t_int = v_t

                    sq_wind_diff = (u_p_int - u_t_int) ** 2 + (
                        v_p_int - v_t_int
                    ) ** 2
                    sum_sq_wind_error += sq_wind_diff.sum(dim=(0, 2))
                    count_wind_cells += b_size * u_p_int.shape[2]

                    # Wind speed threshold contingency: 5, 10, 15 m/s
                    speed_pred = torch.sqrt(u_p_int**2 + v_p_int**2)
                    speed_targ = torch.sqrt(u_t_int**2 + v_t_int**2)
                    for thresh in [5.0, 10.0, 15.0]:
                        h, f, m, cn = metrics.contingency_table(
                            speed_pred, speed_targ, threshold=thresh
                        )
                        wind_contingency_counts[thresh]["h"] += float(h.item())
                        wind_contingency_counts[thresh]["f"] += float(f.item())
                        wind_contingency_counts[thresh]["m"] += float(m.item())
                        wind_contingency_counts[thresh]["cn"] += float(
                            cn.item()
                        )

        # Compute exact dataset-level physical RMSE, Bias, and STDE
        final_mse: dict[str, list[float]] = {}
        final_rmse: dict[str, list[float]] = {}
        final_mbe: dict[str, list[float]] = {}
        final_stde: dict[str, list[float]] = {}

        for var in self.var_names:
            total_cells = max(1, count_cells[var])
            mse_t = sum_sq_error[var] / total_cells
            rmse_t = torch.sqrt(mse_t)
            mbe_t = sum_error[var] / total_cells
            var_t = torch.clamp(mse_t - mbe_t**2, min=0.0)
            stde_t = torch.sqrt(var_t)

            final_mse[var] = mse_t.cpu().tolist()
            final_rmse[var] = rmse_t.cpu().tolist()
            final_mbe[var] = mbe_t.cpu().tolist()
            final_stde[var] = stde_t.cpu().tolist()

        final_wind_rmse: list[float] | None = None
        if has_vector_wind and count_wind_cells > 0:
            final_wind_rmse = (
                torch.sqrt(sum_sq_wind_error / count_wind_cells).cpu().tolist()
            )

        # Average PSD and Log-Spectral Distance (LSD)
        avg_psd_model: dict[str, list[float]] = {}
        avg_psd_target: dict[str, list[float]] = {}
        lsd_dict: dict[str, float] = {}

        for var in self.var_names:
            if psd_model_list[var] and psd_targ_list[var]:
                m_psd = torch.stack(psd_model_list[var]).mean(dim=0)
                t_psd = torch.stack(psd_targ_list[var]).mean(dim=0)
                avg_psd_model[var] = m_psd.tolist()
                avg_psd_target[var] = t_psd.tolist()

                lsd_tensor = metrics.log_spectral_distance(m_psd, t_psd)
                lsd_dict[var] = float(lsd_tensor.item())

        fss_avg: dict[str, dict[int, list[float]]] = {}
        for var in self.var_names:
            fss_avg[var] = {
                k_sz: [float(np.mean(fss_map[var][k_sz]))]
                for k_sz in self.fss_kernel_sizes
                if fss_map[var][k_sz]
            }

        # Wind threshold metrics (FBIAS, ETS)
        wind_threshold_metrics: dict[float, dict[str, float]] = {}
        if has_vector_wind:
            for thresh, counts in wind_contingency_counts.items():
                h = counts["h"]
                f = counts["f"]
                m = counts["m"]
                cn = counts["cn"]
                fb = (h + f) / (h + m + 1e-8)
                total = h + f + m + cn
                h_rand = ((h + m) * (h + f)) / (total + 1e-8)
                ets = (h - h_rand) / (h + m + f - h_rand + 1e-8)
                wind_threshold_metrics[thresh] = {"fbias": fb, "ets": ets}

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
            mse_per_lead_time=final_mse,
            rmse_per_lead_time=final_rmse,
            mbe_per_lead_time=final_mbe,
            stde_per_lead_time=final_stde,
            vector_wind_rmse=final_wind_rmse,
            radial_psd_model=avg_psd_model,
            radial_psd_target=avg_psd_target,
            wavenumbers=k_bins_out,
            log_spectral_distance=lsd_dict,
            fss_scores=fss_avg,
            wind_threshold_metrics=wind_threshold_metrics,
            mean_step_latency_ms=mean_latency,
            is_stable=is_stable,
        )

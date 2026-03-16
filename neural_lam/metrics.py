# Third-party
import torch


class BaseMetric:
    """
    Base class for all metrics. Each metric is a callable object that also
    knows how to aggregate, post-process, and rescale its values for logging.

    This follows the principle that each metric should carry its own logging
    semantics, rather than relying on hardcoded rules in the aggregation code.

    Subclasses must implement `compute_entry_vals` which returns the per-entry
    metric values before mask-and-reduce.
    """

    # The canonical name of the metric (used for registration/lookup)
    name: str = ""

    def __call__(
        self,
        pred,
        target,
        pred_std,
        mask=None,
        average_grid=True,
        sum_vars=True,
    ):
        """
        Compute the metric value.

        (...,) is any number of batch dimensions, potentially different
            but broadcastable
        pred: (..., N, d_state), prediction
        target: (..., N, d_state), target
        pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
        mask: (N,), boolean mask describing which grid nodes to use in metric
        average_grid: boolean, if grid dimension -2 should be reduced
            (mean over N)
        sum_vars: boolean, if variable dimension -1 should be reduced
            (sum over d_state)

        Returns:
        metric_val: One of (...,), (..., d_state), (..., N),
            (..., N, d_state), depending on reduction arguments.
        """
        entry_vals = self.compute_entry_vals(
            pred, target, pred_std
        )  # (..., N, d_state)
        return mask_and_reduce_metric(
            entry_vals,
            mask=mask,
            average_grid=average_grid,
            sum_vars=sum_vars,
        )

    def compute_entry_vals(self, pred, target, pred_std):
        """
        Compute per-entry (per grid node, per variable) metric values
        before any masking or reduction.

        pred: (..., N, d_state)
        target: (..., N, d_state)
        pred_std: (..., N, d_state) or (d_state,)

        Returns: (..., N, d_state) entry-wise metric values
        """
        raise NotImplementedError

    @property
    def display_name(self):
        """
        Name to use when logging this metric.
        Override in subclasses where the logged name differs from the
        computation name (e.g. MSE is logged as RMSE after sqrt).
        """
        return self.name

    def aggregate(self, metric_tensor):
        """
        Aggregate a gathered metric tensor over the evaluation dimension.

        metric_tensor: (N_eval, pred_steps, d_f)
        Returns: (pred_steps, d_f)
        """
        return torch.mean(metric_tensor, dim=0)

    def post_process(self, averaged_tensor):
        """
        Post-processing applied to the batch-averaged metric tensor
        before rescaling to physical units.

        Default: identity (no post-processing).

        averaged_tensor: (pred_steps, d_f)
        Returns: (pred_steps, d_f)
        """
        return averaged_tensor

    def rescale(self, tensor, state_std):
        """
        Rescale metric tensor from standardized space to physical units.

        Default: linear rescaling by state_std.

        tensor: (pred_steps, d_f)
        state_std: (d_f,)
        Returns: (pred_steps, d_f)
        """
        return tensor * state_std

    def prepare_for_logging(self, metric_tensor, state_std):
        """
        Convert a gathered metric tensor into the final tensor to log/plot.

        metric_tensor: (N_eval, pred_steps, d_f)
        state_std: (d_f,)
        Returns: (pred_steps, d_f)
        """
        aggregated_tensor = self.aggregate(metric_tensor)
        post_processed_tensor = self.post_process(aggregated_tensor)
        return self.rescale(post_processed_tensor, state_std)


class MSE(BaseMetric):
    """
    (Unweighted) Mean Squared Error.
    Logged as RMSE (sqrt applied after averaging, then linear rescale).
    """

    name = "mse"

    def compute_entry_vals(self, pred, target, pred_std):
        return torch.nn.functional.mse_loss(
            pred, target, reduction="none"
        )  # (..., N, d_state)

    @property
    def display_name(self):
        return "rmse"

    def post_process(self, averaged_tensor):
        return torch.sqrt(averaged_tensor)


class MAE(BaseMetric):
    """
    (Unweighted) Mean Absolute Error.
    Linear rescaling to physical units.
    """

    name = "mae"

    def compute_entry_vals(self, pred, target, pred_std):
        return torch.nn.functional.l1_loss(
            pred, target, reduction="none"
        )  # (..., N, d_state)


class WMSE(BaseMetric):
    """
    Weighted Mean Squared Error (weighted by 1/pred_std^2).
    Logged as WRMSE (sqrt applied after averaging, then linear rescale).
    """

    name = "wmse"

    def compute_entry_vals(self, pred, target, pred_std):
        entry_mse = torch.nn.functional.mse_loss(
            pred, target, reduction="none"
        )  # (..., N, d_state)
        return entry_mse / (pred_std**2)  # (..., N, d_state)

    @property
    def display_name(self):
        return "wrmse"

    def post_process(self, averaged_tensor):
        return torch.sqrt(averaged_tensor)

    def rescale(self, tensor, state_std):
        """
        Weighted metrics are dimensionless in normalized space and should not
        be rescaled to physical units.
        """
        return tensor


class WMAE(BaseMetric):
    """
    Weighted Mean Absolute Error (weighted by 1/pred_std).
    Linear rescaling to physical units.
    """

    name = "wmae"

    def compute_entry_vals(self, pred, target, pred_std):
        entry_mae = torch.nn.functional.l1_loss(
            pred, target, reduction="none"
        )  # (..., N, d_state)
        return entry_mae / pred_std  # (..., N, d_state)

    def rescale(self, tensor, state_std):
        """
        Weighted metrics are dimensionless in normalized space and should not
        be rescaled to physical units.
        """
        return tensor


class NLL(BaseMetric):
    """
    Negative Log Likelihood loss, for isotropic Gaussian likelihood.
    No rescaling to physical units (NLL is in log-probability space).
    """

    name = "nll"

    def compute_entry_vals(self, pred, target, pred_std):
        # Broadcast pred_std if shaped (d_state,),
        # done internally in Normal class
        dist = torch.distributions.Normal(pred, pred_std)  # (..., N, d_state)
        return -dist.log_prob(target)  # (..., N, d_state)

    def rescale(self, tensor, state_std):
        """NLL values are in log-probability space, not physically
        rescalable."""
        return tensor


class CRPSGauss(BaseMetric):
    """
    (Negative) Continuous Ranked Probability Score (CRPS).
    Closed-form expression based on Gaussian predictive distribution.
    Linear rescaling to physical units.
    """

    name = "crps_gauss"

    def compute_entry_vals(self, pred, target, pred_std):
        std_normal = torch.distributions.Normal(
            torch.zeros((), device=pred.device),
            torch.ones((), device=pred.device),
        )
        target_standard = (target - pred) / pred_std  # (..., N, d_state)

        entry_crps = -pred_std * (
            torch.pi ** (-0.5)
            - 2 * torch.exp(std_normal.log_prob(target_standard))
            - target_standard * (2 * std_normal.cdf(target_standard) - 1)
        )  # (..., N, d_state)

        return entry_crps


class OutputStd(BaseMetric):
    """
    Predicted standard deviation, treated as a metric for logging.
    Linear rescaling to physical units.

    This metric uses the same callable interface as the other metrics:
    `compute_entry_vals` returns the predicted standard deviation field,
    after which masking/reduction, aggregation, and rescaling are handled
    through the shared metric-object pipeline.
    """

    name = "output_std"

    def compute_entry_vals(self, pred, target, pred_std):
        if pred_std.ndim < pred.ndim:
            expand_shape = (1,) * (pred.ndim - 1) + (-1,)
            pred_std = pred_std.view(*expand_shape).expand_as(pred)
        return pred_std


def mask_and_reduce_metric(metric_entry_vals, mask, average_grid, sum_vars):
    """
    Masks and (optionally) reduces entry-wise metric values

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    metric_entry_vals: (..., N, d_state), prediction
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Only keep grid nodes in mask
    if mask is not None:
        metric_entry_vals = metric_entry_vals[
            ..., mask, :
        ]  # (..., N', d_state)

    # Optionally reduce last two dimensions
    if average_grid:  # Reduce grid first
        metric_entry_vals = torch.mean(
            metric_entry_vals, dim=-2
        )  # (..., d_state)
    if sum_vars:  # Reduce vars second
        metric_entry_vals = torch.sum(
            metric_entry_vals, dim=-1
        )  # (..., N) or (...,)

    return metric_entry_vals


# Metric registry: maps string names to metric objects
DEFINED_METRICS = {
    "mse": MSE(),
    "mae": MAE(),
    "wmse": WMSE(),
    "wmae": WMAE(),
    "nll": NLL(),
    "crps_gauss": CRPSGauss(),
    "output_std": OutputStd(),
}


def get_metric(metric_name):
    """
    Get a defined metric object by name.

    metric_name: str, name of the metric

    Returns:
    metric: BaseMetric instance implementing the metric
    """
    metric_name_lower = metric_name.lower()
    assert (
        metric_name_lower in DEFINED_METRICS
    ), f"Unknown metric: {metric_name}"
    return DEFINED_METRICS[metric_name_lower]


# Module-level callable aliases for backward compatibility.
# These allow code like `metrics.mse(pred, target, pred_std)` to still work,
# since each metric object is callable.
mse = DEFINED_METRICS["mse"]
mae = DEFINED_METRICS["mae"]
wmse = DEFINED_METRICS["wmse"]
wmae = DEFINED_METRICS["wmae"]
nll = DEFINED_METRICS["nll"]
crps_gauss = DEFINED_METRICS["crps_gauss"]

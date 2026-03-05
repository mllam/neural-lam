
"""Package-level exports for neural_lam.metrics

This package is a thin wrapper that re-exports the metric functions
implemented in the top-level module `neural_lam/metrics.py` while avoiding
circular import issues caused by having both a package and a module named
`neural_lam.metrics`.

We load the concrete implementation from the file `metrics.py` under a
private module name and re-export the desired symbols into the package
namespace so callers can do `from neural_lam.metrics import mse, get_metric`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load the implementation in neural_lam/metrics.py under a private name.
_metrics_impl_path = Path(__file__).resolve().parent.parent / "metrics.py"
spec = importlib.util.spec_from_file_location(
	"neural_lam._metrics_impl", str(_metrics_impl_path)
)
_metrics_impl = importlib.util.module_from_spec(spec)
sys.modules["neural_lam._metrics_impl"] = _metrics_impl
spec.loader.exec_module(_metrics_impl)  # type: ignore

# Re-export commonly used symbols from the implementation module
for _name in (
	"mse",
	"mae",
	"wmse",
	"wmae",
	"nll",
	"crps_gauss",
	"crps_ensemble",
	"get_metric",
	"DEFINED_METRICS",
):
	if hasattr(_metrics_impl, _name):
		globals()[_name] = getattr(_metrics_impl, _name)


def get_metric(metric_name: str):
	"""Return metric function by name using the implementation mapping."""
	metric_name_lower = metric_name.lower()
	mapping = getattr(_metrics_impl, "DEFINED_METRICS", None)
	if mapping is None:
		raise ImportError("Metric implementation mapping not found")
	if metric_name_lower not in mapping:
		raise KeyError(f"Unknown metric: {metric_name}")
	return mapping[metric_name_lower]


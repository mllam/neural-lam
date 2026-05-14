# Standard library
from typing import NamedTuple, Sequence

# Third-party
import torch


class ForecastBatch(NamedTuple):
    """Named WeatherDataset sample or collated batch.

    DataLoader's default collation preserves this NamedTuple type while adding
    the leading batch dimension to each tensor field.
    """

    init_states: torch.Tensor
    target_states: torch.Tensor
    forcing: torch.Tensor
    target_times: torch.Tensor

    def to(self, device: torch.device | str) -> "ForecastBatch":
        """Move all tensor fields to the same device."""
        return ForecastBatch(
            init_states=self.init_states.to(device),
            target_states=self.target_states.to(device),
            forcing=self.forcing.to(device),
            target_times=self.target_times.to(device),
        )


def coerce_forecast_batch(
    batch: ForecastBatch | Sequence[torch.Tensor],
) -> ForecastBatch:
    """Return batch as ForecastBatch.

    Legacy 4-element batch tuples are accepted during the migration.
    """
    if isinstance(batch, ForecastBatch):
        return batch

    if len(batch) != len(ForecastBatch._fields):
        raise ValueError(
            "Expected a ForecastBatch or 4-element legacy batch tuple, "
            f"got {len(batch)} elements."
        )

    return ForecastBatch(
        init_states=batch[0],
        target_states=batch[1],
        forcing=batch[2],
        target_times=batch[3],
    )

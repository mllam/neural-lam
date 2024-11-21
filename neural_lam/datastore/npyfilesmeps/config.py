# Standard library
from dataclasses import dataclass, field
from typing import Any, Dict, List

# Third-party
import dataclass_wizard


@dataclass
class Projection:
    """Represents the projection information for a dataset, including the type
    of projection and its parameters. Capable of creating a cartopy.crs
    projection object.

    Attributes:
        class_name: The class name of the projection, this should be a valid
        cartopy.crs class.
        kwargs: A dictionary of keyword arguments specific to the projection
        type.

    """

    class_name: str
    kwargs: Dict[str, Any]


@dataclass
class Dataset:
    """Contains information about the dataset, including variable names, units,
    and descriptions.

    Attributes:
        name: The name of the dataset.
        var_names: A list of variable names in the dataset.
        var_units: A list of units for each variable.
        var_longnames: A list of long, descriptive names for each variable.
        num_forcing_features: The number of forcing features in the dataset.

    """

    name: str
    var_names: List[str]
    var_units: List[str]
    var_longnames: List[str]
    num_forcing_features: int
    num_timesteps: int
    step_length: int
    num_ensemble_members: int
    remove_state_features_with_index: List[int] = field(default_factory=list)


@dataclass
class NpyDatastoreConfig(dataclass_wizard.YAMLWizard):
    """Configuration for loading and processing a dataset, including dataset
    details, grid shape, and projection information.

    Attributes:
        dataset: An instance of Dataset containing details about the dataset.
        grid_shape_state: A list representing the shape of the grid state.
        projection: An instance of Projection containing projection details.

    """

    dataset: Dataset
    grid_shape_state: List[int]
    projection: Projection

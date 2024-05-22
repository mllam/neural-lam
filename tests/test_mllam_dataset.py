# Standard library
from pathlib import Path

# First-party
from neural_lam.weather_dataset import WeatherDataset
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.utils import load_graph, load_static_data
from neural_lam.config import Config
from train_model import main

# Third-party
import numpy as np
import weather_model_graphs as wmg


def test_load_reduced_meps_dataset():
    data_config_file = 'data/meps_example_reduced/data_config.yaml'
    dataset_name = 'meps_example_reduced'

    dataset = WeatherDataset(dataset_name="meps_example_reduced")
    config = Config.from_file(data_config_file)

    var_names = config.values['dataset']['var_names']
    var_units = config.values['dataset']['var_units']
    var_longnames = config.values['dataset']['var_longnames']

    assert len(var_names) == len(var_longnames)
    assert len(var_names) == len(var_units)
    
    # TODO: can these two variables be loaded from elsewhere?
    n_grid_static_features = 4
    n_input_steps = 2

    n_forcing_features = config.values['dataset']['num_forcing_features']
    n_state_features = len(var_names)
    n_prediction_timesteps = dataset.sample_length - n_input_steps
    
    nx, ny = config.values['grid_shape_state']
    n_grid = nx * ny

    # check that the dataset is not empty
    assert len(dataset) > 0

    # get the first item
    init_states, target_states, forcing = dataset[0]
    
    # check that the shapes of the tensors are correct
    assert init_states.shape == (
        n_input_steps, 
        n_grid, 
        n_state_features
    )
    assert target_states.shape == (
        n_prediction_timesteps,
        n_grid,
        n_state_features,
    )
    assert forcing.shape == (
        n_prediction_timesteps,
        n_grid,
        n_forcing_features,
    )

    static_data = load_static_data(dataset_name=dataset_name)
    
    required_props = {'border_mask', 'grid_static_features', 'step_diff_mean', 'step_diff_std', 'data_mean', 'data_std', 'param_weights'}
    
    # check the sizes of the props
    assert static_data["border_mask"].shape == (n_grid, 1)
    assert static_data["grid_static_features"].shape == (n_grid, n_grid_static_features)
    assert static_data["step_diff_mean"].shape == (n_state_features,)
    assert static_data["step_diff_std"].shape == (n_state_features,)
    assert static_data["data_mean"].shape == (n_state_features,)
    assert static_data["data_std"].shape == (n_state_features,)
    assert static_data["param_weights"].shape == (n_state_features,)

    assert set(static_data.keys()) == required_props
    

def test_create_graph_reduced_meps_dataset():
    dataset_name = "meps_example_reduced"
    static_dir_path = Path("data", dataset_name, "static")
    graph_dir_path = Path("graphs", "hierarchial")

    # -- Static grid node features --
    xy_grid = np.load(static_dir_path / "nwp_xy.npy")

    # create the full graph
    graph = wmg.create.archetype.create_oscarsson_hierarchical_graph(xy_grid=xy_grid)

    # split the graph by component
    graph_components = wmg.split_graph_by_edge_attribute(
        graph=graph, attr="component"
        # argument attribute seens to have been changed to attr, change also in weather-model-graphs/src/weather_model_graphs/save.py::to_pyg
    )

    m2m_graph = graph_components.pop("m2m")
    m2m_graph_components = wmg.split_graph_by_edge_attribute(
        graph=m2m_graph, attr="direction"
    )
    m2m_graph_components = {
        f"m2m_{name}": graph for name, graph in m2m_graph_components.items()
    }
    graph_components.update(m2m_graph_components)

    # save the graph components to disk in pytorch-geometric format
    for component_name, graph_component in graph_components.items():
        kwargs = {}
        wmg.save.to_pyg(
            graph=graph_component,
            name=component_name,
            output_directory=graph_dir_path,
            **kwargs,
        )


def test_train_model_reduced_meps_dataset():
    args = [
        '--model=hi_lam',
        '--data_config=data/meps_example_reduced/data_config.yaml',
        '--n_workers=1',
        '--epochs=1',
        '--graph=hierarchical',
        '--hidden_dim=16',
        '--hidden_layers=1',
        '--processor_layers=1',
        '--ar_steps=1',
        '--eval=val',
        '--wandb_project=None',
    ]
    main(args)


    
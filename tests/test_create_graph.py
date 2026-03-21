# tests/test_create_graph.py mein add karo ya existing test mein

import json
from pathlib import Path

def test_graph_config_saved(graph_output_dir):
    """Test that graph_config.json is saved with correct keys"""
    config_path = Path(graph_output_dir) / "graph_config.json"
    
    assert config_path.exists(), "graph_config.json should be created"
    
    with open(config_path) as f:
        config = json.load(f)
    
    required_keys = [
        "hierarchical",
        "n_max_levels", 
        "graph_name",
        "neural_lam_version",
        "created_at",
    ]
    for key in required_keys:
        assert key in config, f"Key '{key}' missing from graph_config.json"
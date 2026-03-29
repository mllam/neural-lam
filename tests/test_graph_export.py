import torch
from neural_lam.create_graph import export_graph


def test_export_graph(tmp_path):
    edge_index = torch.tensor([[0, 1], [1, 2]])
    num_nodes = 3

    output = tmp_path / "graph.json"

    export_graph(edge_index, num_nodes, output)

    assert output.exists()
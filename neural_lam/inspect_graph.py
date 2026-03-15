from pathlib import Path
import argparse
from networkx import edges
import torch


EDGE_FILES = [
    "m2m_edge_index.pt",
    "g2m_edge_index.pt",
    "m2g_edge_index.pt",
]

FEATURE_FILES = [
    "mesh_features.pt",
    "g2m_features.pt",
    "m2g_features.pt",
]


def inspect_graph(graph_dir: Path):
    print("\nGraph Summary")
    print("----------------------------")

    for edge_file in EDGE_FILES:
        path = graph_dir / edge_file
        if path.exists():
            edges = torch.load(path)
            num_edges = edges.shape[1]
            num_nodes = edges.max().item() + 1

            print(f"{edge_file}: {num_edges} edges")
            print(f"estimated nodes: {num_nodes}")
            
        else:
            print(f"{edge_file}: missing")

    print("\nFeature Files")
    print("----------------------------")

    for feature_file in FEATURE_FILES:
        path = graph_dir / feature_file
        status = "present" if path.exists() else "missing"
        print(f"{feature_file}: {status}")


def main():
    parser = argparse.ArgumentParser(description="Inspect Neural-LAM graph files")
    parser.add_argument("--graph_dir", required=True)

    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)

    if not graph_dir.exists():
        raise FileNotFoundError(f"{graph_dir} does not exist")

    inspect_graph(graph_dir)


if __name__ == "__main__":
    main()
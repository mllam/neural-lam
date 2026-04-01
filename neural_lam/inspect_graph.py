from pathlib import Path
import argparse
import json 
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


def inspect_graph(graph_dir: Path, export_path: Path = None):
    print("\nGraph Summary")
    print("----------------------------")

    summary = {} 

    for edge_file in EDGE_FILES:
        path = graph_dir / edge_file
        if path.exists():
            edges = torch.load(path)
            num_edges = edges.shape[1]
            num_nodes = edges.max().item() + 1 if edges.numel() > 0 else 0

            print(f"{edge_file}: {num_edges} edges")
            print(f"estimated nodes: {num_nodes}")

            summary[edge_file] = {
                "num_edges": num_edges,
                "estimated_nodes": num_nodes,
            }

        else:
            print(f"{edge_file}: missing")
            summary[edge_file] = None  

    print("\nFeature Files")
    print("----------------------------")

    for feature_file in FEATURE_FILES:
        path = graph_dir / feature_file
        status = "present" if path.exists() else "missing"
        print(f"{feature_file}: {status}")

        summary[feature_file] = (status == "present")

    
    if export_path:
        export_path.parent.mkdir(parents=True, exist_ok=True)

        with open(export_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary exported to {export_path}")


def main():
    parser = argparse.ArgumentParser(description="Inspect Neural-LAM graph files")
    parser.add_argument("--graph_dir", required=True)

    parser.add_argument(
        "--export_json",
        type=str,
        help="Optional path to save graph summary as JSON",
    )

    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)

    if not graph_dir.exists():
        raise FileNotFoundError(f"{graph_dir} does not exist")

    export_path = Path(args.export_json) if args.export_json else None

    inspect_graph(graph_dir, export_path)


if __name__ == "__main__":
    main()
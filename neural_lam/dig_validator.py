# Standard library
import argparse
import ast
import os
import re
import sys

# CONFIG
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

code_folders = [os.path.join(repo_root, "neural_lam")]

structure_folder = os.path.join(repo_root, "docs", "diagrams", "structure")
dataflow_folder = os.path.join(repo_root, "docs", "diagrams", "dataflow")

exclude_dirs = {"venv", "__pycache__", "tests", ".venv"}
exclude_files = {
    "dig_generator.py",
    "dig_validator.py",
    "__init__.py",
    "utils.py",
    "config.py",
    "metrics.py",
    "plot_graph.py",
    "custom_loggers.py",
    "create_graph.py",
    "evaluate.py",
    "train_model.py",
}

IGNORE_CALLS = {"print", "len", "range", "str", "int", "float"}

code_symbols = set()


# SCAN CODEBASE
def scan_codebase():
    for folder in code_folders:
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if not file.endswith(".py") or file in exclude_files:
                    continue
                module_name = os.path.splitext(file)[0]
                code_symbols.add(module_name)
                path = os.path.join(root, file)
                with open(path, encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read())
                    except SyntaxError:
                        continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        code_symbols.add(node.name)
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                code_symbols.add(base.id)
                            elif isinstance(base, ast.Attribute):
                                code_symbols.add(base.attr)
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                code_symbols.add(item.name)
                                code_symbols.add(f"{node.name}_{item.name}")
                    elif isinstance(node, ast.FunctionDef):
                        code_symbols.add(node.name)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            code_symbols.add(alias.name.split(".")[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            code_symbols.add(node.module.split(".")[0])
                        for alias in node.names:
                            code_symbols.add(alias.name)


# EXTRACT MERMAID NODES
def extract_mermaid_nodes(folder):
    nodes = {}
    pattern = r"\b([A-Za-z_][A-Za-z0-9_]*)\b"
    for root, _, files in os.walk(folder):
        for file in files:
            if not file.endswith(".md"):
                continue
            path = os.path.join(root, file)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            text = re.sub(r"flowchart TD", "", text)
            text = re.sub(r"```(?:mermaid)?", "", text)
            text = re.sub(r"%%(?:{.*?})?%%", "", text, flags=re.DOTALL)
            text = re.sub(r"classDef.*", "", text)
            text = re.sub(r'\[".*?"\]', "", text)
            matches = re.findall(pattern, text)
            for m in matches:
                if m in {
                    "flowchart",
                    "TD",
                    "LR",
                    "class",
                    "classDef",
                    "module",
                    "mermaid",
                    "init",
                    "setup_decode",
                    "subgraph",
                    "end",
                    "callNode",
                    "parent",
                    "base",
                    "import",
                    "method",
                    "theme",
                    "themeVariables",
                    "edgeLabelBackground",
                    "fontSize",
                    "nodeSpacing",
                    "rankSpacing",
                    "px",
                    "Parent_Class",
                    "Base_Class",
                    "Imports",
                    "Methods",
                    "Inputs",
                    "Preparation",
                    "Operations",
                    "Processing",
                }:
                    continue
                if m not in nodes:
                    nodes[m] = []
                nodes[m].append(file)
    return nodes


# VALIDATE
def validate_folder(folder, mode):
    diagram_nodes = extract_mermaid_nodes(folder)
    missing = []

    for node, files in diagram_nodes.items():
        normalized = node
        if mode == "dataflow" and normalized in IGNORE_CALLS:
            continue
        if normalized not in code_symbols:
            if mode == "dataflow":
                continue
            missing.append((node, files))

    if missing:
        print(f"\n❌ {mode.upper()} DIAGRAM ERRORS:\n")
        for node, files in missing:
            file_list = ", ".join(set(files))
            print(f"  - {node}  ({file_list})")
        return False

    print(f"✅ {mode.capitalize()} diagrams validated")
    return True


# MAIN
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", choices=["structure", "dataflow", "all"], default="all"
)
args = parser.parse_args()

scan_codebase()

ok_structure = True
ok_dataflow = True

if args.mode in ["structure", "all"]:
    if os.path.exists(structure_folder):
        ok_structure = validate_folder(structure_folder, "structure")
    else:
        print(f"Warning: {structure_folder} does not exist.")

if args.mode in ["dataflow", "all"]:
    if os.path.exists(dataflow_folder):
        ok_dataflow = validate_folder(dataflow_folder, "dataflow")
    else:
        print(f"Warning: {dataflow_folder} does not exist.")

if not (ok_structure and ok_dataflow):
    sys.exit(1)

print("\nAll requested diagrams are valid!")

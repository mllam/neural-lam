import ast
import os
import re
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
code_folders = [os.path.join(repo_root, "neural_lam")]
docs_folder = os.path.join(repo_root, "docs", "diagrams")
exclude_dirs = {"venv", "__pycache__", "tests", "utils"}
code_symbols = set()


def scan_codebase():

    for folder in code_folders:
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file.endswith(".py"):

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

                        if isinstance(node, ast.FunctionDef):
                            code_symbols.add(node.name)

                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                code_symbols.add(alias.name.split(".")[0])

                        if isinstance(node, ast.ImportFrom):
                            if node.module:
                                code_symbols.add(node.module.split(".")[0])


def extract_mermaid_nodes():

    nodes = {}

    pattern = r'\["([^"]+)"\]'

    for root, _, files in os.walk(docs_folder):

        for file in files:

            if file.endswith(".md"):

                path = os.path.join(root, file)

                with open(path, encoding="utf-8") as f:
                    text = f.read()

                matches = re.findall(pattern, text)

                for m in matches:

                    clean = m.replace("()", "")
                    if clean not in nodes:
                        nodes[clean] = []
                    nodes[clean].append(file)

    return nodes


scan_codebase()

diagram_nodes = extract_mermaid_nodes()

missing = []

for node, files in diagram_nodes.items():

    if node not in code_symbols:
        missing.append((node, files))


if missing:

    print("\n❌ Mermaid diagram references missing code:\n")

    for node, files in missing:
        file_list = ", ".join(files)
        print(f"  - {node}  ({file_list})")

    sys.exit(1)

else:
    print("✅ Mermaid diagrams match the codebase")
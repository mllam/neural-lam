# Standard library
import ast
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folders_to_scan = [os.path.join(repo_root, "neural_lam")]

diagrams_folder = os.path.join(repo_root, "docs", "diagrams")
os.makedirs(diagrams_folder, exist_ok=True)

exclude_dirs = {"venv", "__pycache__", "tests", "utils"}
exclude_files = {"dig_generator.py", "dig_validator.py", "__init__.py"}


def diagram_changed(path, content):
    """Write diagram to file only if content has changed."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            if f.read() == content:
                return
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def parse_file(filepath):
    """Parse a Python file and extract structure."""
    module_name = os.path.splitext(os.path.basename(filepath))[0]

    imports = set()
    classes = []
    inheritance = []
    methods = {}
    calls = {}

    with open(filepath, encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            print(f"Skipping {filepath} (syntax error)")
            return module_name, imports, classes, inheritance, methods, calls

    for node in ast.walk(tree):

        if isinstance(node, ast.Import):
            for name in node.names:
                imports.add(name.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

        elif isinstance(node, ast.ClassDef):
            class_name = node.name
            classes.append(class_name)
            methods[class_name] = []

            for base in node.bases:
                if isinstance(base, ast.Name):
                    inheritance.append((class_name, base.id))
                elif isinstance(base, ast.Attribute):
                    inheritance.append((class_name, base.attr))

            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name != "__init__"
                ):
                    method_name = item.name
                    methods[class_name].append(method_name)

                    calls_key = f"{class_name}_{method_name}"
                    calls[calls_key] = []
                    for node2 in ast.walk(item):
                        if isinstance(node2, ast.Call):
                            if isinstance(node2.func, ast.Attribute):
                                if node2.func.attr in ("encode", "decode"):
                                    calls[calls_key].append(node2.func.attr)
                            elif isinstance(node2.func, ast.Name):
                                if node2.func.id in ("encode", "decode"):
                                    calls[calls_key].append(node2.func.id)

    return module_name, imports, classes, inheritance, methods, calls


def generate_mermaid(
    module_name, imports, classes, inheritance, methods, calls
):
    """Generate Mermaid diagram code based on extracted data."""
    diagram = (
        "%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80}}}%%\n"
    )
    diagram += "flowchart TD\n\n"

    # Module node
    diagram += f'module["{module_name}"]\n\n'

    # Parent classes
    diagram += "subgraph Parent_Class\n"
    parents = set(parent for _, parent in inheritance)
    for p in parents:
        diagram += f'    {p}["{p}"]\n'
    diagram += "end\n\n"

    # Base classes
    diagram += "subgraph Base_Class\n"
    for cls in classes:
        diagram += f'    {cls}["{cls}"]\n'
    diagram += "end\n\n"

    # Inheritance arrows
    for child, parent in inheritance:
        diagram += f"    {parent} --> {child}\n"

    # Imports
    diagram += "\nsubgraph Imports\n"
    for imp in imports:
        node_name = imp.replace(".", "_")
        diagram += f'    {node_name}["{imp}"]\n'
    diagram += "end\n\n"

    # Imports → module
    for imp in imports:
        node_name = imp.replace(".", "_")
        diagram += f"    {node_name} --> module\n"

    # Module → classes
    for cls in classes:
        diagram += f"    module --> {cls}\n"

    # Methods
    diagram += "\nsubgraph Methods\n"
    for cls, funcs in methods.items():
        for func in funcs:
            node = f"{cls}_{func}"
            diagram += f'    {node}["{func}()"]\n'
    diagram += "end\n\n"

    # Class → methods
    for cls, funcs in methods.items():
        for func in funcs:
            node = f"{cls}_{func}"
            diagram += f"    {cls} --> {node}\n"

    # Call nodes (encode/decode)
    all_calls = []
    for method_node, call_list in calls.items():
        for call_name in call_list:
            safe_name = (
                call_name.replace(".", "_").replace("(", "").replace(")", "")
            )
            node_name = f"{method_node}_{safe_name}"
            diagram += f"    {method_node} --> {node_name}\n"
            all_calls.append(node_name)

    if all_calls:
        diagram += "class " + ",".join(all_calls) + " callNode\n"

    # Dark-mode styling
    styles = [
        "classDef parent fill:#0f172a,stroke:#3b82f6,"
        "stroke-width:2px,color:#f1f5f9,font-size:16px",
        "classDef base fill:#78350f,stroke:#f59e0b,"
        "stroke-width:1px,color:#fde68a,font-size:16px",
        "classDef import fill:#1f2937,stroke:#6b7280,"
        "stroke-width:1.5px,color:#e5e7eb,font-size:16px",
        "classDef method fill:#2d043f,stroke:#7c3aed,"
        "stroke-width:1.5px,color:#ede9fe,font-size:16px",
        "classDef callNode fill:#064e3b,stroke:#10b981,"
        "stroke-width:1.5px,color:#d1fae5,font-size:16px",
    ]
    diagram += "\n" + "\n".join(styles) + "\n"

    if parents:
        diagram += "class " + ",".join(parents) + " parent\n"
    if classes:
        diagram += "class " + ",".join(classes) + " base\n"
    if imports:
        nodes = [imp.replace(".", "_") for imp in imports]
        diagram += "class " + ",".join(nodes) + " import\n"
    method_nodes = []
    for cls, funcs in methods.items():
        for func in funcs:
            method_nodes.append(f"{cls}_{func}")
    if method_nodes:
        diagram += "class " + ",".join(method_nodes) + " method\n"

    return diagram


found_files = 0
for folder_to_scan in folders_to_scan:
    for root, dirs, files in os.walk(folder_to_scan):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.lower().endswith(".py") and file not in exclude_files:
                found_files += 1
                filepath = os.path.join(root, file)
                print(f"Found Python file: {filepath}")
                module_name, imports, classes, inheritance, methods, calls = (
                    parse_file(filepath)
                )
                mermaid = generate_mermaid(
                    module_name, imports, classes, inheritance, methods, calls
                )
                output_filename = os.path.join(
                    diagrams_folder, f"{module_name}_diagram.md"
                )
                diagram_changed(
                    output_filename, "```mermaid\n" + mermaid + "\n```"
                )
                print(f"Diagram generated: {output_filename}")

if found_files == 0:
    print("No Python files found in the specified folders.")

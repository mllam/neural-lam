# Standard library
import argparse
import ast
import os

# --- Configuration & Ignore Lists ---
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folders_to_scan = [os.path.join(repo_root, "neural_lam")]

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

IGNORE_CALLS = {
    "print",
    "len",
    "range",
    "str",
    "int",
    "float",
    "list",
    "dict",
    "set",
    "bool",
    "tuple",
    "type",
    "super",
    "zip",
    "enumerate",
    "map",
    "filter",
    "sorted",
    "reversed",
    "isinstance",
    "hasattr",
    "getattr",
    "setattr",
    "append",
    "extend",
    "update",
    "items",
    "keys",
    "values",
    "format",
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["structure", "dataflow"],
    default="structure",
    help="Diagram mode",
)
args = parser.parse_args()

if args.mode == "structure":
    diagrams_folder = os.path.join(repo_root, "docs", "diagrams", "structure")
else:
    diagrams_folder = os.path.join(repo_root, "docs", "diagrams", "dataflow")

os.makedirs(diagrams_folder, exist_ok=True)


# --- File & AST Utility Functions ---
def diagram_changed(path, content):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            if f.read() == content:
                return
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def get_full_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        val = get_full_name(node.value)
        return f"{val}.{node.attr}" if val else node.attr
    return None


def clean_self(name):
    if not name:
        return None
    parts = name.split(".")
    if parts[0] == "self":
        return ".".join(parts[1:])
    return name


# --- Architectural Structure Parser ---
def parse_structure(filepath):
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    imports = set()
    classes = []
    inheritance = []
    methods = {}

    with open(filepath, encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return module_name, imports, classes, inheritance, methods

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
            methods[node.name] = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    inheritance.append((node.name, base.id))
                elif isinstance(base, ast.Attribute):
                    inheritance.append((node.name, base.attr))
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name != "__init__":
                        methods[node.name].append(item.name)

    return module_name, imports, classes, inheritance, methods


# --- Dataflow AST Parser ---
def parse_dataflow_for_method(func_node):
    node_types = {}
    edges = []

    var_source = {}

    for arg in func_node.args.args:
        if arg.arg != "self":
            node_types[arg.arg] = "input"
            var_source[arg.arg] = arg.arg

    def resolve_args(call_node):
        names = []
        for a in call_node.args:
            n = clean_self(get_full_name(a))
            if n:
                names.append(n)
        for kw in call_node.keywords:
            n = clean_self(get_full_name(kw.value))
            if n:
                names.append(n)
        return names

    def get_edge_label(var_name, is_input):
        mapping = {
            "m2m_emb": "batched" if is_input else "m2m_emb",
            "m2m_emb_expanded": (
                "context embeddings" if is_input else "expanded embeddings"
            ),
            "mesh_rep": (
                "input mesh" if is_input else "updated mesh representation"
            ),
        }
        if var_name in mapping:
            return mapping[var_name]
        return var_name.replace("_", " ")

    def process_call(func_name, target_names, call_node):
        if not func_name or func_name in IGNORE_CALLS:
            return
        node_types.setdefault(func_name, "function")

        for inp in resolve_args(call_node):
            if inp in var_source:
                lbl = get_edge_label(inp, True)
                edges.append((var_source[inp], func_name, lbl, False))
        for t in target_names:
            if t != "_":
                lbl = get_edge_label(t, False)
                if node_types.get(t) == "input":
                    var_source[t] = func_name
                else:
                    node_types[t] = "variable"
                    edges.append((func_name, t, lbl, False))
                    var_source[t] = t

    for stmt in func_node.body:
        if isinstance(stmt, ast.Assign):
            target_names = []
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    target_names.append(t.id)
                elif isinstance(t, ast.Tuple):
                    for elt in t.elts:
                        if isinstance(elt, ast.Name):
                            target_names.append(elt.id)

            rhs = stmt.value
            if isinstance(rhs, ast.Call):
                func_name = clean_self(get_full_name(rhs.func))
                process_call(func_name, target_names, rhs)
            elif isinstance(rhs, ast.Tuple):
                for elt in rhs.elts:
                    if isinstance(elt, ast.Call):
                        func_name = clean_self(get_full_name(elt.func))
                        process_call(func_name, target_names, elt)
            else:
                src = clean_self(get_full_name(rhs))
                if src in var_source:
                    for t in target_names:
                        var_source[t] = var_source[src]

        elif isinstance(stmt, ast.Return) and stmt.value is not None:
            ret = stmt.value
            node_types["output"] = "output"
            if isinstance(ret, ast.Call):
                func_name = clean_self(get_full_name(ret.func))
                process_call(func_name, [], ret)
                if func_name and func_name not in IGNORE_CALLS:
                    edges.append(
                        (func_name, "output", "updated representation", True)
                    )
            else:
                src_name = clean_self(get_full_name(ret))
                if src_name and src_name in var_source:
                    is_main = True
                    lbl = (
                        get_edge_label(src_name, False)
                        if var_source[src_name] != src_name
                        else get_edge_label(src_name, True)
                    )
                    edges.append((var_source[src_name], "output", lbl, is_main))

    return node_types, edges


def extract_all_methods(filepath):
    module = os.path.splitext(os.path.basename(filepath))[0]
    methods = {}
    with open(filepath, encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return module, methods

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name != "__init__":
                        full_name = f"{node.name}.{item.name}"
                        methods[full_name] = item
            self.generic_visit(node)

    Visitor().visit(tree)
    return module, methods


# --- Mermaid Diagram Generators ---
def generate_structure(module_name, imports, classes, inheritance, methods):
    lines = []
    lines.append(
        "%%{init: {'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80}}}%%"
    )
    lines.append("flowchart TD\n")
    lines.append(f'module["{module_name}"]\n')

    parents = {parent for child, parent in inheritance}
    bases = set(classes)

    if parents:
        lines.append("subgraph Parent_Class")
        for p in parents:
            lines.append(f'    {p}["{p}"]')
        lines.append("end\n")

    if bases:
        lines.append("subgraph Base_Class")
        for b in bases:
            lines.append(f'    {b}["{b}"]')
        lines.append("end\n")

    for child, parent in inheritance:
        lines.append(f"    {parent} --> {child}")
    if inheritance:
        lines.append("")

    if imports:
        lines.append("subgraph Imports")
        for i in imports:
            lines.append(f'    {i}["{i}"]')
        lines.append("end\n")
        for i in imports:
            lines.append(f"    {i} --> module")

    for b in bases:
        lines.append(f"    module --> {b}")
    if bases or imports:
        lines.append("")

    all_methods = []
    for cls, funcs in methods.items():
        for f in funcs:
            all_methods.append(f"{cls}_{f}")

    if all_methods:
        lines.append("subgraph Methods")
        for m in all_methods:
            func_name = m.split("_")[-1]
            lines.append(f'    {m}["{func_name}()"]')
        lines.append("end\n")

    for cls, funcs in methods.items():
        for f in funcs:
            lines.append(f"    {cls} --> {cls}_{f}")
    if all_methods:
        lines.append("")

    lines.append(
        "classDef parent fill:#0f172a,stroke:#3b82f6,stroke-width:2px,"
        "color:#f1f5f9,font-size:16px"
    )
    lines.append(
        "classDef base fill:#78350f,stroke:#f59e0b,stroke-width:1px,"
        "color:#fde68a,font-size:16px"
    )
    lines.append(
        "classDef import fill:#1f2937,stroke:#6b7280,stroke-width:1.5px,"
        "color:#e5e7eb,font-size:16px"
    )
    lines.append(
        "classDef method fill:#2d043f,stroke:#7c3aed,stroke-width:1.5px,"
        "color:#ede9fe,font-size:16px"
    )
    lines.append(
        "classDef callNode fill:#064e3b,stroke:#10b981,stroke-width:1.5px,"
        "color:#d1fae5,font-size:16px"
    )

    if parents:
        lines.append(f"class {','.join(parents)} parent")
    if bases:
        lines.append(f"class {','.join(bases)} base")
    if imports:
        lines.append(f"class {','.join(imports)} import")
    if all_methods:
        lines.append(f"class {','.join(all_methods)} method")

    return "\n".join(lines)


def generate_dataflow(node_types, edges, method_name):
    input_nodes = [n for n, t in node_types.items() if t == "input"]
    func_nodes = [n for n, t in node_types.items() if t == "function"]
    var_nodes = [n for n, t in node_types.items() if t == "variable"]
    has_output = "output" in node_types

    lines = []
    lines.append(
        "%%{init: {'theme': 'dark', "
        "'themeVariables': {'edgeLabelBackground': '#000000'}, "
        "'flowchart': {'nodeSpacing': 60, 'rankSpacing': 80}}}%%"
    )
    lines.append("flowchart LR")
    lines.append("")

    if input_nodes:
        lines.append("subgraph Inputs")
        for n in input_nodes:
            lines.append(f'    {n}["{n}"]')
        lines.append("end")
        lines.append("")

    all_op_nodes = func_nodes + var_nodes
    input_set = set(input_nodes)

    reachable_from_inputs = set()
    frontier = list(input_set)
    visited = set(input_set)
    while frontier:
        nxt = []
        for node in frontier:
            for edge in edges:
                src = edge[0]
                dst = edge[1]
                if src == node and dst not in visited:
                    visited.add(dst)
                    reachable_from_inputs.add(dst)
                    nxt.append(dst)
        frontier = nxt

    prep_group = [n for n in all_op_nodes if n not in reachable_from_inputs]
    proc_group = [n for n in all_op_nodes if n in reachable_from_inputs]

    if prep_group and proc_group:
        lines.append("subgraph Preparation")
        for n in prep_group:
            lines.append(f'    {n}["{n}"]')
        lines.append("end")
        lines.append("")
        lines.append("subgraph Processing")
        for n in proc_group:
            lines.append(f'    {n}["{n}"]')
        lines.append("end")
        lines.append("")
    else:
        combined = prep_group or proc_group
        if combined:
            lines.append("subgraph Operations")
            for n in combined:
                lines.append(f'    {n}["{n}"]')
            lines.append("end")
            lines.append("")

    if has_output:
        lines.append('    output(["output"])')
        lines.append("")

    seen = set()
    for edge in edges:
        src, dst = edge[0], edge[1]
        label = edge[2] if len(edge) > 2 else None
        is_main = edge[3] if len(edge) > 3 else False

        key = (src, dst)
        if key in seen:
            continue
        seen.add(key)

        link_str = "==>" if is_main else "-->"
        if label:
            lines.append(f'    {src} {link_str}|"{label}"| {dst}')
        else:
            lines.append(f"    {src} {link_str} {dst}")

    lines.append("")

    lines.append("classDef base fill:#78350f,stroke:#f59e0b,color:#fde68a")
    lines.append("classDef method fill:#2d043f,stroke:#7c3aed,color:#ede9fe")
    lines.append("classDef callNode fill:#064e3b,stroke:#10b981,color:#d1fae5")
    lines.append("")

    if input_nodes:
        lines.append(f"class {','.join(input_nodes)} base")
    if func_nodes:
        lines.append(f"class {','.join(func_nodes)} method")
    output_class_nodes = var_nodes + (["output"] if has_output else [])
    if output_class_nodes:
        lines.append(f"class {','.join(output_class_nodes)} callNode")

    return "\n".join(lines) + "\n"


# --- Core Application Loop ---
found_files = 0

for folder_to_scan in folders_to_scan:
    for root, dirs, files in os.walk(folder_to_scan):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if not file.lower().endswith(".py") or file in exclude_files:
                continue
            found_files += 1
            filepath = os.path.join(root, file)
            print(f"Processing: {filepath}")

            if args.mode == "structure":
                module_name, imports, classes, inheritance, methods = (
                    parse_structure(filepath)
                )
                mermaid = generate_structure(
                    module_name, imports, classes, inheritance, methods
                )
                output_filename = os.path.join(
                    diagrams_folder, f"{os.path.splitext(file)[0]}_diagram.md"
                )
                diagram_changed(
                    output_filename, "```mermaid\n" + mermaid + "\n```\n"
                )
                print(f"  Generated: {output_filename}")

            else:
                module_name, methods = extract_all_methods(filepath)

                core_methods = {}
                for m_name, f_node in methods.items():
                    ml = m_name.lower()
                    if (
                        "process_step" in ml
                        or "forward" in ml
                        or "hi_processor_step" in ml
                        or "common_step" in ml
                    ):
                        core_methods[m_name] = f_node
                methods = core_methods

                if not methods:
                    continue

                output_filename = os.path.join(
                    diagrams_folder, f"{os.path.splitext(file)[0]}_dataflow.md"
                )

                md_blocks = [f"# Dataflow: `{module_name}`\n"]
                all_node_types = {}
                all_edges = []

                for method_full_name, func_node in methods.items():
                    node_types, edges = parse_dataflow_for_method(func_node)
                    all_node_types.update(node_types)
                    all_edges.extend(edges)

                if all_edges:
                    mermaid = generate_dataflow(
                        all_node_types, all_edges, module_name
                    )
                    md_blocks.append("```mermaid\n" + mermaid + "```\n")
                    diagram_changed(output_filename, "\n".join(md_blocks))
                    print(f"  Generated: {output_filename}")

if found_files == 0:
    print("No Python files found.")

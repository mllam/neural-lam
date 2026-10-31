"""Tests for agent skills metadata, syntax, imports, and symbol validity."""

# Standard library
import ast
import importlib
import importlib.util
import re
from pathlib import Path

# Third-party
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = REPO_ROOT / ".agents" / "skills"
SKILL_FILES = sorted(list(SKILLS_DIR.glob("*/SKILL.md")))

EXPECTED_SKILLS = [
    "add-datastore",
    "add-loss",
    "add-metric",
    "add-step-predictor",
    "create-graph",
    "prepare-pr",
    "review-pr",
    "smoke-train",
]

SKILL_PACKAGE_CONTEXT = {
    "add-datastore": "neural_lam.datastore",
    "add-step-predictor": "neural_lam.models",
}

PLACEHOLDER_NAMES = {
    "my_custom",
    "MyCustomDatastore",
    "my_model",
    "MyCustomPredictor",
}


def parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from markdown file."""
    match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    assert match, "SKILL.md must start with YAML frontmatter bounded by '---'"
    return yaml.safe_load(match.group(1))


def extract_python_blocks(content: str) -> list[str]:
    """Extract python code snippets from markdown."""
    pattern = r"```python\n(.*?)```"
    return re.findall(pattern, content, re.DOTALL)


def extract_prose_symbols() -> list[str]:
    """Dynamically extract code symbols mentioned in skill markdown files."""
    symbol_pattern = re.compile(
        r"`((?:neural_lam|tests)\.[a-zA-Z0-9_.]+)(?:\([^`]*\))?`"
    )
    found_symbols = set()
    for skill_file in SKILL_FILES:
        text = skill_file.read_text(encoding="utf-8")
        for match in symbol_pattern.finditer(text):
            found_symbols.add(match.group(1))
    return sorted(list(found_symbols))


def extract_prose_paths() -> list[str]:
    """
    Dynamically extract repo-relative paths mentioned in skill markdown files.
    """
    path_pattern = re.compile(
        r"`((?:neural_lam|tests|docs)/[a-zA-Z0-9_./]+\.[a-zA-Z0-9_]+)`"
    )
    found_paths = set()
    for skill_file in SKILL_FILES:
        text = skill_file.read_text(encoding="utf-8")
        for match in path_pattern.finditer(text):
            found_paths.add(match.group(1))
    return sorted(list(found_paths))


def test_expected_skills_exist():
    """Verify all expected canonical skills exist."""
    assert len(SKILL_FILES) >= len(EXPECTED_SKILLS)
    skill_names = [p.parent.name for p in SKILL_FILES]
    for expected in EXPECTED_SKILLS:
        assert expected in skill_names, f"Missing expected skill: {expected}"


@pytest.mark.parametrize("skill_file", SKILL_FILES, ids=lambda p: p.parent.name)
def test_skill_frontmatter(skill_file: Path):
    """Verify skill frontmatter conforms to Agent Skills specification."""
    content = skill_file.read_text(encoding="utf-8")
    data = parse_frontmatter(content)

    assert "name" in data, f"{skill_file} missing 'name' in frontmatter"
    assert "description" in data, (
        f"{skill_file} missing 'description' in frontmatter"
    )

    skill_name = data["name"]
    dir_name = skill_file.parent.name
    assert skill_name == dir_name, (
        f"Skill name '{skill_name}' does not match directory '{dir_name}'"
    )
    assert 1 <= len(skill_name) <= 64, (
        f"Skill name '{skill_name}' length must be 1 to 64 characters"
    )
    assert re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", skill_name), (
        f"Skill name '{skill_name}' must be kebab-case without consecutive "
        "hyphens"
    )
    assert isinstance(data["description"], str), (
        f"{skill_file} description must be a string"
    )
    desc = data["description"].strip()
    assert 1 <= len(desc) <= 1024, (
        f"{skill_file} description length must be 1 to 1024 characters"
    )


@pytest.mark.parametrize("skill_file", SKILL_FILES, ids=lambda p: p.parent.name)
def test_skill_python_code_blocks(skill_file: Path):
    """Verify Python code snippets in skills are syntactically valid."""
    content = skill_file.read_text(encoding="utf-8")
    blocks = extract_python_blocks(content)

    for i, code in enumerate(blocks):
        try:
            ast.parse(code)
        except SyntaxError as e:
            err_msg = (
                f"Syntax error in {skill_file} block {i + 1}: {e}\n"
                f"Code:\n{code}"
            )
            pytest.fail(err_msg)


@pytest.mark.parametrize("skill_file", SKILL_FILES, ids=lambda p: p.parent.name)
def test_skill_code_block_imports_exist(skill_file: Path):
    """
    Extract and verify that imports in code blocks resolve to real targets.

    Validates both absolute and package-relative imports, while distinguishing
    intentional template placeholders from real project dependencies to
    prevent documentation rot.
    """
    skill_name = skill_file.parent.name
    package_context = SKILL_PACKAGE_CONTEXT.get(skill_name)
    content = skill_file.read_text(encoding="utf-8")
    blocks = extract_python_blocks(content)

    for block in blocks:
        tree = ast.parse(block)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    try:
                        importlib.import_module(alias.name)
                    except ImportError as e:
                        pytest.fail(
                            f"{skill_file} imports unresolvable module "
                            f"'{alias.name}': {e}"
                        )
            elif isinstance(node, ast.ImportFrom):
                module_parts = (node.module or "").split(".")
                is_placeholder = any(
                    part in PLACEHOLDER_NAMES for part in module_parts
                ) or any(
                    alias.name in PLACEHOLDER_NAMES for alias in node.names
                )

                if is_placeholder:
                    # If relative, verify non-placeholder prefix package exists
                    if node.level > 0 and package_context:
                        prefix_parts = []
                        for part in module_parts:
                            if part in PLACEHOLDER_NAMES:
                                break
                            prefix_parts.append(part)
                        if prefix_parts:
                            rel_prefix = "." * node.level + ".".join(
                                prefix_parts
                            )
                            resolved_prefix = importlib.util.resolve_name(
                                rel_prefix, package_context
                            )
                            try:
                                importlib.import_module(resolved_prefix)
                            except ImportError as e:
                                pytest.fail(
                                    f"{skill_file} imports placeholder from "
                                    f"unresolvable package "
                                    f"'{resolved_prefix}': {e}"
                                )
                    continue

                if node.level > 0:
                    if not package_context:
                        rel_str = "." * node.level + (node.module or "")
                        pytest.fail(
                            f"{skill_file} uses relative import '{rel_str}' "
                            "without a declared package context"
                        )
                    rel_module = "." * node.level + (node.module or "")
                    module_name = importlib.util.resolve_name(
                        rel_module, package_context
                    )
                else:
                    module_name = node.module

                try:
                    module = importlib.import_module(module_name)
                except ImportError as e:
                    pytest.fail(
                        f"{skill_file} imports from unresolvable module "
                        f"'{module_name}': {e}"
                    )

                for alias in node.names:
                    if hasattr(module, alias.name):
                        continue
                    # Check if symbol is an unimported submodule
                    submodule_path = f"{module_name}.{alias.name}"
                    try:
                        importlib.import_module(submodule_path)
                    except ImportError:
                        pytest.fail(
                            f"{skill_file} imports missing symbol "
                            f"'{alias.name}' from '{module_name}'"
                        )


@pytest.mark.parametrize("symbol", extract_prose_symbols())
def test_skill_referenced_symbols_exist(symbol: str):
    """
    Verify code symbols in skills resolve to real modules or attributes.
    """
    parts = symbol.split(".")
    try:
        importlib.import_module(symbol)
        return
    except ImportError:
        pass

    module_name = ".".join(parts[:-1])
    attr_name = parts[-1]
    module = importlib.import_module(module_name)
    assert hasattr(module, attr_name), (
        f"Module '{module_name}' has no attribute '{attr_name}'"
    )


@pytest.mark.parametrize("rel_path", extract_prose_paths())
def test_skill_referenced_paths_exist(rel_path: str):
    """Verify file paths referenced in skills exist in the repository."""
    target_path = REPO_ROOT / rel_path
    assert target_path.exists(), (
        f"Referenced path '{rel_path}' does not exist on disk"
    )


@pytest.mark.requires_training
def test_smoke_train_workflow_execution():
    """Verify that verbatim smoke-train workflow executes cleanly."""
    smoke_skill = SKILLS_DIR / "smoke-train" / "SKILL.md"
    content = smoke_skill.read_text(encoding="utf-8")
    blocks = extract_python_blocks(content)
    assert len(blocks) == 1, (
        f"Expected exactly 1 python block in {smoke_skill}, found {len(blocks)}"
    )

    code = blocks[0]
    exec(code, {"__name__": "__main__"})  # noqa: S102

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

SKILLS_DIR = Path(__file__).resolve().parent.parent / ".agents" / "skills"
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

PROSE_REFERENCED_SYMBOLS = [
    ("neural_lam.metrics", "DEFINED_METRICS"),
    ("neural_lam.metrics", "get_metric"),
    ("neural_lam.metrics", "mask_and_reduce_metric"),
    ("neural_lam.loss_weighting", "get_manual_state_feature_weights"),
    ("neural_lam.loss_weighting", "get_uniform_state_feature_weights"),
    ("neural_lam.datastore", "DATASTORES"),
    ("neural_lam.datastore", "DATASTORE_CLASSES"),
    ("neural_lam.datastore.base", "BaseDatastore"),
    ("neural_lam.models", "ForecasterModule"),
    ("neural_lam.models", "MODELS"),
    ("neural_lam.models.step_predictors.base", "StepPredictor"),
    ("neural_lam.models.step_predictors.graph.base", "BaseGraphModel"),
    ("neural_lam.api", "create_graph"),
    ("neural_lam.api", "train"),
    ("neural_lam.api", "evaluate"),
    ("tests.dummy_datastore", "DummyDatastore"),
]


def parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from markdown file."""
    match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    assert match, "SKILL.md must start with YAML frontmatter bounded by '---'"
    return yaml.safe_load(match.group(1))


def extract_python_blocks(content: str) -> list[str]:
    """Extract python code snippets from markdown."""
    pattern = r"```python\n(.*?)```"
    return re.findall(pattern, content, re.DOTALL)


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


@pytest.mark.parametrize(
    ("module_path", "symbol_name"),
    PROSE_REFERENCED_SYMBOLS,
)
def test_skill_prose_referenced_symbols_exist(
    module_path: str, symbol_name: str
):
    """Verify classes and registries mentioned in skill prose exist."""
    module = importlib.import_module(module_path)
    assert hasattr(module, symbol_name), (
        f"Module '{module_path}' has no attribute '{symbol_name}'"
    )


@pytest.mark.requires_training
def test_smoke_train_workflow_execution(tmp_path: Path):
    """Verify that the end-to-end smoke-train workflow executes cleanly."""
    # Standard library
    import os

    # First-party
    from neural_lam import api
    from neural_lam.config import DatastoreSelection, NeuralLAMConfig
    from neural_lam.datastore import DATASTORES
    from tests.dummy_datastore import DummyDatastore

    os.environ["WANDB_MODE"] = "offline"
    DATASTORES[DummyDatastore.SHORT_NAME] = DummyDatastore

    datastore = DummyDatastore(n_grid_points=100, n_timesteps=15)
    config = NeuralLAMConfig(
        datastores={
            "main": DatastoreSelection(
                kind=datastore.SHORT_NAME,
                config_path=str(datastore.root_path),
            )
        }
    )
    api.create_graph(datastore=datastore, name="1level", archetype="keisler")
    runs_root = tmp_path / "runs"
    train_run = api.train(
        datastore=datastore,
        config=config,
        model="graph_lam",
        graph="1level",
        epochs=1,
        ar_steps_train=1,
        val_interval=1,
        num_sanity_val_steps=0,
        batch_size=1,
        num_workers=0,
        hidden_dim=4,
        processor_layers=1,
        hidden_layers=1,
        runs_root=str(runs_root),
        logger_run_name="smoke_train_skill_test",
        val_steps_to_log=[1],
    )
    assert train_run.checkpoint_path is not None
    assert train_run.checkpoint_path.exists()

    eval_run = api.evaluate(
        datastore=datastore,
        config=config,
        load=str(train_run.checkpoint_path),
        model="graph_lam",
        graph="1level",
        hidden_dim=4,
        processor_layers=1,
        hidden_layers=1,
        batch_size=1,
        num_workers=0,
        ar_steps_eval=2,
        val_steps_to_log=[1],
        create_gif=True,
        runs_root=str(runs_root),
        logger_run_name="smoke_eval_skill_test",
    )
    assert eval_run.run_dir.exists()

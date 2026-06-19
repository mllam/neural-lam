# Coding Standards

## Code Style

Our codebase enforces strict styling guidelines using the following tools:
- **Formatter**: `black`
- **Import Sorting**: `isort`
- **Linting**: `flake8`
- **Type Checking**: `mypy`
- **Docstring Coverage**: `interrogate`
- **Spell Checking**: `codespell`

## Pull Request Process

1. **Search before creating**: Search existing issues or PRs to avoid duplicates.
2. **Every PR requires an issue**: Open an issue first if none exists.
3. **Link the issue**: Include `closes #<N>` or `refs #<N>` in the PR body.
4. **Use the PR template**: Fill out every section of the provided template.
5. **Run pre-commit hooks**: Ensure code passes locally via `uvx pre-commit run --all-files`.
6. **Run tests**: Run `pytest tests/` and fix any failures before opening the PR.
7. **Update CHANGELOG**: Add a line to `CHANGELOG.md` in the appropriate section.

## Docstring Standard

- We use **NumPy-style** docstrings.
- We require **100% docstring coverage** for public functions, methods, and classes.
- Always include `Parameters`, `Returns`, and `Raises` sections if applicable.
- Make sure to specify **tensor shapes** in the docstrings when passing or returning tensors.

## Commit Messages

- Must be in the **imperative form** (e.g., "Add test for feature X" instead of "Added test...").
- Keep **one concern per PR** to ensure unrelated changes are not mixed.
- AI attribution of tool names is mandatory if used and should be mentioned in the commit message trailer as `Co-authored-by <tool>`.

## Adding Documentation

When adding new modules or classes:
1. Include detailed docstrings directly in the code.
2. If appropriate, create a new Markdown page under `docs/architecture/` (or update an existing one).
3. Ensure the new page is added to the `_toc.yml` so it appears in the documentation navigation structure.

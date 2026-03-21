# Contributing to Neural-LAM

Thank you for your interest in contributing to Neural-LAM! 🎉

We welcome contributions of all kinds, including bug fixes, documentation improvements, and new features.

---

## 🛠️ Getting Started

### 1. Fork the repository
Click the "Fork" button on the top right of the repository page.

### 2. Clone your fork
```bash
git clone https://github.com/YOUR_USERNAME/neural-lam.git
cd neural-lam
3. Create a new branch
git checkout -b your-branch-name
⚙️ Setup the development environment

We recommend installing in editable mode with development dependencies:

pip install --group dev -e .

Or using uv:

uv pip install --group dev -e .
✅ Before making a PR

Run pre-commit checks:

pre-commit run --all-files

Make sure:

Code is formatted correctly

No linting errors

Tests pass

🚀 Making a Pull Request

Commit your changes:

git add .
git commit -m "Your message"

Push your branch:

git push origin your-branch-name

Open a Pull Request on GitHub

💡 Guidelines

Keep PRs small and focused

Write clear commit messages

Link related issues (e.g., Fixes #123)

Be respectful and collaborative

🙌 Need Help?

Feel free to open an issue or ask questions in the Slack channel.

Happy coding! 🚀


---

## 💾 Step 3: Commit

```bash
git add CONTRIBUTING.md
git commit -m "Add CONTRIBUTING.md for new contributors (Fixes #406)"
git push
🚀 Step 4: Create PR
Title:
Add CONTRIBUTING.md to guide new contributors
Description:
Fixes #406

Added a CONTRIBUTING.md file to guide new contributors on how to set up the project, run checks, and submit pull requests.

This improves onboarding and developer experience.
🔥 After this PR

You will have:

✅ Docs fix (#465)

✅ Docs fix (#464)

✅ New file contribution (#406)

👉 This is VERY strong for GSoC

🚀 Next level (after this)

Then we move to:

small bug fix OR

type hints PR

Say:
👉 “done #406”
or
👉 “error while pushing”

I’ll help instantly 👍

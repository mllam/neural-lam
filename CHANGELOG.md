# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased](https://github.com/joeloskarsson/neural-lam/compare/v0.1.0...HEAD)

### Added
- Added tests for loading dataset, creating graph, and training model based on reduced MEPS dataset stored on AWS S3, along with automatic running of tests on push/PR to GitHub, including push to main branch. Added caching of test data to speed up running tests.
  [\#38](https://github.com/mllam/neural-lam/pull/38) [\#55](https://github.com/mllam/neural-lam/pull/55)
  @SimonKamuk

- Replaced `constants.py` with `data_config.yaml` for data configuration management
  [\#31](https://github.com/joeloskarsson/neural-lam/pull/31)
  @sadamov

- new metrics (`nll` and `crps_gauss`) and `metrics` submodule, stddiv output option
  [c14b6b4](https://github.com/joeloskarsson/neural-lam/commit/c14b6b4323e6b56f1f18632b6ca8b0d65c3ce36a)
  @joeloskarsson

- ability to "watch" metrics and log
  [c14b6b4](https://github.com/joeloskarsson/neural-lam/commit/c14b6b4323e6b56f1f18632b6ca8b0d65c3ce36a)
  @joeloskarsson

- pre-commit setup for linting and formatting
  [\#6](https://github.com/joeloskarsson/neural-lam/pull/6), [\#8](https://github.com/joeloskarsson/neural-lam/pull/8)
  @sadamov, @joeloskarsson

- added github pull-request template to ease contribution and review process
  [\#53](https://github.com/mllam/neural-lam/pull/53), @leifdenby

- ci/cd setup for running both CPU and GPU-based testing both with pdm and pip based installs [\#37](https://github.com/mllam/neural-lam/pull/37), @khintz, @leifdenby

### Changed

  Optional multi-core/GPU support for statistics calculation in `create_parameter_weights.py`
  [\#22](https://github.com/mllam/neural-lam/pull/22)
  @sadamov

- Robust restoration of optimizer and scheduler using `ckpt_path`
  [\#17](https://github.com/mllam/neural-lam/pull/17)
  @sadamov

- Updated scripts and modules to use `data_config.yaml` instead of `constants.py`
  [\#31](https://github.com/joeloskarsson/neural-lam/pull/31)
  @sadamov

- Added new flags in `train_model.py` for configuration previously in `constants.py`
  [\#31](https://github.com/joeloskarsson/neural-lam/pull/31)
  @sadamov

- moved batch-static features ("water cover") into forcing component return by `WeatherDataset`
  [\#13](https://github.com/joeloskarsson/neural-lam/pull/13)
  @joeloskarsson

- change validation metric from `mae` to `rmse`
  [c14b6b4](https://github.com/joeloskarsson/neural-lam/commit/c14b6b4323e6b56f1f18632b6ca8b0d65c3ce36a)
  @joeloskarsson

- change RMSE definition to compute sqrt after all averaging
  [\#10](https://github.com/joeloskarsson/neural-lam/pull/10)
  @joeloskarsson

### Removed

- `WeatherDataset(torch.Dataset)` no longer returns "batch-static" component of
  training item (only `prev_state`, `target_state` and `forcing`), the batch static features are
  instead included in forcing
  [\#13](https://github.com/joeloskarsson/neural-lam/pull/13)
  @joeloskarsson

### Maintenance

- simplify pre-commit setup by 1) reducing linting to only cover static
  analysis excluding imports from external dependencies (this will be handled
  in build/test cicd action introduced later), 2) pinning versions of linting
  tools in pre-commit config (and remove from `requirements.txt`) and 3) using
  github action to run pre-commit.
  [\#29](https://github.com/mllam/neural-lam/pull/29)
  @leifdenby

- change copyright formulation in license to encompass all contributors
  [\#47](https://github.com/mllam/neural-lam/pull/47)
  @joeloskarsson

- Fix incorrect ordering of x- and y-dimensions in comments describing tensor
  shapes for MEPS data
  [\#52](https://github.com/mllam/neural-lam/pull/52)
  @joeloskarsson

- Cap numpy version to < 2.0.0 (this cap was removed in #37, see below)
  [\#68](https://github.com/mllam/neural-lam/pull/68)
  @joeloskarsson

- Remove numpy < 2.0.0 version cap
  [\#37](https://github.com/mllam/neural-lam/pull/37)
  @leifdenby

- turn `neural-lam` into a python package by moving all `*.py`-files into the
  `neural_lam/` source directory and updating imports accordingly. This means
  all cli functions are now invoke through the package name, e.g. `python -m
  neural_lam.train_model` instead of `python train_model.py` (and can be done
  anywhere once the package has been installed).
  [\#32](https://github.com/mllam/neural-lam/pull/32), @leifdenby

- move from `requirements.txt` to `pyproject.toml` for defining package dependencies.
  [\#37](https://github.com/mllam/neural-lam/pull/37), @leifdenby

## [v0.1.0](https://github.com/joeloskarsson/neural-lam/releases/tag/v0.1.0)

First tagged release of `neural-lam`, matching Oskarsson et al 2023 publication
(<https://arxiv.org/abs/2309.17370>)

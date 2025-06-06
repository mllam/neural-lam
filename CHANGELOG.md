# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased](https://github.com/mllam/neural-lam/compare/v0.3.0...HEAD)

### Added

- Add support for MLFlow logging and metrics tracking. [\#77](https://github.com/mllam/neural-lam/pull/77)
  @khintz

- Add support for multi-node training.
[\#103](https://github.com/mllam/neural-lam/pull/103) @simonkamuk @sadamov

- Add option to clamp output prediction using limits specified in config file [\#92](https://github.com/mllam/neural-lam/pull/92) @SimonKamuk

### Fixed
- Only print on rank 0 to avoid duplicates of all print statements.
[\#103](https://github.com/mllam/neural-lam/pull/103) @simonkamuk @sadamov

- Fix MLFlow exception import introduced in [\#77](https://github.com/mllam/neural-lam/pull/77).
  [\#111](https://github.com/mllam/neural-lam/pull/111)
  @observingClouds

- Fix duplicate tensor copy to CPU [\#106](https://github.com/mllam/neural-lam/pull/106) @observingClouds

- Fix bug where the inverse_softplus used in clamping caused nans in the gradients [\#123](https://github.com/mllam/neural-lam/pull/123) @SimonKamuk

- Add standardization to state diff stats from mdp datastore [\#122](https://github.com/mllam/neural-lam/pull/122) @SimonKamuk

- Set ci/cd badges to refer to the new test matrix [\#130](https://github.com/mllam/neural-lam/pull/130) @SimonKamuk

### Maintenance
- update AWS GPU ci/cd to use ami with larger (200GB) root volume and ensure
  nvme drive is used for pip venvn
  [\#126](https://github.com/mllam/neural-lam/pull/126), @leifdenby

- update ci/cd testing setup to install torch version compatible with neural-lam
  dependencies [\#115](https://github.com/mllam/neural-lam/pull/115), @leifdenby

- switch to new npyfiles MEPS and mdp DANRA test datasets which are coincident
  in time and space (on cropped ~100x100 grid-point domain)
  [\#110](https://github.com/mllam/neural-lam/pull/110), @leifdenby

- use dynamic versioning based on git tags and commit hashes
  [\#118](https://github.com/mllam/neural-lam/pull/118), @observingClouds

 - add detect_anomaly=True to pl.Trainer in test_training.py [\#124](https://github.com/mllam/neural-lam/pull/124), @SimonKamuk

## [v0.3.0](https://github.com/mllam/neural-lam/releases/tag/v0.3.0)

This release introduces Datastores to represent input data from different sources (including zarr and numpy) while keeping graph generation within neural-lam.

### Added

- Introduce Datastores to represent input data from different sources, including zarr and numpy.
  [\#66](https://github.com/mllam/neural-lam/pull/66)
 @leifdenby @sadamov

- Implement standardization of static features when loaded in ARModel [\#96](https://github.com/mllam/neural-lam/pull/96) @joeloskarsson

### Fixed

- Fix wandb environment variable disabling wandb during tests. Now correctly uses WANDB_MODE=disabled. [\#94](https://github.com/mllam/neural-lam/pull/94) @joeloskarsson

- Fix bugs introduced with datastores functionality relating visualation plots [\#91](https://github.com/mllam/neural-lam/pull/91) @leifdenby

## [v0.2.0](https://github.com/mllam/neural-lam/releases/tag/v0.2.0)

### Added
- Added tests for loading dataset, creating graph, and training model based on reduced MEPS dataset stored on AWS S3, along with automatic running of tests on push/PR to GitHub, including push to main branch. Added caching of test data to speed up running tests.
  [\#38](https://github.com/mllam/neural-lam/pull/38) [\#55](https://github.com/mllam/neural-lam/pull/55)
  @SimonKamuk

- Replaced `constants.py` with `data_config.yaml` for data configuration management
  [\#31](https://github.com/mllam/neural-lam/pull/31)
  @sadamov

- new metrics (`nll` and `crps_gauss`) and `metrics` submodule, stddiv output option
  [c14b6b4](https://github.com/mllam/neural-lam/commit/c14b6b4323e6b56f1f18632b6ca8b0d65c3ce36a)
  @joeloskarsson

- ability to "watch" metrics and log
  [c14b6b4](https://github.com/mllam/neural-lam/commit/c14b6b4323e6b56f1f18632b6ca8b0d65c3ce36a)
  @joeloskarsson

- pre-commit setup for linting and formatting
  [\#6](https://github.com/mllam/neural-lam/pull/6), [\#8](https://github.com/mllam/neural-lam/pull/8)
  @sadamov, @joeloskarsson

- added github pull-request template to ease contribution and review process
  [\#53](https://github.com/mllam/neural-lam/pull/53), @leifdenby

- ci/cd setup for running both CPU and GPU-based testing both with pdm and pip based installs [\#37](https://github.com/mllam/neural-lam/pull/37), @khintz, @leifdenby

### Changed

- Clarify routine around requesting reviewer and assignee in PR template
  [\#74](https://github.com/mllam/neural-lam/pull/74)
  @joeloskarsson

- Argument Parser updated to use action="store_true" instead of 0/1 for boolean arguments.
  (https://github.com/mllam/neural-lam/pull/72)
  @ErikLarssonDev

-  Optional multi-core/GPU support for statistics calculation in `create_parameter_weights.py`
  [\#22](https://github.com/mllam/neural-lam/pull/22)
  @sadamov

- Robust restoration of optimizer and scheduler using `ckpt_path`
  [\#17](https://github.com/mllam/neural-lam/pull/17)
  @sadamov

- Updated scripts and modules to use `data_config.yaml` instead of `constants.py`
  [\#31](https://github.com/mllam/neural-lam/pull/31)
  @sadamov

- Added new flags in `train_model.py` for configuration previously in `constants.py`
  [\#31](https://github.com/mllam/neural-lam/pull/31)
  @sadamov

- moved batch-static features ("water cover") into forcing component return by `WeatherDataset`
  [\#13](https://github.com/mllam/neural-lam/pull/13)
  @joeloskarsson

- change validation metric from `mae` to `rmse`
  [c14b6b4](https://github.com/mllam/neural-lam/commit/c14b6b4323e6b56f1f18632b6ca8b0d65c3ce36a)
  @joeloskarsson

- change RMSE definition to compute sqrt after all averaging
  [\#10](https://github.com/mllam/neural-lam/pull/10)
  @joeloskarsson

### Removed

- `WeatherDataset(torch.Dataset)` no longer returns "batch-static" component of
  training item (only `prev_state`, `target_state` and `forcing`), the batch static features are
  instead included in forcing
  [\#13](https://github.com/mllam/neural-lam/pull/13)
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

- Add slack and new publication info to readme
  [\#78](https://github.com/mllam/neural-lam/pull/78)
  @joeloskarsson

## [v0.1.0](https://github.com/mllam/neural-lam/releases/tag/v0.1.0)

First tagged release of `neural-lam`, matching Oskarsson et al 2023 publication
(<https://arxiv.org/abs/2309.17370>)

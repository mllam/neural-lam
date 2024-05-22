# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased](https://github.com/joeloskarsson/neural-lam/compare/v0.1.0...HEAD)

### Added

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

### Changed

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

## [v0.1.0](https://github.com/joeloskarsson/neural-lam/releases/tag/v0.1.0)

First tagged release of `neural-lam`, matching Oskarsson et al 2023 publication
(<https://arxiv.org/abs/2309.17370>)

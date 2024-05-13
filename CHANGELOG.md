# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [unrelease](https://github.com/joeloskarsson/neural-lam/compare/v0.1.0...HEAD)

### Added

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

- moved batch-static features ("water cover") into forcing component return by `WeatherDataset`
  [\#13](https://github.com/joeloskarsson/neural-lam/pull/13)
  @joeloskarsson

- change validation metric from `mae` to `rmse`
  [c14b6b4](https://github.com/joeloskarsson/neural-lam/commit/c14b6b4323e6b56f1f18632b6ca8b0d65c3ce36a)
  @joeloskarsson

- compute `rmse` after spatial averaging
  [\#10](https://github.com/joeloskarsson/neural-lam/pull/10)
  @joeloskarsson

### Removed

- WeatherDataset(torch.Dataset) no longer returns static component, static is
  instead included in forcing
  [\#13](https://github.com/joeloskarsson/neural-lam/pull/13)
  @joeloskarsson


## [v0.1.0](https://github.com/joeloskarsson/neural-lam/releases/tag/v0.1.0)

First tagged release of `neural-lam`, matching Oscarsson et al 2023 publication
(https://arxiv.org/abs/2309.17370)

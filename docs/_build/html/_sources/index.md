# Neural-LAM Documentation

Neural-LAM is a repository of graph-based neural weather prediction models for
Limited Area Modeling (LAM). It implements models from recent research using
[PyTorch](https://pytorch.org/), [PyTorch Lightning](https://lightning.ai/pytorch-lightning),
and [PyTorch Geometric](https://pyg.org/).

The repository contains LAM versions of:

- The graph-based model from [Keisler (2022)](https://arxiv.org/abs/2202.07575)
- GraphCast, by [Lam et al. (2023)](https://arxiv.org/abs/2212.12794)
- The hierarchical model from [Oskarsson et al. (2023)](https://arxiv.org/abs/2309.17370)

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting_started
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user_guide/data
user_guide/graphs
user_guide/training
user_guide/evaluation
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Development

contributing
```

## Citing Neural-LAM

If you use Neural-LAM in your work, please cite the relevant paper(s):

```bibtex
@inproceedings{oskarsson2023graphbased,
    title={Graph-based Neural Weather Prediction for Limited Area Modeling},
    author={Oskarsson, Joel and Landelius, Tomas and Lindsten, Fredrik},
    booktitle={NeurIPS 2023 Workshop on Tackling Climate Change with ML},
    year={2023}
}
```

```bibtex
@inproceedings{oskarsson2024probabilistic,
    title={Probabilistic Weather Forecasting with Hierarchical Graph Neural Networks},
    author={Oskarsson, Joel and Landelius, Tomas and Deisenroth, Marc Peter and Lindsten, Fredrik},
    booktitle={Advances in Neural Information Processing Systems},
    volume={37},
    year={2024},
}
```

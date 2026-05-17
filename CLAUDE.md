# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a PyTorch research framework for **clean-label data poisoning via gradient matching**. The attack crafts imperceptible perturbations on training images so that a model trained on the poisoned dataset misclassifies specific target images. See [arXiv:2009.02276](https://arxiv.org/abs/2009.02276).

## Dependencies

```
PyTorch >= 1.6
torchvision > 0.5
efficientnet_pytorch  # only if using EfficientNet
python-lmdb           # only if using LMDB datasets
```

## Running Experiments

**Basic run (CIFAR10, ResNet18, default settings):**
```bash
python brew_poison.py
```

**Reproduce paper validation set:**
```bash
python brew_poison.py --vruns 8 --poisonkey 2000000000
```

**ImageNet example:**
```bash
python brew_poison.py --net ResNet34 --eps 8 --budget 0.001 --pretrained --dataset ImageNet --data_path /your/path/to/ImageNet --pbatch 128
```

**Distributed (multi-GPU, single node):**
```bash
python -m torch.distributed.launch --nproc_per_node=<N_GPUS> --master_port=20704 dist_brew_poison.py --extra_args
```

**Dry run (fast smoke test):**
```bash
python brew_poison.py --dryrun
```

**Sanity check (validate exported poisons independently):**
```bash
# First export poisons, then:
python tests/sanity_check.py
```

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--net` | `ResNet18` | Model(s) to poison on (comma-separated for ensemble) |
| `--dataset` | `CIFAR10` | Dataset: `CIFAR10`, `CIFAR100`, `MNIST`, `ImageNet`, `TinyImageNet` |
| `--recipe` | `gradient-matching` | Attack: `gradient-matching`, `gradient-matching-private`, `watermark`, `poison-frogs`, `metapoison`, `bullseye` |
| `--eps` | `16` | Perturbation bound (in pixel space / 255) |
| `--budget` | `0.01` | Fraction of training data to poison |
| `--ensemble` | `1` | Number of surrogate models for transfer |
| `--threatmodel` | `single-class` | `single-class`, `third-party`, `random-subset` |
| `--poisonkey` | `None` | Fix the random setup for reproducibility |
| `--vruns` | `1` | Validation re-runs with fresh model init |
| `--save` | `None` | Export format: `full`, `limited`, `automl`, `numpy` |

Full argument list: `forest/options.py`.

## Architecture

The framework has three core abstractions, each instantiated via a factory function in `forest/__init__.py`:

### `Kettle` (`forest/data/kettle.py`)
Data manager. Holds `trainloader`, `validloader`, `poisonloader`, `poisonset`, `targetset`, and `poison_lookup` (maps image IDs → slice index in `poison_delta`). Handles poison setup construction (random, deterministic, or benchmark). The `poison_delta` tensor stores **only the adversarial perturbation**, not the full image — perturbations are added on-the-fly during batches.

### `Victim` (`forest/victims/`)
Training and evaluation backend. Factory selects among:
- `_VictimSingle` — single GPU, single model
- `_VictimEnsemble` — multiple models on one node
- `_VictimDistributed` — multi-node via `torch.distributed`

Key methods: `train()`, `retrain()`, `validate()`, `gradient()`, `compute()`, `step()`.

### `Witch` (`forest/witchcoven/`)
Implements the poisoning attack. Factory selects the recipe class. All iterative attacks inherit from `_Witch` (`witch_base.py`), which runs the outer restart loop in `_brew()`. Subclasses override `_define_objective()` to define the loss closure, or override `_brew()` entirely for non-iterative methods.

**Attack flow in `brew_poison.py`:**
1. `model.train(data)` — train surrogate victim (skip if `--pretrained`)
2. `witch.brew(model, data)` → `poison_delta` — compute perturbations
3. `model.validate(data, poison_delta)` — evaluate attack success

## Adding a New Attack

1. Create `forest/witchcoven/witch_<name>.py` subclassing `_Witch`
2. Override `_define_objective()` (for iterative attacks) or `_brew()` (for non-iterative)
3. Register the class in `forest/witchcoven/__init__.py` and add it to the `--recipe` choices in `forest/options.py`

## Reproducibility

- Results vary by target/source class selection. Use `--poisonkey` to fix the setup (a plain integer seeds random selection; a dash-separated triplet like `5-3-1` selects deterministically).
- Use `--modelkey` to fix model initialization; add `--deterministic` to enable cuDNN deterministic mode.
- Results are logged to `tables/` as CSV; exported poisons go to `poisons/`.

# Part 1: Deep Context-Dependent Choice Model (DeepHalo) — JPM MLCOE Exercise

This repository contains the Part 1 implementation of Zhang et al. (2025) **DeepHalo** in TensorFlow/TensorFlow Probability style, plus synthetic experiments and tests.

## Quick start

### 1) Create environment

Recommended (conda):

```bash
conda env create -f environment.yml
conda activate q3-part1  # or the name defined in the yml
```

On Apple Silicon, you may prefer `environment_mac.yaml`.

### 2) Run tests

```bash
pytest -q
```

Expected: all tests pass.

### 3) Run all Part 1 experiments

```bash
python run_all.py
```

This runs the synthetic replications and writes outputs under:

- `results/` (tables, logs, csv)
- `results/figures/` (all plots)

## Outputs

### Figures (saved to `results/figures/`)
- `attraction_effect_probs_tf.png`
- `attraction_effect_probs_torch.png`
- `decoy_effect_probs.png`
- `heatmap_target.png`
- `heatmap_predicted.png`
- `influence_matrix.png`

### Data / logs (saved to `results/`)
- `table1_predictions.csv`
- `influence_matrix.csv`
- `compromise_effect_tf.txt`
- `run_all.log`

## Authors-mode replication

By default, the TensorFlow model supports two internal variants:

- **authors-mode**: ports the authors' PyTorch algorithm (multiplicative residual updates).
- **simplified-mode**: an earlier permutation-equivariant halo block approximation.

See `docs/IMPLEMENTATION_AUTHORS_MODE.md` for a detailed mapping between the authors' PyTorch code and the TensorFlow implementation.

## How to run individual experiments

Each experiment is a Python module under `experiments/`. For example:

```bash
python -m experiments.reproduce_table1
python -m experiments.decoy_effect
python -m experiments.attraction_effect_tf
python -m experiments.attraction_effect_torch
python -m experiments.compromise_effect_tf
python -m experiments.influence_map
```

## Notes on paths

All experiments use `experiments/paths.py`:

- `results_dir()` → `results/`
- `figures_dir()` → `results/figures/`

If you want to redirect outputs (e.g., in CI), set:

```bash
export PART1_RESULTS_DIR=/path/to/output
```

## Troubleshooting

- If you see a Keras warning about `build()` not being implemented, it is safe to ignore for this project.
- If `attraction_effect_torch` fails due to missing PyTorch dependencies, you can still complete Part 1 with the TensorFlow experiments; the Torch experiment is included as an optional cross-framework parity check.

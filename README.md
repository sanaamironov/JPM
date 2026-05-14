# JPM — MLCOE Internship Exercises

<p align="center">
  <a href="https://www.linkedin.com/in/sanaamironov/">
    <img src="https://img.shields.io/badge/Sanaa_Mironov-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="Sanaa Mironov">
  </a>
</p>

This repository contains my submission for the JPM MLCOE take-home assignment (Question 3: Discrete Choice + Credit Card Offers). It is organized for **reviewer reproducibility**: a single install path, a quick smoke test, and pre-computed outputs for each part.

**Python 3.10 is required** (`>=3.10,<3.11`). TensorFlow 2.16.2 does not support Python 3.11+.

## Reviewer Quickstart

### Option A — macOS Apple Silicon (tested end-to-end)

```bash
conda deactivate || true
conda create -n jpm-clean python=3.10 -y
conda activate jpm-clean
python -m pip install -U pip

# TensorFlow stack (Apple Silicon — must install before the package)
python -m pip install "tensorflow-macos==2.16.2" "tensorflow==2.16.2" "tensorflow-probability[tf]==0.24.*"

# Install package + dev deps
python -m pip install -e ".[dev]"

# Run all tests (107 tests, ~60 s)
pytest -q

# Part 2 smoke run (all DGPs, ~5 min)
jpmq3-replicate-lu25 --smoke --out results/part2/lu25_smoke --n-jobs 1
```

### Option B — Linux / Windows / non-Apple macOS (Python 3.10 venv)

```bash
python3.10 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"

# Run all tests (107 tests, ~60 s)
pytest -q

# Part 2 smoke run
jpmq3-replicate-lu25 --smoke --out results/part2/lu25_smoke --n-jobs 1
```

> **Note:** On Linux/Windows `tensorflow==2.16.2` is installed automatically from PyPI. On non-Apple macOS Intel you may need to install `tensorflow-macos` manually (same as Option A).

## What this submission contains

### Part 1 — DeepHalo (Zhang 2025)

Context-dependent choice model implemented in TensorFlow/Keras under `src/choice_learn_ext/models/deep_context/`. Captures halo effects via masked context aggregation and runs natively in Keras graph mode (`model.compile` + `model.fit`).

```bash
# Install plotting extra (needed for figure output)
python -m pip install -e ".[dev,part1]"

# Smoke run
jpmq3-part1-experiments --smoke --out results/part1/part1_smoke

# Selected experiments
jpmq3-part1-experiments --only reproduce_table1,decoy_effect --out results/part1/part1_selected
```

Outputs write to `results/part1/<run_name>/`. Figures go under `results/part1/<run_name>/figures/`.

---

### Part 2 — BLP + MCMC Shrinkage (Lu & Shimizu 2025)

Approximate reimplementation of the Section 4 simulation study: BLP contraction mapping + TFP `RandomWalkMetropolis` on a collapsed spike-and-slab model.

```bash
# Smoke run (all 4 DGPs at T=25 J=15, 5 reps each, ~5 min)
jpmq3-replicate-lu25 --smoke --out results/part2/lu25_smoke --n-jobs 1

# Single cell, more reps
jpmq3-replicate-lu25 \
  --out results/part2/lu25_onecell \
  --grid DGP2:25:15 \
  --n-reps 10 \
  --R-mc 50 \
  --n-jobs 1

# Aggregate tables across a run directory
jpmq3-format-lu25-tables --in results/part2/lu25_onecell
```

Each grid cell writes:
- `paper_table_like.csv` — Bias/SD rows in paper layout
- `summary.csv` — long-format metrics
- `config.json` — true parameters + metadata

Pre-computed 10-rep results matching the report tables are tracked under `results/part2/lu25_section4_rep10/` (`paper_table_like.csv`, `summary.csv`, `config.json` for each DGP). An additional wider grid run (T=25 and T=100) is stored in `results/part2/lu25_fullgrid_10rep/` (config only; CSVs not tracked) but was not used for the report.

---

### Part 4 — Zhang-Sparse Hybrid

`ZhangSparseDeepHalo` augments the DeepHalo backbone with a Lu-style sparse shock layer. Estimated by MAP with an ℓ1 penalty on the demand shocks via a two-stage procedure (Stage 1: Halo + market mean; Stage 2: freeze Halo, train market mean + sparse shocks).

Results pre-computed under `results/hybrid/zhang_lu_sparse/`.

---

### Bonus 1 — Dynamic Storable Goods (Ching 2020)

Dynamic discrete choice model with stockpiling and endogenous prices. Consumers solve exact backward induction (pure NumPy DGP); estimation uses a neural continuation-value approximation in TF graph mode with a Petrin & Train (2010) control function for price endogeneity. Includes a simulation study with parameter recovery and Laplace credible intervals.

```bash
# Demo run (fits model, runs counterfactual, saves to results/bonus1/dynamic_model/)
jpmq3-run-bonus1

# Simulation study (parameter recovery, ~10 min)
python -m jpm_q3.bonus1.dynamic_model.simulation_study
```

Results pre-computed under `results/bonus1/`.

---

### ChoiceLearn comparison (optional)

Not required for evaluation. Runs the Lu & Shimizu Section 4 setup through the choice-learn library for comparison.

```bash
python -m pip install -e ".[dev,choicelearn]"
jpmq3-run-lu25-choicelearn --help
```

## Repository map

```
src/
  choice_learn_ext/models/
    deep_context/          # Part 1: DeepHalo model (TF/Keras)
    lu25_sparse_shocks/    # ChoiceLearn comparison runner (optional)
  jpm_q3/
    cli/                   # CLI entry points
    lu25/                  # Part 2: BLP + MCMC shrinkage (Lu & Shimizu 2025)
    zhang25/               # Part 1: experiment scripts
    hybrid/                # Part 4: Zhang-Sparse hybrid
    bonus1/dynamic_model/  # Bonus 1: dynamic storable goods model
tests/
  part1/    # DeepHalo unit tests (masking, equivariance, graph mode, dtype, NLL)
  part2/    # BLP, shrinkage, hybrid tests
  bonus/    # Bonus 1 DGP, model, training, counterfactual, coverage tests
results/    # Pre-computed outputs (kept for reviewer convenience)
Report.pdf  # Write-up, methodology, results (66 pages)
```

## Troubleshooting

- **Python version:** use exactly Python 3.10. TF 2.16.2 is not compatible with 3.11+.
- **Multiprocessing on macOS:** the replication driver uses the `spawn` start method. Always pass `--n-jobs 1` (and optionally `--threads-per-job 1`) to avoid oversubscription on Apple Silicon.
- **TF log noise:** the CLI suppresses the TF device banner by default. Set `JPM_TF_LOG_LEVEL=0` to see full TF logs.
- **Reproducibility issues:** the most helpful info is the full command, terminal output (stderr + stdout), Python version, and OS.

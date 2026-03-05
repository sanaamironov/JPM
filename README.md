# JPM - MLCOE Internship Exercises
<p align="center">
  <a href="https://github.com/sanaamironov/JPM/">
  </a>
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/sanaamironov/">
    <img src="https://img.shields.io/badge/Sanaa_Mironov-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="Sanaa Mironov">
  </a>
</p>

# JPM Take-Home: Discrete Choice / Demand Estimation (Part 1 + Part 2 +  Bonus Questiopn)

This repository contains my submission for the JPM take-home assignment. It is organized for **reviewer reproducibility**: there is a single “golden path” that installs, runs a quick smoke test, and produces the expected output tables.

## Reviewer Quickstart

### Option A (recommended on macOS Apple Silicon): Conda + pinned TF stack

This is the setup I tested end-to-end (install, tests, Part 2 smoke).

```bash
conda deactivate || true
conda create -n jpm-clean python=3.10 -y
conda activate jpm-clean
python -m pip install -U pip

# TensorFlow stack (Apple Silicon)
python -m pip install "tensorflow-macos==2.16.2" "tensorflow==2.16.2" "tensorflow-probability[tf]==0.24.*"

# Install package + dev deps
python -m pip install -e ".[dev]"

# Tests
pytest -q

# Part 2 smoke run
jpmq3-replicate-lu25 --smoke --out results/part2/lu25_smoke
```

### Option B (generic Python venv)

If you are not on Apple Silicon macOS, you can usually do:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest -q

jpmq3-replicate-lu25 --smoke --out results/part2/lu25_smoke
```

## Where to look for outputs

### Part 2 outputs

After the smoke run, open:

* `results/part2/lu25_smoke/DGP*_T*_J*/paper_table_like.csv`
* `results/part2/lu25_smoke/DGP*_T*_J*/summary.csv`

Each grid cell writes:

* `paper_table_like.csv`: Bias/SD rows in a paper-like layout
* `summary.csv`: long-format metrics
* `config.json`: true parameters + metadata

### Part 1 outputs

Part 1 experiments write under `results/part1/<run_name>/` and may write figures under `results/part1/<run_name>/figures/`.

## What this submission contains

### Part 2 (main replication deliverable)

Replication driver for Lu & Shimizu (2025) Section 4 simulation study.

* CLI entry point: `jpmq3-replicate-lu25`
* Canonical driver: `src/jpm_q3/lu25/experiments/replicate_section4.py`
* Table aggregation: `jpmq3-format-lu25-tables`

Run a single grid cell (more substantial than smoke, still reasonable):

```bash
jpmq3-replicate-lu25 \
  --out results/part2/lu25_onecell \
  --grid DGP2:25:15 \
  --n-reps 10 \
  --R-mc 50 \
  --n-jobs 1
```

Aggregate tables across all cells under a run directory:

```bash
jpmq3-format-lu25-tables --in results/part2/lu25_onecell
```

### Part 1 (DeepHalo / context-dependent model)

Part 1 code lives in two places, the model is under `src//choice_learn_ext/models/deep_context` and the experiments are under `src/jpm_q3/zhang25/` and test are `test/part1`. The report describes the model, training, and evaluation.

To run Part 1 experiments, install the Part 1 extra (for plotting) and run the Part 1 CLI:

```bash
python -m pip install -e ".[dev,part1]"

# Smoke run
jpmq3-part1-experiments --smoke --out results/part1/part1_smoke

# Selected experiments
jpmq3-part1-experiments --only reproduce_table1,decoy_effect --out results/part1/part1_selected
```

### Optional extensions (not part of the core evaluation)

This repo may include optional “bonus” code. It is presented as a **prototype / minimal working example** (small-scale simulated demonstration) and is not required to evaluate Parts 1–2.

* ChoiceLearn comparison runner (optional):

  ```bash
  python -m pip install -e ".[dev,choicelearn]"
  jpmq3-run-lu25-choicelearn --help
  ```

## Repository map

* `src/jpm_q3/`
  * `cli/` — command-line entry points
  * `lu25/` — Part 2 replication (Lu & Shimizu 2025)
  * `zhang25/` — Part 1 experiments and utilities
  * `bonus1/` — Bonus extension (prototype)
  * `hybrid/` — hybrid / extension experiments
*  `src/src/choice_learn_ext/` - ChoiceLearn extensions (used only if running the ChoiceLearn CLI)
  * `models/`
      * `lu25_sparse_shocks/` — ChoiceLearn implementation of Lu & Shimizu (2025) Section 4 setup / comparison runner
      * `deep_content/` — Deep-content / deep context-dependent model implementation used in Part 1
* `tests/` — unit tests
* `results/` — precomputed outputs (kept for reviewer convenience)
* `Report.pdf` — write-up / methodology / results

## Troubleshooting

* **Multiprocessing:** for macOS, the replication driver uses the `spawn` start method for safety. If you set `--n-jobs > 1`, also set `--threads-per-job=1` to avoid oversubscription.
* **TensorFlow log noise:** the CLI wrapper suppresses TF device banner logs by default. To see full TF logs, set `JPM_TF_LOG_LEVEL=0` before running.

## Notes

If you hit a reproducibility issue, the most helpful info is the full command used and the terminal output (stderr/stdout), plus Python version and OS.

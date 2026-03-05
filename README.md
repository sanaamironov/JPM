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

## MLCOE Internship Exercises (Question 3)

This repository contains my implementation and report artifacts for **Question 3 (Discrete choice models & credit card offers)**:

- **Part 1:** Zhang (2025) Deep Context-Dependent Choice (DeepHalo) in TensorFlow, integrated in a `choice-learn`-style API, with unit tests and synthetic experiments.
- **Part 2:** Lu & Shimizu (2025) sparse market–product shocks, including **BLP benchmarks** and a **shrinkage estimator implemented with `tfp.mcmc`**, plus paper-style replication outputs.
- **Bonus 1:** A *toy* dynamic discrete choice demo for storable goods/stockpiling (context + sparse shocks + continuation value baseline), with smoke tests and saved artifacts.

All runnable entry points write outputs under `results/` by default.
# JPM Take-Home: Discrete Choice / Demand Estimation (Part 1 + Part 2)

## Reviewer Quickstart 
**Recommended:** Python 3.10 on macOS/Linux.

```bash
conda deactivate || true
# 1) Create and activate a virtualenv, I use Conda
conda create -n jpm-mironov python=3.10 -y
#python -m venv .venv
conda activate jpm-mironov
python -m pip install -U pip
python -m pip install "tensorflow-macos==2.16.2" "tensorflow==2.16.2" "tensorflow-probability==0.24.*"

# 2) Install the package (editable) + dev deps
python -m pip install -e ".[dev]"

# 3) Run unit tests (fast)
pytest -q

# 4) Part 2 (Lu & Shimizu 2025, Section 4) smoke run
jpmq3-replicate-lu25 --smoke --out results/part2/lu25_smoke
```

### Where to look for outputs

After the smoke run, open:

* `results/part2/lu25_smoke/DGP*_T*_J*/paper_table_like.csv`
* `results/part2/lu25_smoke/DGP*_T*_J*/summary.csv`

Each grid cell writes:

* `paper_table_like.csv`: Bias/SD rows in a paper-like layout
* `summary.csv`: long-format metrics
* `config.json`: true parameters + metadata

## What this submission contains
```bash
#To run all the different parts of the project you can use the scripts below
[project.scripts]
jpmq3-run-all = "jpm_q3.cli.run_all:main"
jpmq3-replicate-lu25 = "jpm_q3.cli.replicate_lu25:main"
jpmq3-format-lu25-tables = "jpm_q3.lu25.experiments.format_section4_tables:main"
jpmq3-run-lu25-choicelearn = "choice_learn_ext.models.lu25_sparse_shocks.lu25_section4_choicelearn:main"
jpmq3-run-bonus1 = "jpm_q3.bonus1.dynamic_model.run_demo:main"
```
### Part 2 (Main replication deliverable)

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

Part 1 code lives under `src/jpm_q3/deephalo/` (and extensions under `src/choice_learn_ext/` if present). The report describes the model, training, and evaluation choices.

### Optional extensions (not part of the core evaluation)

This repo may include optional “bonus” code. It is presented as a **prototype / minimal working example** (small-scale simulated demonstration), and is not required to evaluate Parts 1–2.

## Repository map

* `src/jpm_q3/`

  * `cli/` — command-line entry points
  * `lu25/` — Part 2 replication (Lu & Shimizu 2025)
  * `deephalo/` — Part 1 model code
* `tests/` — unit tests
* `results/` — precomputed outputs (kept for reviewer convenience)
* `Report.pdf` — write-up / methodology / results

## Installation notes and dependency stability

This project uses scientific Python + ML dependencies (notably TensorFlow / TensorFlow Probability). If you encounter installation issues on your machine, the fastest workaround is to use a clean Python 3.10 environment.

If you want stricter reproducibility, pin versions using a `requirements.txt`/lockfile. For review, the Quickstart path above is the intended route.

## Troubleshooting

* **Multiprocessing:** for macOS, the replication driver uses the `spawn` start method for safety. If you set `--n-jobs > 1`, also set `--threads-per-job=1` to avoid oversubscription.
* **TensorFlow log noise:** the CLI wrapper suppresses TF device banner logs by default. To see full TF logs, set `JPM_TF_LOG_LEVEL=0` before running.

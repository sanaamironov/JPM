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

---

## 1) Environment Setup (recommended)

### Option A: Conda (Python 3.10)
```bash
conda create -n jpm python=3.10 -y
conda activate jpm
pip install -e ".[dev]"
```

### Option B: venv (Python 3.10)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```
Notes
This repo targets Python 3.10.
On macOS Apple Silicon, TensorFlow uses the Metal backend; you may see Metal device logs.

### Option C: Installed CLI commands
  ```bash
   After pip install -e ., these commands are available:
  
   jpmq3-run-all — runs the end-to-end exercise suite (supports --smoke)

  jpmq3-replicate-lu25 — Lu(25) Section 4 replication driver

  jpmq3-format-lu25-tables — formats replication outputs into paper-style LaTeX tables

  jpmq3-run-lu25-choicelearn — optional choice-learn extension runner (if present)
```


## 2) Run Unit Tests
Run everything:
```bash
python -m unittest discover -s tests -p "test*.py" -v
```
If you have a separate tests/bonus/ folde

```bash
python -m unittest discover -s tests/bonus -p "test*.py" -v
```

## 3) Part 1 — DeepHalo (Zhang 2025)
What’s implemented
 - choice_learn_ext.models.deep_context.DeepHalo supports:
 - authors_mode=True: TensorFlow port of the authors’ released design (AuthorsFeaturelessNetTF / AuthorsFeatureBasedNetTF)
 - authors_mode=False: a simplified permutation-equivariant halo stack (BaseEncoder + HaloBlock) kept as an ablation/diagnostic

 Run the Part 1 experiment suite (smoke)
 ```bash
 jpmq3-run-all --smoke
```

## 4) Lu & Shimizu (2025) Section 4 replication
Smoke run (fast)
 ```bash
 jpmq3-replicate-lu25 --smoke --out results/part2/lu25_smoke```

Example debug run (single DGP cell)
 ```bash
jpmq3-replicate-lu25 \
  --out results/part2/lu25_debug \
  --grid DGP2:25:15 \
  --n-reps 1 \
  --R-mc 10 \
  --n-jobs 1 \
  --seed 0 \
  --shrink-n-iter 50 \
  --shrink-burn 25 \
  --shrink-thin 1
  ```
  
  Per grid cell outputs:
    summary.csv
    paper_table_like.csv
    config.json
  Saved under:
    results/part2/<run_name>/<DGP>_T<T>_J<J>/
    
## 5) Part 2(d) Hybrid — Zhang-Sparse (DeepHalo + sparse shocks)
Smoke tests (also writes artifacts):
 ```bash
python -m unittest tests.part2.test_zhang_lu_sparse_smoke -v
``` 
## 6) Bonus 1 — Storable goods / stockpiling (toy demo)
Run the demo:
 ```bash
python -m jpm_q3.bonus1.dynamic_model.run_demo
``` 
Smoke tests:
 ```bash
python -m unittest tests.bonus.test_bonus1_dynamic_smoke -v
```
Notes (macOS / Apple Silicon):
  TensorFlow may print Metal device logs; this is expected. If you need CPU-only for a run, set:
   ```bash
  export BONUS1_FORCE_CPU=1
  ```

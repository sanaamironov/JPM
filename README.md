# Question 3 – Discrete Choice Models for Credit Card Offers

This directory contains the solution for **Question 3**, split into two parts.


## Part 1 (Zhang 2025: Deep context-dependent choice model)
# Question 3 — Part 1 (Zhang et al. 2025): Deep Context-Dependent Choice

This directory contains the **Part 1** work for Question 3: reproducing and probing the synthetic experiments for the deep context-dependent choice model described in Zhang et al. (2025).

All scripts are designed to be runnable **from any working directory** and to write outputs to `results/` (not into `experiments/`).

```
part_1/
├── experiments/        # Runnable experiment scripts
├── choice_learn_ext/   # Model implementation and utilities
├── authors/            # Authors’ original PyTorch DeepHalo code
├── results/            # Auto-created; all outputs go here
├── environment.yml     # Conda env (Linux / Windows)
├── environment_mac.yml # Conda env (macOS Apple Silicon)
├── run_all.py          # Runs all Part 1 experiments
└── README.md
```

## Environment setup

Two Conda environment files are provided:

- **macOS (Apple Silicon)**: `environment_mac.yml`
- **Linux / Windows / non-macOS**: `environment.yml`

## macOS (Apple Silicon)

```bash
cd question_3/part_1
conda env create -f environment_mac.yml
conda activate q3-part1

# run all 6 experiments
python run_all.py
```

## Windows 
```
cd question_3/part_1
conda env create -f environment.yml
conda activate q3-part1

# run all 6 experiments
python run_all.py
```


## Part 2 (Lu 2025: Sparse market–product shocks, Section 4 replication)
```
Part 2
├── Standalone replication (Lu 2025, Section 4)
│   ├── replication_lu25/
│   └── run_replication_study.py
│
├── choice-learn extension (Lu-style sparse shocks)
│   └── choice_learn_extension/
```


See: `part_2/README.md`  
Contains: the main executable replication driver (BLP ± cost IV, Shrinkage, optional Lu25 MAP),
plus environment setup instructions.


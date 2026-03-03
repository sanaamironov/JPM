# Lu & Shimizu (2025) Section 4 Replication

This folder contains the Part 2 implementation for **Lu & Shimizu (2025)** (Section 4 simulation study) and the tooling needed to:

1) **Simulate** synthetic markets with market–product demand shocks and (optionally) price endogeneity  
2) **Estimate** demand under different assumptions (BLP with/without IV, shrinkage with sparse shocks)  
3) **Aggregate** Monte Carlo results into paper-style tables  
4) **Debug** behavior on a single dataset when replication results differ from the paper

If you only want the “reviewer path,” use the CLI commands shown below.  
If you want to run modules directly (without CLI), this README is the canonical place to do it.

---

## Quickstart - For Reviewers

### A) Run the replication (compute)

### A-1: Smoke test
```bash
jpmq3-replicate-lu25 --smoke --out results/part2/lu25_smoke
```

### A-2: Larger Run

```bash
jpmq3-replicate-lu25 \
  --out results/part2/lu25_section4 \
  --n-reps 50 --R-mc 200 --seed 0 \
  --n-jobs 4 --threads-per-job 1 \
  --shrink-n-iter 800 --shrink-burn 400 --shrink-thin 2
```

### A-3: Run a single cell only
```bash
jpmq3-replicate-lu25 \
  --out results/part2/lu25_onecell \
  --grid DGP2:25:15 --n-reps 10 --R-mc 50
```

### A-4: Format combined tables (no recomputation)
``` bash
jpmq3-format-lu25-tables --in results/part2/lu25_section4
```
This reads per-cell output files and produces combined “paper-like” CSV tables.

### A-5: run a full grid with 16 cells
  
```bash
jpmq3-replicate-lu25 \
    --out results/part2/lu25_fullgrid \
    --n-reps 50 --R-mc 200 --seed 0 \
    --n-jobs 4 --threads-per-job 1 \
    --shrink-n-iter 800 --shrink-burn 400 --shrink-thin 2 \
    --grid DGP1:25:5,DGP1:100:5,DGP1:25:15,DGP1:100:15,\
      DGP2:25:5,DGP2:100:5,DGP2:25:15,DGP2:100:15,\
      DGP3:25:5,DGP3:100:5,DGP3:25:15,DGP3:100:15,\
      DGP4:25:5,DGP4:100:5,DGP4:25:15,DGP4:100:15
```

## 2) How to run without the CLI
Everything exposed through CLI is also runnable as Python modules.
### 2-A: Run replication directly
`python -m jpm_q3.lu25.experiments.replicate_section4 --smoke --out results/part2/lu25_smoke`

### 2-B: Single-cell direct run

```bash  
python -m jpm_q3.lu25.experiments.replicate_section4 \
    --out results/part2/lu25_onecell \
    --grid DGP2:25:15 --n-reps 10 --R-mc 50
```

### 2-C: Run table formatter directly
`python -m jpm_q3.lu25.experiments.format_section4_tables --in results/part2/lu25_section4`

### 2-D: Run a single replication debug tool
This runs one dataset and prints/exports a compact single-run diagnostic

```bash
  python -m jpm_q3.lu25.experiments.run_single_replication \
    --dgp DGP2 --T 25 --J 15 --seed 123 --R-mc 50
```

### 2-D: Optional MAP benchmark
`python -m jpm_q3.lu25.experiments.run_single_replication --include-map`

---

## 3) What is “canonical” vs “debug-only” 
### 3-A: Canonical pipeline
    
- **experiments/replicate_section4.py**: The only script intended to produce the simulation study results. This is what the CLI calls.
- **experiments/format_section4_tables.py**: Post-processing only: merges outputs from the canonical runner into combined paper-style tables.

### 3-B: Debug Tools
- **experiments/run_single_replication.py**: For diagnosing behavior on one simulated dataset. Not used to generate paper tables.
    
### 3-C: Deprecated Tools
Any scripts moved into _deprecate/ are not part of the final pipeline and are kept only for reference. 

## 4) File Structure (with annotations)

```
experiments/ : Everything that orchestrates simulations and writes outputs.
| - replicate_section4.py (canonical): Runs Monte Carlo replications, writes per-cell outputs. Implements paper-aligned reporting choices (see below).
| - format_section4_tables.py (canonical post-processing): Reads paper_table_like.csv and summary.csv per cell and writes combined tables.
| - run_single_replication.py (debug): One dataset → run estimators → export small diagnostics. Optional MAP benchmark.

simulation/ : Synthetic data generation for Lu(25) Section 4.
| - config.py: SimConfig: true parameter values and DGP settings (sparsity fraction, endogeneity strength, etc.).
| - dgp.py: Implements DGP1–DGP4 by generating:
    -> eta_star (sparse/dense shock component)
    -> alpha_star (price endogeneity component correlated with shocks)
| - market.py: Simulates one market: draws product characteristics, shocks, endogenous prices, and demand shares.
| - simulate.py: Simulates a dataset: list of markets, each a dict with keys required by estimators.

estimators/ : Estimator components used by the experiments.
| - blp.py: BLP building blocks: contraction mapping for delta, 2SLS/GMM utilities.
| - shrinkage.py: TFP MCMC shrinkage regression core:
    -> input: delta_vec, X
    -> output: posterior mean beta, posterior inclusion probs gamma_prob, score, acceptance rate
    -> Sigma search and table reporting are handled in the experiment runner.
| - lu25_map.py: (OPTIONAL/BENCHMARK): Optional MAP-style estimator (not required by the paper table replication). Used only when explicitly enabled in the single-rep debug tool.
```

## 5) Outputs, and how to Interpret

For each cell folder DGPk_T{T}_J{J}/:
    
- `paper_table_like.csv` : Two rows per method
  - Row=Bias: bias of Int, beta_p, beta_w, sigma + xi/probability metrics
  - Row=SD: sd of Int, beta_p, beta_w, sigma (xi/probability columns blank or NaN by design)
- `summary.csv` : Long format, suitable for programmatic slicing/plotting.
    

The formatter combines these into:
    
- `paper_table_like_combined.csv` (stacked across cells)
- `paper_table_wide.csv` (pivoted, easy to paste into a report)
- `summary_long_combined.csv` (if present)

## 6) Performance notes
    
Delta inversion (contraction mapping) and shrinkage MCMC dominate runtime.
    
Outputs are written after a grid cell finishes (not after every replication).
    
For multi-core runs on macOS:
    
use `--n-jobs > 1`
    
keep `--threads-per-job 1` to avoid oversubscription

    
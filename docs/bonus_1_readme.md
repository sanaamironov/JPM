# Dynamic Context + Sparse Shocks for Storable Goods

## Overview

This folder contains a **dynamic discrete choice scaffold** that combines:

1. **Context-dependent utility modeling** (DeepHalo backbone from Part 1),
2. **Sparse unobserved market-product shocks** (Lu-style decomposition from Part 2),
3. **Inventory-based forward-looking behavior** (storable-goods demand setting).

The goal is to provide a clean, runnable foundation for dynamic demand estimation where
both within-choice-set context effects and latent market heterogeneity matter.

---

## Key Ideas

- **Context dependence (Part 1)**  
  Utility includes a DeepHalo term that depends on the full offered set.

- **Structured unobserved heterogeneity (Part 2 / Lu-style)**  
  Latent shock for inside goods is decomposed as:
  $$
  \xi_{tj} = \mu_t + d_{tj}
  $$
  with sparse-product deviations encouraged via a spike-and-slab-flavored prior penalty.

- **Dynamic storable-goods behavior**  
  Inventory is a state variable; the model learns a value function and uses a TD/Bellman-style loss.

---

## Model Formulation

### Utility

For market $t$, item $j$, observation $n$:
$$
u_{ntj} =
u^{\text{halo}}_{ntj}
 + 1\{j \neq 0\}\left(\mu_t + d_{tj}\right)
$$

- $u^{\text{halo}}_{ntj}$: context utility from DeepHalo  
- $\mu_t$: market-level latent shock  
- $d_{tj}$: product-level deviation (sparse across $j$)

### Choice Probabilities

$$
P(y_n=j \mid s_n) = \frac{\exp(u_{ntj})}{\sum_{k \in \mathcal{A}_n}\exp(u_{ntk})}
$$
with availability mask $\mathcal{A}_n$.

### Inventory Transition (synthetic demo)

$$
I_{n,t+1} = \max\{0,\ I_{nt} + q(y_{nt}) - c_{nt}\}
$$

- $q(y_{nt})$: purchased quantity (0 for outside option)  
- $c_{nt}$: stochastic consumption

### Training Objective

$$
\mathcal{L}
=
\underbrace{\text{NLL}_{\text{choice}}}_{\text{discrete choice fit}}
+
\lambda_{\text{TD}}\underbrace{\text{MSE}(V - [r + \beta V'])}_{\text{dynamic consistency}}
+
\lambda_{\text{prior}}\underbrace{\mathcal{P}_{\text{sparse}}(\mu,d,\pi)}_{\text{Lu-style shrinkage}}
$$

Where \(\mathcal{P}_{\text{sparse}}\) uses a spike-and-slab mixture on \(d\) and a Beta prior on \(\pi\).

---

## What Was Added / Changed

- Added a new package: `bonus_1/dynamic_model/`
- Integrated `bonus_1.deephalo` into a dynamic model class:
  - `DynamicContextSparseChoiceModel` in `model.py`
- Added synthetic dynamic panel simulation with inventory transitions:
  - `simulate_dynamic_panel` in `data.py`
- Added trainer for joint optimization of:
  - choice likelihood,
  - TD loss,
  - sparse-shock prior penalty
- Added demo runner:
  - `run_demo.py`
- Added Apple Silicon runtime safeguards:
  - CPU-safe default,
  - optional legacy Adam and non-compiled train step controls.

---

## Implementation Structure

- `config.py`  
  Hyperparameters for architecture, priors, dynamics, and runtime behavior.

- `model.py`  
  Core model:
  - DeepHalo context utilities,
  - Lu-style \((\mu_t, d_{tj})\) layer,
  - value-function head,
  - composite loss construction.

- `data.py`  
  Synthetic storable-goods panel generator with market assignment, latent shocks, choices, and inventory transitions.

- `trainer.py`  
  Training loop with optimizer setup, batch updates, and epoch metrics.

- `run_demo.py`  
  End-to-end script: simulate data, train model, print diagnostics.

---

## How to Run

From repository root:

```bash
python -m bonus_1.dynamic_model.run_demo
```

---

"""
zhang_lu_sparse.py

Zhang-Sparse (Lu25-aligned) estimator using DeepHalo (Zhang 2025)
with a Lu(2025)-style sparse market×product shock layer.

Key idea
--------
DeepHalo provides context-dependent utilities u_halo(t,j).
Lu(2025) suggests unobserved shocks decompose as xi_{t j} = mu_t + d_{t j} with d sparse.

We implement:
  u_{t j} = u_halo(t,j) + 1{j!=0} * ( mu_t + d_{t j} )

and estimate by MAP:
  minimize mean-NLL
    + lambda * mean(|d|)
    + (1/(2*mu_sd^2)) * mean(mu^2)

Run
---
python -m jpm_q3.hybrid.zhang_lu_sparse
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from choice_learn.data import ChoiceDataset


class _IgnoreChoiceLearnFeatureNameWarnings(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Shared Features Names were not provided" in msg:
            return False
        if "Items Features Names were not provided" in msg:
            return False
        return True


logging.getLogger().addFilter(_IgnoreChoiceLearnFeatureNameWarnings())

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

# Import DeepHalo from your packaged choice-learn extension
from choice_learn_ext.models.deep_context.config import DeepHaloConfig
from choice_learn_ext.models.deep_context.deep_halo_core import DeepHalo

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class ZhangSparseConfig:
    # DeepHalo architecture
    d_embed: int = 16
    n_blocks: int = 2
    n_heads: int = 2
    residual_variant: str = "fixed_base"
    dropout: float = 0.0

    # Training
    lr: float = 2e-3
    batch_size: int = 256
    epochs: int = 20
    verbose: int = 1

    # MAP penalties (mean-scaled)
    l1_strength: float = 0.25
    mu_sd: float = 5.0
    center_d_within_market: bool = True

    # diagnostics/support detection
    tau_detect: float = 0.5

    seed: int = 123


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class ZhangSparseDeepHalo(tf.keras.Model):
    """DeepHalo + 1{j!=0}(mu_t + d_tj)."""

    def __init__(self, num_items: int, T: int, J_inside: int, cfg: ZhangSparseConfig):
        super().__init__()
        self.cfg = cfg
        self.T = int(T)
        self.J_inside = int(J_inside)

        halo_cfg = DeepHaloConfig(
            d_embed=cfg.d_embed,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_blocks,
            residual_variant=cfg.residual_variant,
            featureless=True,
            vocab_size=num_items,
            dropout=cfg.dropout,
        )
        self.halo = DeepHalo(halo_cfg)

        # shocks: inside goods only
        self.mu = tf.Variable(tf.zeros([self.T], dtype=tf.float32), name="mu")
        self.d = tf.Variable(
            tf.zeros([self.T, self.J_inside], dtype=tf.float32), name="d"
        )

    def call(
        self, inputs: Dict[str, tf.Tensor], training: bool = False
    ) -> Dict[str, tf.Tensor]:
        out = self.halo(inputs, training=training)
        u = (
            out["utilities"] if isinstance(out, dict) and "utilities" in out else out
        )  # (B, n_items)

        market_id = tf.reshape(tf.cast(inputs["market_id"], tf.int32), [-1])  # (B,)

        mu_t = tf.gather(self.mu, market_id)  # (B,)
        d_t = tf.gather(self.d, market_id)  # (B, J_inside)

        # Identification constraint: within-market mean(d_t) = 0
        if self.cfg.center_d_within_market:
            d_t = d_t - tf.reduce_mean(d_t, axis=1, keepdims=True)

        # pad outside option (index 0)
        d_pad = tf.concat([tf.zeros_like(d_t[:, :1]), d_t], axis=1)  # (B, n_items)

        # apply mu only to inside goods
        B = tf.shape(u)[0]
        n_items = tf.shape(u)[1]
        inside_mask = tf.concat(
            [tf.zeros([B, 1], dtype=u.dtype), tf.ones([B, n_items - 1], dtype=u.dtype)],
            axis=1,
        )

        u_aug = u + inside_mask * mu_t[:, None] + d_pad

        # availability mask
        avail = tf.cast(inputs["available"], tf.float32)
        u_masked = tf.where(avail > 0.5, u_aug, tf.cast(-1e9, u_aug.dtype))
        log_probs = tf.nn.log_softmax(u_masked, axis=1)

        return {"utilities": u_aug, "log_probs": log_probs}

    def nll(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        out = self.call(inputs, training=training)
        logP = out["log_probs"]
        y = tf.cast(inputs["choice"], tf.int32)
        idx = tf.stack([tf.range(tf.shape(y)[0], dtype=tf.int32), y], axis=1)
        return -tf.reduce_mean(tf.gather_nd(logP, idx))

    def map_objective(
        self, inputs: Dict[str, tf.Tensor], training: bool = False
    ) -> tf.Tensor:
        nll = self.nll(inputs, training=training)
        l1 = self.cfg.l1_strength * tf.reduce_mean(tf.abs(self.d))
        mu_ridge = 0.5 * tf.reduce_mean(tf.square(self.mu)) / (self.cfg.mu_sd**2)
        return nll + l1 + mu_ridge


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------


class AblationTrainer:
    def __init__(
        self, model: ZhangSparseDeepHalo, lr: float, train_vars: list[tf.Variable]
    ):
        self.model = model
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.train_vars = train_vars

    @tf.function
    def train_step(self, batch: Dict[str, tf.Tensor]) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = self.model.map_objective(batch, training=True)
        grads = tape.gradient(loss, self.train_vars)

        # Filter None grads (can happen if a var is disconnected in an ablation)
        pairs = [(g, v) for g, v in zip(grads, self.train_vars) if g is not None]
        self.opt.apply_gradients(pairs)
        return loss

    def fit(
        self,
        data: Dict[str, np.ndarray],
        batch_size: int,
        epochs: int,
        verbose: int = 1,
    ):
        N = int(len(data["choice"]))
        idx = np.arange(N)

        for ep in range(1, epochs + 1):
            np.random.shuffle(idx)
            losses = []
            for s in range(0, N, batch_size):
                b = idx[s : s + batch_size]
                batch = {k: tf.convert_to_tensor(v[b]) for k, v in data.items()}
                losses.append(float(self.train_step(batch).numpy()))
            if verbose:
                print(f"Epoch {ep:03d} | MAP loss {np.mean(losses):.4f}")


# -----------------------------------------------------------------------------
# Data conversion
# -----------------------------------------------------------------------------


def choice_dataset_to_tensors(
    dataset: ChoiceDataset, n_items: int
) -> Dict[str, np.ndarray]:
    shared = dataset.shared_features_by_choice
    if isinstance(shared, (tuple, list)):
        shared = shared[0]
    shared = np.asarray(shared)

    choices = np.asarray(dataset.choices).reshape(-1)
    N = int(choices.shape[0])

    if shared.ndim == 1:
        market_id = shared.astype(np.int32).reshape(-1)
    else:
        market_id = shared[:, 0].astype(np.int32).reshape(-1)

    item_ids = np.tile(np.arange(n_items, dtype=np.int32)[None, :], (N, 1))
    available = np.ones((N, n_items), dtype=np.float32)

    return {
        "item_ids": item_ids,
        "available": available,
        "choice": choices.astype(np.int32),
        "market_id": market_id,
    }


# -----------------------------------------------------------------------------
# Simulation (matches the model: mu applies to inside goods only)
# -----------------------------------------------------------------------------


def simulate_context_plus_sparse(
    T: int,
    J_inside: int,
    N_t: int,
    seed: int,
    d_embed: int,
    n_blocks: int,
    n_heads: int,
    mu_sd: float = 2.0,
    sparse_frac_nonzero: float = 0.4,
    d_sd: float = 1.0,
) -> Tuple[ChoiceDataset, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    n_items = J_inside + 1
    N = T * N_t

    market_id = np.repeat(np.arange(T, dtype=np.int32), N_t)
    item_ids = np.tile(np.arange(n_items, dtype=np.int32)[None, :], (N, 1))
    available = np.ones((N, n_items), dtype=np.float32)

    halo_cfg = DeepHaloConfig(
        d_embed=d_embed,
        n_heads=n_heads,
        n_layers=n_blocks,
        residual_variant="fixed_base",
        featureless=True,
        vocab_size=n_items,
        dropout=0.0,
    )
    halo = DeepHalo(halo_cfg)

    out = halo(
        {"available": tf.constant(available), "item_ids": tf.constant(item_ids)},
        training=False,
    )
    u_halo = (
        out["utilities"].numpy() if isinstance(out, dict) else out.numpy()
    ).astype(np.float32)

    mu_true = rng.normal(0.0, mu_sd, size=(T,)).astype(np.float32)
    d_true = np.zeros((T, J_inside), dtype=np.float32)
    gamma_true = np.zeros((T, J_inside), dtype=np.int32)

    k = max(1, int(sparse_frac_nonzero * J_inside))
    for t in range(T):
        idx = rng.choice(J_inside, size=k, replace=False)
        d_true[t, idx] = rng.normal(0.0, d_sd, size=(k,)).astype(np.float32)
        gamma_true[t, idx] = 1

    d_pad = np.concatenate([np.zeros((T, 1), dtype=np.float32), d_true], axis=1)

    inside_mask = np.concatenate(
        [
            np.zeros((N, 1), dtype=np.float32),
            np.ones((N, n_items - 1), dtype=np.float32),
        ],
        axis=1,
    )

    u_total = u_halo + inside_mask * mu_true[market_id][:, None] + d_pad[market_id]

    u_max = u_total.max(axis=1, keepdims=True)
    expu = np.exp(u_total - u_max)
    P = expu / expu.sum(axis=1, keepdims=True)

    choices = np.array([rng.choice(n_items, p=P[i]) for i in range(N)], dtype=np.int32)

    try:
        dataset = ChoiceDataset(
            shared_features_by_choice=market_id.reshape(-1, 1),
            items_features_by_choice=np.zeros((N, n_items, 1), dtype=np.float32),
            choices=choices,
            shared_features_names=["market_id"],
            items_features_names=["dummy_item_feature"],
        )
    except TypeError:
        dataset = ChoiceDataset(
            shared_features_by_choice=market_id.reshape(-1, 1),
            items_features_by_choice=np.zeros((N, n_items, 1), dtype=np.float32),
            choices=choices,
        )

    meta = {"mu_true": mu_true, "d_true": d_true, "gamma_true": gamma_true}
    return dataset, meta


# -----------------------------------------------------------------------------
# Ablation runner
# -----------------------------------------------------------------------------


def set_halo_trainable(model: ZhangSparseDeepHalo, trainable: bool) -> None:
    model.halo.trainable = bool(trainable)


def build_and_init_model(
    cfg: ZhangSparseConfig,
    n_items: int,
    T: int,
    J_inside: int,
    init_from_weights: Optional[list[np.ndarray]] = None,
) -> ZhangSparseDeepHalo:
    model = ZhangSparseDeepHalo(num_items=n_items, T=T, J_inside=J_inside, cfg=cfg)

    # Build variables by running one forward pass
    dummy = {
        "item_ids": tf.constant(np.arange(n_items, dtype=np.int32)[None, :]),
        "available": tf.constant(np.ones((1, n_items), dtype=np.float32)),
        "choice": tf.constant([0], dtype=tf.int32),
        "market_id": tf.constant([0], dtype=tf.int32),
    }
    _ = model.call(dummy, training=False)

    if init_from_weights is not None:
        model.set_weights(init_from_weights)

    model.mu.assign(tf.zeros_like(model.mu))
    model.d.assign(tf.zeros_like(model.d))
    return model


def select_train_vars(
    model: ZhangSparseDeepHalo, learn_mu: bool, learn_d: bool
) -> list[tf.Variable]:
    vars_ = []
    vars_.extend(model.halo.trainable_variables)
    if learn_mu:
        vars_.append(model.mu)
    if learn_d:
        vars_.append(model.d)
    return vars_


def evaluate_nll(model: ZhangSparseDeepHalo, data: Dict[str, np.ndarray]) -> float:
    tensors = {k: tf.convert_to_tensor(v) for k, v in data.items()}
    return float(model.nll(tensors, training=False).numpy())


def objective_breakdown(
    model: ZhangSparseDeepHalo, cfg: ZhangSparseConfig, data: Dict[str, np.ndarray]
) -> Dict[str, float]:
    tensors = {k: tf.convert_to_tensor(v) for k, v in data.items()}
    nll = float(model.nll(tensors, training=False).numpy())
    l1 = float((cfg.l1_strength * tf.reduce_mean(tf.abs(model.d))).numpy())
    mu_ridge = float(
        (0.5 * tf.reduce_mean(tf.square(model.mu)) / (cfg.mu_sd**2)).numpy()
    )
    return {
        "nll": nll,
        "l1": l1,
        "mu_ridge": mu_ridge,
        "mean_abs_d": float(tf.reduce_mean(tf.abs(model.d)).numpy()),
        "std_d": float(tf.math.reduce_std(model.d).numpy()),
        "std_mu": float(tf.math.reduce_std(model.mu).numpy()),
    }


def run_one_ablation(
    name: str,
    cfg: ZhangSparseConfig,
    data: Dict[str, np.ndarray],
    n_items: int,
    T: int,
    J_inside: int,
    learn_mu: bool,
    learn_d: bool,
    init_weights: list[np.ndarray],
):
    print(f"\n=== Ablation: {name} ===")

    cfg_local = ZhangSparseConfig(**vars(cfg))
    if not learn_d:
        cfg_local.l1_strength = 0.0
    if not learn_mu:
        cfg_local.mu_sd = 1e9

    model = build_and_init_model(
        cfg_local, n_items, T, J_inside, init_from_weights=init_weights
    )
    train_vars = select_train_vars(model, learn_mu=learn_mu, learn_d=learn_d)

    trainer = AblationTrainer(model, lr=cfg_local.lr, train_vars=train_vars)
    trainer.fit(
        data,
        batch_size=cfg_local.batch_size,
        epochs=cfg_local.epochs,
        verbose=cfg_local.verbose,
    )

    nll = evaluate_nll(model, data)
    bd = objective_breakdown(model, cfg_local, data)

    print(f"{name} results:")
    print("  NLL:", nll)
    print("  breakdown:", bd)
    return model, bd


def run_two_stage_full(
    cfg: ZhangSparseConfig,
    data: Dict[str, np.ndarray],
    n_items: int,
    T: int,
    J_inside: int,
    init_weights: list[np.ndarray],
    stage1_epochs: Optional[int] = None,
    stage2_epochs: Optional[int] = None,
):
    print(
        "\n=== Two-stage: Full (DeepHalo + mu) then (freeze halo, learn mu + sparse d) ==="
    )

    # Stage 1: Halo + mu (L1 off; d not trained)
    cfg1 = ZhangSparseConfig(**vars(cfg))
    cfg1.l1_strength = 0.0
    if stage1_epochs is not None:
        cfg1.epochs = int(stage1_epochs)

    model = build_and_init_model(
        cfg1, n_items, T, J_inside, init_from_weights=init_weights
    )

    set_halo_trainable(model, True)
    model.cfg = cfg1
    train_vars_1 = select_train_vars(model, learn_mu=True, learn_d=False)

    trainer1 = AblationTrainer(model, lr=cfg1.lr, train_vars=train_vars_1)
    trainer1.fit(
        data, batch_size=cfg1.batch_size, epochs=cfg1.epochs, verbose=cfg1.verbose
    )

    # Stage 2: freeze halo, train mu + d (L1 on)
    cfg2 = ZhangSparseConfig(**vars(cfg))
    if stage2_epochs is not None:
        cfg2.epochs = int(stage2_epochs)

    set_halo_trainable(model, False)
    model.cfg = cfg2
    train_vars_2: list[tf.Variable] = [model.mu, model.d]

    trainer2 = AblationTrainer(model, lr=cfg2.lr, train_vars=train_vars_2)
    trainer2.fit(
        data, batch_size=cfg2.batch_size, epochs=cfg2.epochs, verbose=cfg2.verbose
    )

    nll = evaluate_nll(model, data)
    bd = objective_breakdown(model, cfg2, data)

    print("Two-stage Full results:")
    print("  NLL:", nll)
    print("  breakdown:", bd)

    return model, bd


def compute_support_metrics(
    d_hat: np.ndarray, gamma_true: np.ndarray, taus: list[float]
) -> list[dict]:
    true_nz = gamma_true == 1
    true_z = gamma_true == 0
    true_nz_rate = float(gamma_true.mean())

    out = []
    for tau in taus:
        gamma_hat = (np.abs(d_hat) > tau).astype(np.int32)
        sens = (
            float((gamma_hat[true_nz] == 1).mean())
            if true_nz.sum() > 0
            else float("nan")
        )
        spec = (
            float((gamma_hat[true_z] == 0).mean()) if true_z.sum() > 0 else float("nan")
        )
        pred_nz = float(gamma_hat.mean())
        out.append(
            {
                "tau": float(tau),
                "sensitivity": sens,
                "specificity": spec,
                "pred_nz": pred_nz,
                "true_nz_rate": true_nz_rate,
            }
        )
    return out


def save_results(out_dir: str | Path, payload: dict) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1) JSON (full payload)
    (out_path / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # 2) summary.csv
    summary_rows = payload["summary_rows"]
    with (out_path / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # 3) support.csv
    support_rows = payload["support_rows"]
    with (out_path / "support.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(support_rows[0].keys()))
        w.writeheader()
        w.writerows(support_rows)


def run_experiment(
    *,
    cfg: ZhangSparseConfig,
    T: int = 25,
    J_inside: int = 15,
    N_t: int = 1000,
    sparse_frac_nonzero: float = 0.4,
    mu_sd_sim: float = 2.0,
    d_sd_sim: float = 1.0,
    taus: Optional[list[float]] = None,
) -> dict:
    """
    Runs the three-model ablation and returns a structured payload suitable for saving.

    Returns a dict with:
      - config_used
      - metrics: per-model breakdowns
      - support_rows: list of tau metrics for the Full model
      - summary_rows: compact per-model table
    """
    if taus is None:
        taus = [0.25, float(cfg.tau_detect)]

    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    n_items = J_inside + 1

    dataset, meta = simulate_context_plus_sparse(
        T=T,
        J_inside=J_inside,
        N_t=N_t,
        seed=cfg.seed,
        d_embed=cfg.d_embed,
        n_blocks=cfg.n_blocks,
        n_heads=cfg.n_heads,
        sparse_frac_nonzero=float(sparse_frac_nonzero),
        mu_sd=float(mu_sd_sim),
        d_sd=float(d_sd_sim),
    )
    data = choice_dataset_to_tensors(dataset, n_items=n_items)

    base_model = build_and_init_model(cfg, n_items, T, J_inside, init_from_weights=None)
    init_weights = base_model.get_weights()

    results = {}

    _, results["DeepHalo-only"] = run_one_ablation(
        name="DeepHalo-only (mu=d frozen at 0)",
        cfg=cfg,
        data=data,
        n_items=n_items,
        T=T,
        J_inside=J_inside,
        learn_mu=False,
        learn_d=False,
        init_weights=init_weights,
    )

    _, results["Halo+mu"] = run_one_ablation(
        name="DeepHalo + mu (d frozen at 0)",
        cfg=cfg,
        data=data,
        n_items=n_items,
        T=T,
        J_inside=J_inside,
        learn_mu=True,
        learn_d=False,
        init_weights=init_weights,
    )

    full_model, results["Full"] = run_two_stage_full(
        cfg=cfg,
        data=data,
        n_items=n_items,
        T=T,
        J_inside=J_inside,
        init_weights=init_weights,
        stage1_epochs=cfg.epochs,
        stage2_epochs=cfg.epochs,
    )

    d_hat = full_model.d.numpy()
    gamma_true = meta["gamma_true"]
    support_rows = compute_support_metrics(d_hat, gamma_true, taus)

    summary_rows = []
    for k in ["DeepHalo-only", "Halo+mu", "Full"]:
        r = results[k]
        summary_rows.append(
            {
                "model": k,
                "nll": float(r["nll"]),
                "mean_abs_d": float(r["mean_abs_d"]),
                "std_mu": float(r["std_mu"]),
            }
        )

    payload = {
        "config_used": {
            **vars(cfg),
            "T": int(T),
            "J_inside": int(J_inside),
            "N_t": int(N_t),
            "sparse_frac_nonzero": float(sparse_frac_nonzero),
            "mu_sd_sim": float(mu_sd_sim),
            "d_sd_sim": float(d_sd_sim),
            "taus": [float(x) for x in taus],
        },
        "metrics": results,
        "summary_rows": summary_rows,
        "support_rows": support_rows,
    }
    return payload


def main():
    cfg = ZhangSparseConfig()
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    # Run 1: moderately sparse (paper/demo default)
    payload_40 = run_experiment(
        cfg=cfg,
        T=25,
        J_inside=15,
        N_t=1000,
        sparse_frac_nonzero=0.4,
        taus=[0.25, 0.35, cfg.tau_detect],
    )
    save_results("results/hybrid/zhang_lu_sparse/latest_sparse40", payload_40)

    print("\n=== Summary (in-sample) [sparse_frac_nonzero=0.4] ===")
    print(f"{'Model':<32} {'NLL':>10} {'mean|d|':>10} {'std(mu)':>10}")
    print("-" * 68)
    for row in payload_40["summary_rows"]:
        print(
            f"{row['model']:<32} {row['nll']:>10.4f} {row['mean_abs_d']:>10.4f} {row['std_mu']:>10.4f}"
        )
    print("\nFull model support recovery (|d| > tau)")
    for r in payload_40["support_rows"]:
        print(
            f"  tau={r['tau']:.2f}  sensitivity={r['sensitivity']:.3f}  specificity={r['specificity']:.3f}  "
            f"pred_nz={r['pred_nz']:.3f}  true_nz={r['true_nz_rate']:.3f}"
        )

    # Run 2: more sparse (optional, nice to show)
    payload_20 = run_experiment(
        cfg=cfg,
        T=25,
        J_inside=15,
        N_t=1000,
        sparse_frac_nonzero=0.2,
        taus=[0.25, 0.35, cfg.tau_detect],
    )
    save_results("results/hybrid/zhang_lu_sparse/latest_sparse20", payload_20)

    print("\n=== Summary (in-sample) [sparse_frac_nonzero=0.2] ===")
    print(f"{'Model':<32} {'NLL':>10} {'mean|d|':>10} {'std(mu)':>10}")
    print("-" * 68)
    for row in payload_20["summary_rows"]:
        print(
            f"{row['model']:<32} {row['nll']:>10.4f} {row['mean_abs_d']:>10.4f} {row['std_mu']:>10.4f}"
        )
    print("\nFull model support recovery (|d| > tau)")
    for r in payload_20["support_rows"]:
        print(
            f"  tau={r['tau']:.2f}  sensitivity={r['sensitivity']:.3f}  specificity={r['specificity']:.3f}  "
            f"pred_nz={r['pred_nz']:.3f}  true_nz={r['true_nz_rate']:.3f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

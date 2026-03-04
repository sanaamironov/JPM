from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import DynamicModelConfig
from .data import simulate_dynamic_panel
from .model import DynamicContextSparseChoiceModel
from .trainer import DynamicTrainer


def _save_bonus_results(out_dir: str | Path, payload: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    support_rows = payload.get("support_rows", [])
    if support_rows:
        with (out / "support.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(support_rows[0].keys()))
            w.writeheader()
            w.writerows(support_rows)


def main() -> None:
    cfg = DynamicModelConfig()

    # optional CPU force
    if os.getenv("BONUS1_FORCE_CPU", "1") == "1":
        cfg.force_cpu = True

    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    if cfg.force_cpu:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    print("Simulating synthetic dynamic panel...")
    data, meta = simulate_dynamic_panel(cfg)
    print(
        f"N obs: {len(data['choice'])}, items: {cfg.num_items}, markets: {cfg.num_markets}"
    )

    model = DynamicContextSparseChoiceModel(cfg)
    # Force sparse shocks to explain residual variation
    model.halo.trainable = False
    trainer = DynamicTrainer(model, cfg)
    trainer.fit(data)
    d_hat = model.d.numpy()  # shape: [num_markets, num_items-1] (inside only)

    print("\nLearned d diagnostics:")
    print("  max|d|:", float(np.max(np.abs(d_hat))))
    print("  mean|d|:", float(np.mean(np.abs(d_hat))))

    pi_hat = float(tf.math.sigmoid(model.logit_pi).numpy())
    mu_sd_hat = float(tf.math.reduce_std(model.mu).numpy())
    mean_abs_d = float(tf.reduce_mean(tf.abs(model.d)).numpy())

    print("\nLearned sparse-shock diagnostics:")
    print(f"  pi_hat:     {pi_hat:.4f}")
    print(f"  std(mu):    {mu_sd_hat:.4f}")
    print(f"  mean|d|:    {mean_abs_d:.4f}")

    # ---- Support recovery vs truth (inside goods only) ----
    gamma_true = np.asarray(
        meta["gamma_true"], dtype=np.int32
    )  # expected shape [T, J_inside]
    if gamma_true.shape != d_hat.shape:
        raise ValueError(
            f"Shape mismatch: gamma_true {gamma_true.shape} vs d_hat {d_hat.shape}. "
            "Check simulator: gamma_true should be inside-goods only."
        )

    true_nz = gamma_true == 1
    true_z = gamma_true == 0
    true_nz_rate = float(gamma_true.mean())

    taus = [0.10, 0.15, 0.20, 0.25]
    print("\nSupport recovery (thresholding |d| > tau):")
    print(f"  true_nz_rate: {true_nz_rate:.3f}")
    for tau in taus:
        gamma_hat = (np.abs(d_hat) > tau).astype(np.int32)
        pred_nz_rate = float(gamma_hat.mean())
        sens = (
            float((gamma_hat[true_nz] == 1).mean())
            if true_nz.sum() > 0
            else float("nan")
        )
        spec = (
            float((gamma_hat[true_z] == 0).mean()) if true_z.sum() > 0 else float("nan")
        )
        print(
            f"  tau={tau:.2f}  pred_nz_rate={pred_nz_rate:.3f}  "
            f"sensitivity={sens:.3f}  specificity={spec:.3f}"
        )

    from dataclasses import asdict

    # pick taus that make sense for the learned scale
    max_abs_d = float(np.max(np.abs(d_hat)))
    taus = [0.10, 0.12, 0.14, 0.15]
    taus = [t for t in taus if t < max_abs_d + 1e-12]  # avoid useless taus

    gamma_true = np.asarray(meta["gamma_true"], dtype=np.int32)
    true_nz = gamma_true == 1
    true_z = gamma_true == 0
    true_nz_rate = float(gamma_true.mean())

    support_rows = []
    for tau in taus:
        gamma_hat = (np.abs(d_hat) > tau).astype(np.int32)
        pred_nz_rate = float(gamma_hat.mean())
        sens = (
            float((gamma_hat[true_nz] == 1).mean())
            if true_nz.sum() > 0
            else float("nan")
        )
        spec = (
            float((gamma_hat[true_z] == 0).mean()) if true_z.sum() > 0 else float("nan")
        )
        support_rows.append(
            dict(
                tau=float(tau),
                pred_nz_rate=pred_nz_rate,
                sensitivity=sens,
                specificity=spec,
                true_nz_rate=true_nz_rate,
            )
        )

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "train": {
            "halo_frozen": True,
            "n_obs": int(len(data["choice"])),
            "num_items": int(cfg.num_items),
            "num_markets": int(cfg.num_markets),
        },
        "learned": {
            "pi_hat": pi_hat,
            "std_mu": mu_sd_hat,
            "mean_abs_d": mean_abs_d,
            "max_abs_d": max_abs_d,
        },
        "support_rows": support_rows,
    }

    out_dir = Path("results/bonus1/dynamic_model") / f"demo_{cfg.seed}"
    _save_bonus_results(out_dir, payload)
    print(f"\nSaved bonus artifacts to: {out_dir}")


if __name__ == "__main__":
    main()

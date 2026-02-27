from __future__ import annotations

import os
import numpy as np
from .config import DynamicModelConfig
from .data import simulate_dynamic_panel

# Runtime safety defaults for Apple/Metal:
# - Use legacy Keras path where available
# - Default to CPU to avoid known tensorflow-metal optimizer crashes
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
if os.getenv("BONUS1_FORCE_CPU", "1") == "1":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from .model import DynamicContextSparseChoiceModel
from .trainer import DynamicTrainer


def main() -> None:
    cfg = DynamicModelConfig()
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
    data = simulate_dynamic_panel(cfg)
    print(f"N obs: {len(data['choice'])}, items: {cfg.num_items}, markets: {cfg.num_markets}")

    model = DynamicContextSparseChoiceModel(cfg)
    trainer = DynamicTrainer(model, cfg)
    trainer.fit(data)

    pi_hat = float(tf.math.sigmoid(model.logit_pi).numpy())
    mu_sd_hat = float(tf.math.reduce_std(model.mu).numpy())
    mean_abs_d = float(tf.reduce_mean(tf.abs(model.d)).numpy())
    print("\nLearned sparse-shock diagnostics:")
    print(f"  pi_hat:     {pi_hat:.4f}")
    print(f"  std(mu):    {mu_sd_hat:.4f}")
    print(f"  mean|d|:    {mean_abs_d:.4f}")


if __name__ == "__main__":
    main()

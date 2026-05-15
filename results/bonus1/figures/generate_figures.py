"""
generate_figures.py — Sensitivity analysis figures for Bonus 1.

Result 1 (CF correction, 2 plots):
  1a. beta_price estimate vs endogeneity gamma — with/without CF lines
  1b. Bias-reduction fraction vs gamma — shows where CF earns its keep

Result 2 (mu_t recovery, 2 plots):
  2a. corr(mu_hat, mu_true) vs signal strength sigma_mu
  2b. corr(mu_hat, mu_true) vs sample size I

Uses a small fast config (J=3, T=10, I=50, 15 epochs) for the sweeps.
"""
import os, sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

from jpm_q3.bonus1.dynamic_model.config import DynamicModelConfig
from jpm_q3.bonus1.dynamic_model.data import simulate_dynamic_panel
from jpm_q3.bonus1.dynamic_model.model import DynamicContextSparseChoiceModel
from jpm_q3.bonus1.dynamic_model.simulation_study import _fit_stage

OUT = os.path.dirname(__file__)

# ── palette ────────────────────────────────────────────────────────────
C_CF    = "#1976D2"   # blue  – with CF
C_NOCF  = "#E53935"   # red   – without CF
C_MU_S  = "#7B1FA2"   # purple – mu / sigma sweep
C_MU_I  = "#F57C00"   # orange – mu / I sweep
C_TRUE  = "#212121"   # near-black – truth reference

PLT_RC = {
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
plt.rcParams.update(PLT_RC)

# Two configs:
#   _CF_CFG  — used for the gamma sweep (Sweep A): T=20 so the OLS first stage
#              has enough markets for the control function to work reliably
#              (same T as the main simulation study).
#   _MU_CFG  — used for mu/sample-size sweeps (B, C): T=12 keeps each run fast.
# Both use I=150 to match the main simulation scale.
SEED = 0
_CF_CFG = dict(J=3, S_max=3, T=20, num_households=150, batch_size=128,
               epochs=30, compile_train_step=False, force_cpu=True, seed=SEED)
_MU_CFG = dict(J=3, S_max=3, T=12, num_households=150, batch_size=128,
               epochs=20, compile_train_step=False, force_cpu=True, seed=SEED)
_FAST = _MU_CFG   # default (mu/sample sweeps)


def _run(overrides: dict, with_cf: bool) -> dict:
    cfg = DynamicModelConfig(**{**_FAST, **overrides})
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data, meta = simulate_dynamic_panel(cfg, seed=SEED)
    model = DynamicContextSparseChoiceModel(cfg)

    if not with_cf:
        model.lambda_control.assign(0.0)
        model.lambda_control.trainable = False

    # Stage 1 — econometric params, static NLL
    model.halo.trainable = False
    model.market_embed.trainable = False
    model.value_head.trainable = False
    econ = [v for v in [
        model.beta_price, model.lambda_control,
        model.mu, model.d, model.eta, model.logit_pi, model.kappa_0,
    ] if v.trainable]
    _fit_stage(model, cfg, data, econ,
               epochs=cfg.epochs, lr=1e-3, label="S1", use_static_nll=True)

    # Stage 2 — joint fine-tune
    model.halo.trainable = True
    model.market_embed.trainable = True
    model.value_head.trainable = True
    _fit_stage(model, cfg, data, model.trainable_variables,
               epochs=max(4, cfg.epochs // 3), lr=3e-4, label="S2")

    mu_hat  = model.mu.numpy()
    mu_true = meta["mu_true"]
    corr    = float(np.corrcoef(mu_hat, mu_true)[0, 1])
    beta    = float(model.beta_price.numpy())
    return {"beta": beta, "beta_true": cfg.true_beta_price,
            "mu_corr": corr, "mu_hat": mu_hat, "mu_true": mu_true}


# ═══════════════════════════════════════════════════════════════════════
#  SWEEP A: gamma_endogeneity × {with_cf, without_cf}
# ═══════════════════════════════════════════════════════════════════════
gammas = [0.0, 0.3, 0.6, 0.9]

def _run_cf(overrides, with_cf):
    return _run({**_CF_CFG, **overrides}, with_cf)

sweep_a_cf, sweep_a_nocf = [], []
for g in gammas:
    ov = {"gamma_endogeneity": g}
    print(f"  gamma={g}  with_cf=True  ...", flush=True)
    sweep_a_cf.append(_run_cf(ov, with_cf=True))
    print(f"  gamma={g}  with_cf=False ...", flush=True)
    sweep_a_nocf.append(_run_cf(ov, with_cf=False))

# ═══════════════════════════════════════════════════════════════════════
#  SWEEP B: sigma_mu (signal strength)
# ═══════════════════════════════════════════════════════════════════════
mu_sds = [0.3, 0.6, 1.0, 1.5]
sweep_b = []
for sd in mu_sds:
    ov = {"mu_true_sd": sd}
    print(f"  mu_true_sd={sd} ...", flush=True)
    sweep_b.append(_run(ov, with_cf=True))

# ═══════════════════════════════════════════════════════════════════════
#  SWEEP C: num_households (sample size)
# ═══════════════════════════════════════════════════════════════════════
hh_vals = [30, 60, 100, 150]
sweep_c = []
for hh in hh_vals:
    ov = {"num_households": hh}
    print(f"  num_households={hh} ...", flush=True)
    sweep_c.append(_run(ov, with_cf=True))

# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Result 1: CF correction
#    1a  Recovery fraction (%) grouped bar chart: with/without CF × gamma
#    1b  mu_hat vs mu_true scatter at gamma=0.6 (with CF), T=20
#         — shows the direction-identification property directly
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax1, ax2 = axes

beta_true = sweep_a_cf[0]["beta_true"]   # −1.5
rec_cf   = [abs(r["beta"]) / abs(beta_true) * 100 for r in sweep_a_cf]
rec_nocf = [abs(r["beta"]) / abs(beta_true) * 100 for r in sweep_a_nocf]

# 1a — recovery fraction grouped bar chart
x   = np.arange(len(gammas))
w   = 0.35
b1  = ax1.bar(x - w/2, rec_cf,   width=w, color=C_CF,   label="With CF",
              edgecolor="white", linewidth=1.2, zorder=3)
b2  = ax1.bar(x + w/2, rec_nocf, width=w, color=C_NOCF,  label="Without CF",
              edgecolor="white", linewidth=1.2, zorder=3)
ax1.axhline(100, color=C_TRUE, linestyle="--", lw=1.3,
            label=r"True ($100\%$)", zorder=2)
for bar, val in zip(list(b1) + list(b2), rec_cf + rec_nocf):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f"{val:.0f}%", ha="center", va="bottom", fontsize=8.5)
ax1.set_xlabel(r"Endogeneity strength $\gamma$")
ax1.set_ylabel(r"Recovery: $|\hat{\beta}|/|\beta^\star|$ (%)")
ax1.set_title(r"Price-coefficient recovery vs. $\gamma$"
              "\nwith and without Petrin–Train CF", fontweight="bold")
ax1.set_xticks(x); ax1.set_xticklabels([str(g) for g in gammas])
ax1.set_ylim(0, 130)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25, linestyle=":", axis="y")

# 1b — mu_hat vs mu_true scatter at gamma≈0.5 (index 2 = 0.6 condition)
idx   = 2   # gamma=0.6 ≈ main simulation gamma=0.5
r_sc  = sweep_a_cf[idx]
mh, mt = r_sc["mu_hat"], r_sc["mu_true"]
lim_lo = min(mh.min(), mt.min()) - 0.2
lim_hi = max(mh.max(), mt.max()) + 0.2
ax2.scatter(mt, mh, s=55, color=C_MU_S, alpha=0.8, zorder=3, edgecolors="white", lw=0.6)
lin = np.linspace(lim_lo, lim_hi, 60)
ax2.plot(lin, lin, color=C_TRUE, linestyle="--", lw=1.3, alpha=0.6,
         label="Perfect recovery ($y=x$)")
# OLS fit line
m, b_fit = np.polyfit(mt, mh, 1)
ax2.plot(lin, m*lin + b_fit, color=C_CF, lw=1.8, alpha=0.9,
         label=f"OLS fit (slope={m:.2f})")
ax2.set_xlabel(r"True $\mu^\star_t$")
ax2.set_ylabel(r"Estimated $\hat{\mu}_t$")
ax2.set_title(r"Market-shock direction identification ($\gamma=0.6$, with CF)"
              f"\n$r={r_sc['mu_corr']:.2f}$; slope<1 reflects MAP shrinkage",
              fontweight="bold")
ax2.legend(fontsize=9)
ax2.set_xlim(lim_lo, lim_hi); ax2.set_ylim(lim_lo, lim_hi)
ax2.grid(True, alpha=0.25, linestyle=":")
ax2.set_aspect("equal", adjustable="box")

fig.suptitle(
    r"Result 1 — CF Correction: recovery improves at high $\gamma$; "
    r"$\mu_t$ direction is identified despite scale compression",
    fontsize=11, fontweight="bold", y=1.03,
)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"bonus1_result1_cf_correction.{ext}"),
                bbox_inches="tight", dpi=180)
plt.close(fig)
print("Saved Figure 1", flush=True)

# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Result 2: mu_t recovery
#    2a  corr vs sigma_mu  (signal strength — mu vs sparse-d SNR)
#    2b  beta recovery (%) vs I  (sample size)
#         — uses beta, a scalar, which is more stable than T=12 mu-corr
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax1, ax2 = axes

corrs_b     = [r["mu_corr"] for r in sweep_b]
corrs_a_cf  = [r["mu_corr"] for r in sweep_a_cf]

# 2a — corr vs sigma_mu (signal strength)
ax1.plot(mu_sds, corrs_b, "o-", color=C_MU_S, lw=2, ms=8)
ax1.fill_between(mu_sds, 0, corrs_b, alpha=0.12, color=C_MU_S)
ax1.axhline(1.0, color=C_TRUE, linestyle="--", lw=1.2, alpha=0.5,
            label="Perfect recovery")
ax1.set_xlabel(r"Market-shock signal strength $\sigma_\mu$")
ax1.set_ylabel(r"$\mathrm{corr}(\hat{\mu}_t,\, \mu^\star_t)$")
ax1.set_title(r"$\mu_t$ recovery vs. shock signal strength"
              "\n(sparse noise $\\sigma_d=0.8$ fixed)", fontweight="bold")
ax1.set_xticks(mu_sds)
ax1.set_ylim(-0.05, 1.12)
ax1.legend()
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax1.grid(True, alpha=0.25, linestyle=":")

# 2b — corr vs gamma (endogeneity) with CF
ax2.plot(gammas, corrs_a_cf, "o-", color=C_MU_I, lw=2, ms=8)
ax2.fill_between(gammas, 0, corrs_a_cf, alpha=0.12, color=C_MU_I)
ax2.axhline(1.0, color=C_TRUE, linestyle="--", lw=1.2, alpha=0.5,
            label="Perfect recovery")
for xi, yi in zip(gammas, corrs_a_cf):
    ax2.annotate(f"{yi:.2f}", (xi, yi),
                 textcoords="offset points", xytext=(0, 7),
                 ha="center", fontsize=9)
ax2.set_xlabel(r"Endogeneity strength $\gamma$")
ax2.set_ylabel(r"$\mathrm{corr}(\hat{\mu}_t,\, \mu^\star_t)$")
ax2.set_title(r"$\mu_t$ recovery vs. endogeneity (with CF)"
              "\n($\\sigma_\\mu=1.0$, $T=20$ fixed)", fontweight="bold")
ax2.set_xticks(gammas)
ax2.set_ylim(-0.05, 1.12)
ax2.legend()
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax2.grid(True, alpha=0.25, linestyle=":")

fig.suptitle(
    r"Result 2 — Market Shock Recovery: $\mu_t$ direction identified; "
    "signal strength and sample size govern quality",
    fontsize=11, fontweight="bold", y=1.03,
)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"bonus1_result2_mu_recovery.{ext}"),
                bbox_inches="tight", dpi=180)
plt.close(fig)
print("Saved Figure 2", flush=True)
print("All figures generated.", flush=True)

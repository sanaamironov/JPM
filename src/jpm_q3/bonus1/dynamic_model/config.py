from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DynamicModelConfig:
    # ---- Problem dimensions (fixed by the revised question) ----
    J: int = 5          # number of brands; item 0 is the outside option, items 1..J are brands
    S_max: int = 5      # maximum inventory level (integer)
    T: int = 20         # number of time periods
    num_households: int = 100   # number of consumers I

    # ---- True DGP parameters (known to data generator, recovered by estimator) ----
    true_beta_price: float = -1.5   # true price sensitivity (negative: higher price lowers utility)
    gamma_endogeneity: float = 0.5  # endogeneity strength: p_jt = c_j + gamma*xi_jt + w_jt + noise
    sigma_price_noise: float = 0.3  # std of idiosyncratic price noise eta_jt
    sigma_alpha: float = 0.5        # std of brand fixed effects alpha_j
    kappa_stockout: float = 2.0     # utility penalty for outside option when inventory = 0
    delta_min: float = 0.70         # lower bound of consumer discount factor distribution
    delta_max: float = 0.95         # upper bound of consumer discount factor distribution
    min_avail: int = 3              # minimum number of brands in each choice set A_t

    # ---- Observable cost shifter w_jt for Petrin & Train (2010) control function ----
    # w_jt enters the price equation but is excluded from utility — the IV/control
    # function instrument that breaks the price endogeneity problem.
    sigma_w_cost: float = 0.8       # std of observable brand-time cost shifter w_jt
    pi_w_true: float = 1.0          # true coefficient on w_jt in the price equation

    # ---- Sparse market-product shock DGP ----
    sparse_frac: float = 0.30       # fraction of nonzero d_jt entries per market
    mu_true_sd: float = 1.0         # std of market-level mean shock mu_t
    d_true_sd: float = 0.8          # std of nonzero sparse deviation d_jt

    # ---- Consumer-specific unobserved taste eta_ij (revised question item (2)) ----
    sigma_eta: float = 0.5          # std of consumer-brand taste eta_ij (DGP and prior)

    # ---- DeepHalo backbone (Halo effect) ----
    d_embed: int = 16
    n_blocks: int = 2
    n_heads: int = 2
    residual_variant: str = "fixed_base"
    dropout: float = 0.0

    # ---- Estimation parameters (used by estimator, not DGP) ----
    # Common discount used in estimation (approximation; true delta_i are consumer-specific).
    # Stated as a modeling assumption in the report.
    discount: float = 0.90
    # Initialise beta_price at a negative value (a reasonable prior: higher
    # price lowers demand). Starting at 0 leaves the optimiser in a flat
    # local-minimum basin and prevents the control function from converging.
    beta_price_init: float = -1.0

    # ---- Lu-style sparse shock prior ----
    v0: float = 0.05        # spike variance
    v1: float = 1.0         # slab variance
    a_pi: float = 1.0
    b_pi: float = 9.0       # prior mean pi ≈ 0.1 (sparse)
    mu_sd: float = 3.0      # Gaussian prior std on mu_t
    center_d_within_market: bool = True
    prior_weight: float = 0.005

    # ---- Training ----
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 20
    seed: int = 123
    compile_train_step: bool = True
    force_cpu: bool = True

    # ---- Derived quantities (read-only) ----
    @property
    def num_items(self) -> int:
        """Total number of choice options including outside option (index 0)."""
        return self.J + 1

    @property
    def num_markets(self) -> int:
        """Number of markets = number of time periods T."""
        return self.T

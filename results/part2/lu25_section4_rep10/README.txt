Results produced by:
  jpmq3-replicate-lu25 --out results/part2/lu25_section4_rep10 --n-reps 10

MCMC / shrinkage config:
  n_iter=800, burn=400, thin=1
  v0=0.05, v1=1.0
  a_pi=1.0, b_pi=9.0
  beta_var=1e6
  beta_rw_scale=0.05, pi_rw_scale=0.20

These parameters are now stored in each config.json produced by new runs.
Pre-existing config.json files in this folder contain only the 5-key legacy
format (dgp, T, J, true_params, timestamp); the values above apply to all of them.

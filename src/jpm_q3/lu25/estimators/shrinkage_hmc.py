import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfm = tfp.mcmc

@tf.function(reduce_retracing=True)
def _sample_beta_pi_chain_hmc(
    *,
    X_tf: tf.Tensor,
    y_tf: tf.Tensor,
    beta_init_tf: tf.Tensor,
    logit_pi_init_tf: tf.Tensor,
    v0_tf: tf.Tensor,
    v1_tf: tf.Tensor,
    beta_var_tf: tf.Tensor,
    a_pi_tf: tf.Tensor,
    b_pi_tf: tf.Tensor,
    num_results: tf.Tensor,
    num_burnin: tf.Tensor,
    seed: tf.Tensor,
    # HMC params
    num_leapfrog_steps: tf.Tensor,
    init_step_size: tf.Tensor,
    adapt_steps: tf.Tensor,
    target_accept: tf.Tensor,
):
    eps_tf = tf.constant(1e-12, dtype=tf.float64)
    log2pi_tf = tf.constant(np.log(2.0 * np.pi), dtype=tf.float64)

    def target_log_prob_fn(beta, logit_pi):
        pi = tf.math.sigmoid(logit_pi)
        r = y_tf - tf.linalg.matvec(X_tf, beta)

        logn0 = -0.5 * (tf.math.log(v0_tf) + log2pi_tf + tf.square(r) / v0_tf)
        logn1 = -0.5 * (tf.math.log(v1_tf) + log2pi_tf + tf.square(r) / v1_tf)

        log_mix = tf.reduce_logsumexp(
            tf.stack(
                [
                    tf.math.log1p(-pi + eps_tf) + logn0,
                    tf.math.log(pi + eps_tf) + logn1,
                ],
                axis=0,
            ),
            axis=0,
        )
        ll = tf.reduce_sum(log_mix)

        beta_lp = tf.reduce_sum(
            -0.5 * (tf.square(beta) / beta_var_tf + tf.math.log(beta_var_tf) + log2pi_tf)
        )

        pi_lp = tfd.Beta(concentration1=a_pi_tf, concentration0=b_pi_tf).log_prob(pi)
        log_abs_det_jac = tf.math.log(pi + eps_tf) + tf.math.log1p(-pi + eps_tf)

        return ll + beta_lp + pi_lp + log_abs_det_jac

    base_kernel = tfm.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=init_step_size,
        num_leapfrog_steps=num_leapfrog_steps,
    )

    kernel = tfm.DualAveragingStepSizeAdaptation(
        inner_kernel=base_kernel,
        num_adaptation_steps=adapt_steps,
        target_accept_prob=target_accept,
    )

    def trace_fn(_cs, kr):
        inner = kr.inner_results
        # DualAveraging wraps the kernel results
        logp = inner.accepted_results.target_log_prob
        is_acc = inner.is_accepted
        step = kr.new_step_size
        return (logp, is_acc, step)

    states, trace = tfm.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin,
        current_state=[beta_init_tf, logit_pi_init_tf],
        kernel=kernel,
        trace_fn=trace_fn,
        seed=seed,
    )

    beta_draws, logit_pi_draws = states
    logpost_draws, is_accepted, step_sizes = trace
    acc_rate = tf.reduce_mean(tf.cast(is_accepted, tf.float64))
    step_size_final = step_sizes[-1]
    return beta_draws, logit_pi_draws, logpost_draws, acc_rate, step_size_final
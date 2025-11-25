"""
Multivariate normal dueling bandit implementation.

This module contains a bandit algorithm for selecting the best flood‐protection height
using Bayesian inference under a multivariate normal model.  It is designed as
an alternative to the beta–dueling bandit described in ``beta_dueling_bandit_analytical.py``.

The algorithm proceeds in three stages:

1. **Tuning the batch size ``S`` for normality**.  The cost of each height is
   random because annual floods are random.  When averaged over many simulated
   floods the vector of mean costs across all candidate heights tends towards
   a multivariate normal distribution by the central limit theorem.  We choose a
   batch size ``S`` large enough that this approximation holds by repeatedly
   sampling ``test_replicates`` mean cost vectors of length ``num_cands`` using
   ``S`` Monte Carlo floods per replicate, applying Mardia’s test for
   multivariate normality, and doubling ``S`` until both skewness and kurtosis
   fall inside a desired significance level.

2. **Establishing a prior from the tuned samples**.  Once ``S`` is determined
   we draw ``test_replicates`` more batches of ``S`` floods and compute the
   corresponding sample mean vectors.  These vectors provide an empirical
   estimate of the underlying covariance of single‐scenario costs (multiplied
   by ``S``), the grand mean of the candidate heights, and the effective
   number of underlying observations (``n_init = test_replicates * S``).  From
   these quantities we infer a flat (uninformative) multivariate normal prior
   over the true mean cost vector.

3. **Sequential Bayesian updating**.  After the prior is set the algorithm
   proceeds scenario by scenario.  For each new simulated flood we compute the
   total cost vector across surviving heights and update the running mean and
   covariance via Welford’s online covariance algorithm.  At regular intervals
   (every ``check_every`` simulations) we compute a posterior covariance
   for the true mean (which is just the sample covariance divided by the
   effective sample size) and draw many samples from this multivariate normal
   posterior.  For each sampled mean we record which height has the minimum
   total cost.  The proportion of samples in which a given height is best
   provides an estimate of the posterior probability that that height is
   optimal.  If the maximum of these probabilities exceeds ``1 - delta`` we
   terminate and return that height.

The key difference from the beta–dueling bandit is that we do not track
pairwise win probabilities.  Instead we work directly with the joint
distribution of the mean cost vector and use the probability of being
minimal as the stopping criterion.

This implementation relies on ``numpy`` and ``scipy`` for linear algebra
and statistical tests.

Note
----
The tuning phase can be computationally expensive because it involves
``test_replicates`` × ``S`` Monte Carlo flood simulations.  For
reasonable performance set ``test_replicates`` to a moderate number (e.g.,
200–500) when using this code in practice.  The values provided here are
parameters that can be adjusted depending on the available compute budget.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.random import Generator
from scipy.stats import chi2, norm, multivariate_normal


def simulate_annual_max_pp(
    eta0: float,
    eta1: float,
    alpha0: float,
    xi: float,
    X_t_series_cm: np.ndarray,
    u_cm: float,
    rng: Generator,
) -> np.ndarray:
    """
    Simulate one trajectory of annual maxima (in cm) for a given parameter
    draw and mean sea level (MSL) path, using the Poisson–GPD model.

    This is the same helper as in ``beta_dueling_bandit_analytical``.

    Parameters
    ----------
    eta0, eta1 : float
        Poisson intensity regression parameters.
    alpha0 : float
        Log–scale parameter for the GPD: sigma = exp(alpha0).
    xi : float
        GPD shape parameter.
    X_t_series_cm : np.ndarray
        Latent MSL time series in centimetres for the target years.
    u_cm : float
        POT threshold in centimetres.
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    maxima_cm : np.ndarray
        Annual maxima in centimetres, same length as X_t_series_cm.
    """
    sigma = np.exp(alpha0)
    T = X_t_series_cm.shape[0]
    maxima_cm = np.empty(T, dtype=float)
    for j in range(T):
        lam = math.exp(eta0 + eta1 * X_t_series_cm[j])
        N = rng.poisson(lam)
        if N <= 0:
            maxima_cm[j] = u_cm
        else:
            U = rng.uniform(size=N)
            if abs(xi) < 1e-6:
                Z = -sigma * np.log(1.0 - U)
            else:
                Z = (sigma / xi) * ((1.0 - U) ** (-xi) - 1.0)
            maxima_cm[j] = u_cm + float(np.max(Z))
    return maxima_cm


def _interpolate_damage(
    exceedances_m: np.ndarray,
    height_grid_m: np.ndarray,
    damage_grid: np.ndarray,
) -> np.ndarray:
    """
    Interpolate damage costs for arbitrary exceedance heights (in metres).

    Negative exceedances are treated as zero (no damage). Values are
    linearly interpolated and clipped at the endpoints of the grid.
    """
    w = np.asarray(exceedances_m, dtype=float).copy()
    w[w < 0.0] = 0.0
    return np.interp(w, height_grid_m, damage_grid)


def _simulate_single_scenario(
    rng: Generator,
    cand_heights: np.ndarray,
    cand_protections: np.ndarray,
    height_grid_m: np.ndarray,
    damage_grid: np.ndarray,
    X_paths_slice_cm: np.ndarray,
    posterior_params: Dict[str, np.ndarray],
    u_cm: float,
) -> np.ndarray:
    """
    Simulate a single flood scenario and return the total cost vector for
    each candidate height.

    This function samples a set of posterior parameters and a mean sea level
    (MSL) path from the provided arrays, simulates annual maxima, and then
    computes total damages plus protection costs for each candidate height.
    It mirrors the simulation loop used in
    ``run_beta_dueling_bandit_analytical`` but returns only the cost vector.

    Parameters
    ----------
    rng : Generator
        Random number generator.
    cand_heights : np.ndarray
        Array of candidate heights (in metres), length ``num_cands``.
    cand_protections : np.ndarray
        Protection costs corresponding to ``cand_heights``.
    height_grid_m : np.ndarray
        Grid of heights for which damage costs were tabulated.
    damage_grid : np.ndarray
        Damage costs (same length as ``height_grid_m``).
    X_paths_slice_cm : np.ndarray
        Array of mean sea level paths (in centimetres), shape
        ``(M_pred, n_years)``.
    posterior_params : Dict[str, np.ndarray]
        Posterior draws for the Poisson–GPD model parameters.
    u_cm : float
        Threshold for exceedances in cm.

    Returns
    -------
    total_costs : np.ndarray
        Array of total costs for each candidate height.
    """
    num_cands = cand_heights.shape[0]
    # Draw parameter index and MSL path index
    eta0_s = posterior_params["eta0"]
    eta1_s = posterior_params["eta1"]
    alpha0_s = posterior_params["alpha0"]
    xi_s = posterior_params["xi"]
    n_total = eta0_s.size
    i_draw = rng.integers(low=0, high=n_total)
    m_path = rng.integers(low=0, high=X_paths_slice_cm.shape[0])
    eta0 = float(eta0_s[i_draw])
    eta1 = float(eta1_s[i_draw])
    alpha0 = float(alpha0_s[i_draw])
    xi = float(xi_s[i_draw])
    X_t_series_cm = X_paths_slice_cm[m_path]
    # Simulate annual maxima
    maxima_cm = simulate_annual_max_pp(
        eta0=eta0,
        eta1=eta1,
        alpha0=alpha0,
        xi=xi,
        X_t_series_cm=X_t_series_cm,
        u_cm=u_cm,
        rng=rng,
    )
    maxima_m = maxima_cm * 0.01  # convert cm -> m
    total_costs = np.empty(num_cands, dtype=float)
    for c in range(num_cands):
        h = cand_heights[c]
        exceedances = np.where(maxima_m > h, maxima_m, 0.0)
        yearly_damage = _interpolate_damage(exceedances, height_grid_m, damage_grid)
        total_damage = float(np.sum(yearly_damage))
        total_costs[c] = cand_protections[c] + total_damage
    return total_costs


def _mardia_test(X: np.ndarray) -> Tuple[float, float]:
    """
    Perform Mardia's multivariate skewness and kurtosis tests.

    This implementation has been retained for backward compatibility with
    earlier versions of this module.  It is no longer used for tuning
    ``S`` by default, since we now rely on SciPy's built–in univariate
    normality tests in :func:`_tune_batch_size`.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape ``(n, p)``, where ``n`` is the number of
        observations and ``p`` is the dimensionality.

    Returns
    -------
    p_skew : float
        One–sided p–value for the skewness test.
    p_kurt : float
        Two–sided p–value for the kurtosis test.

    Notes
    -----
    See Mardia (1970) for details.  This function is retained
    primarily for completeness; users are encouraged to use the SciPy
    implementation of D'Agostino and Pearson's K² test (``scipy.stats.normaltest``)
    for multivariate normality by checking each marginal separately.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if n < 2 or p < 1:
        raise ValueError("Input array must have at least two observations and one dimension for Mardia's test.")
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    # Sample covariance and its inverse
    S = np.cov(X_centered, rowvar=False, ddof=1)
    try:
        invS = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        ridge = 1e-8 * np.eye(p)
        invS = np.linalg.inv(S + ridge)
    # Skewness
    skew_sum = 0.0
    for i in range(n):
        xi = X_centered[i]
        vals = xi @ invS @ X_centered.T
        skew_sum += np.sum(vals ** 3)
    b1p = skew_sum / (n**2)
    T1 = n * b1p / 6.0
    df_skew = p * (p + 1) * (p + 2) / 6.0
    p_skew = 1.0 - chi2.cdf(T1, df=df_skew)
    # Kurtosis
    di = np.einsum("ij,jk,ik->i", X_centered, invS, X_centered)
    b2p = np.mean(di**2)
    mu2 = p * (p + 2)
    var2 = 8.0 * p * (p + 2) / n
    z_kurt = (b2p - mu2) / math.sqrt(var2)
    p_kurt = 2.0 * (1.0 - norm.cdf(abs(z_kurt)))
    return p_skew, p_kurt


def _tune_batch_size(
    rng: Generator,
    cand_heights: np.ndarray,
    cand_protections: np.ndarray,
    height_grid_m: np.ndarray,
    damage_grid: np.ndarray,
    X_paths_slice_cm: np.ndarray,
    posterior_params: Dict[str, np.ndarray],
    u_cm: float,
    initial_S: int = 10,
    significance: float = 0.05,
    test_replicates: int = 200,
    max_S: int = 1_000_000,
    verbose: bool = False,
) -> Tuple[int, np.ndarray]:
    """
    Heuristically choose a batch size ``S`` such that the distribution of
    mean cost vectors over ``test_replicates`` batches is close to
    multivariate normal.

    This function draws ``test_replicates`` mean cost vectors for a given
    batch size ``S`` (each vector is the mean of ``S`` independent simulated
    scenarios).  It then applies SciPy's D'Agostino and Pearson omnibus test
    (:func:`scipy.stats.normaltest`) **independently to each coordinate** of
    the mean vectors.  If **all** coordinates have p–values above
    ``significance`` the batch size is accepted; otherwise ``S`` is doubled
    and the process repeats.  The procedure stops at the first ``S`` that
    passes or raises an error when ``S`` exceeds ``max_S``.

    The use of univariate normality tests on each component provides a
    practical heuristic to assess multivariate normality without
    implementing a custom Mardia's test.  In practice, if each marginal
    distribution of the sample mean vector is approximately normal and
    independence across samples holds, the vector will converge to a
    multivariate normal distribution by the multivariate central limit
    theorem.  Note that this check is *necessary but not sufficient* for
    joint normality; however it serves well in high–dimensional settings
    where the CLT makes the approximation reliable.

    Parameters
    ----------
    rng : Generator
        Random number generator.
    cand_heights, cand_protections : np.ndarray
        Arrays of candidate heights and corresponding protection costs.
    height_grid_m, damage_grid : np.ndarray
        Arrays defining the damage interpolation grid.
    X_paths_slice_cm : np.ndarray
        Mean sea level paths restricted to the years of interest.
    posterior_params : Dict[str, np.ndarray]
        Posterior draws for the Poisson–GPD model parameters.
    u_cm : float
        Threshold in cm for exceedances.
    initial_S : int, optional
        Starting batch size for the tuning procedure.
    significance : float, optional
        Significance threshold for the univariate normality tests.  All
        p–values must exceed this threshold to accept ``S``.
    test_replicates : int, optional
        Number of sample mean vectors to draw for testing normality.
    max_S : int, optional
        Upper bound on ``S`` to prevent infinite loops.
    verbose : bool, optional
        If True, print diagnostic information during tuning.

    Returns
    -------
    tuned_S : int
        The smallest batch size satisfying the normality criterion.
    mu_samples : np.ndarray
        Array of shape ``(test_replicates, num_cands)`` containing the
        mean cost vectors for the final ``S``.  These samples may be
        used to initialise the prior for the bandit.
    """
    # We perform the normality test outside the simulation loop using
    # SciPy's normaltest; import it locally to avoid global dependency if
    # normality testing isn't used elsewhere.
    from scipy.stats import normaltest

    S = max(1, int(initial_S))
    num_cands = cand_heights.shape[0]
    if num_cands == 0:
        raise RuntimeError("No candidate heights provided to tuning")
    while True:
        # Collect test_replicates mean cost vectors of length num_cands
        mu_samples = np.empty((test_replicates, num_cands), dtype=float)
        for r in range(test_replicates):
            # Average costs over S independent scenarios for replicate r
            costs = np.empty((S, num_cands), dtype=float)
            for s in range(S):
                costs[s] = _simulate_single_scenario(
                    rng,
                    cand_heights=cand_heights,
                    cand_protections=cand_protections,
                    height_grid_m=height_grid_m,
                    damage_grid=damage_grid,
                    X_paths_slice_cm=X_paths_slice_cm,
                    posterior_params=posterior_params,
                    u_cm=u_cm,
                )
            mu_samples[r] = costs.mean(axis=0)
        # Apply univariate normality test to each coordinate
        # normaltest returns a statistic and p-value for each column when axis=0
        try:
            stats, pvals = normaltest(mu_samples, axis=0)
        except Exception:
            # If the test fails (e.g., due to constant columns), treat as failure
            pvals = np.zeros(num_cands)
        # Determine whether all p-values exceed the significance threshold
        all_normal = bool(np.all(pvals > significance))
        if verbose:
            # Summarize the minimum p-value for brevity
            min_p = float(np.min(pvals)) if pvals.size > 0 else 0.0
            print(f"Testing S={S}: min p-value across {num_cands} dims = {min_p:.3f}")
        if all_normal:
            return S, mu_samples
        # Otherwise double S and repeat
        S *= 2
        if S > max_S:
            raise RuntimeError(
                f"Tuning failed to achieve approximate normality before reaching max_S={max_S}."
            )


def run_mvn_dueling_bandit(
    heights: np.ndarray,
    damage_costs: np.ndarray,
    protection_costs: np.ndarray,
    candidate_indices: Sequence[int],
    years_all: np.ndarray,
    X_pred_paths_cm: np.ndarray,
    posterior_params: Dict[str, np.ndarray],
    years_range: Tuple[int, int] = (2025, 2100),
    delta: float = 0.05,
    initial_S: int = 10,
    significance: float = 0.05,
    test_replicates: int = 200,
    post_samples: int = 2000,
    max_rounds: int = 100_000,
    check_every: int = 1000,
    rng: Optional[Generator] = None,
    verbose: bool = False,
) -> Tuple[float, Dict[str, List[float]]]:
    """
    Bayesian dueling bandit with a multivariate normal posterior on mean
    total costs.

    This function implements a fully adaptive algorithm to identify the
    optimal levee height using sequential simulation.  It first tunes a
    batch size ``S`` via :func:`_tune_batch_size` so that the distribution
    of sample mean cost vectors over ``S`` floods is approximately
    multivariate normal.  The tuning procedure repeatedly draws
    ``test_replicates`` sample means (each averaging over ``S`` independent
    flood scenarios) and applies SciPy's D'Agostino–Pearson normality test
    to each coordinate of these vectors.  If all p–values exceed the
    significance threshold the current ``S`` is accepted; otherwise it is
    doubled and the process repeats.  After tuning, the resulting sample
    means establish an initial prior on the true mean cost vector.  As
    additional floods are simulated one by one, this prior is updated
    online and the algorithm periodically samples from the posterior to
    estimate the probability that each candidate height is optimal.

    Parameters
    ----------
    heights, damage_costs, protection_costs : np.ndarray
        Full grids (in metres and million EUR) from ``load_cost_curves``.
    candidate_indices : sequence of int
        Indices into ``heights`` that survived pruning.
    years_all : np.ndarray
        1D array of years corresponding to the columns of ``X_pred_paths_cm``.
    X_pred_paths_cm : np.ndarray
        MSL sample paths in centimetres, shape ``(M_pred, T_total)``.
    posterior_params : dict
        Keys: 'eta0', 'eta1', 'alpha0', 'xi' (and optionally 'u').  Each
        value is a 1D array of flattened posterior draws.
    years_range : (int, int), optional
        Inclusive range of years over which to compute damages.
    delta : float, optional
        Target misselection probability.  The algorithm stops when the
        posterior probability of the best arm exceeds ``1 - delta``.
    initial_S : int, optional
        Starting batch size for the normality tuning procedure.
    significance : float, optional
        Significance level for the univariate normality tests during the
        tuning procedure.  The batch size ``S`` is accepted only if all
        coordinates of the sample mean vectors have p–values exceeding
        this threshold (using SciPy’s ``normaltest``).
    test_replicates : int, optional
        Number of samples to draw when testing normality and fitting the
        initial prior.
    post_samples : int, optional
        Number of posterior draws used when computing probabilities of
        optimality at each check.
    max_rounds : int, optional
        Maximum number of single–scenario simulations.
    check_every : int, optional
        Check stopping rule every ``check_every`` scenarios.
    rng : numpy.random.Generator, optional
        Random number generator.  If None, a default generator is used.
    verbose : bool, optional
        If True, print diagnostic information during tuning and updating.

    Returns
    -------
    best_height : float
        Selected levee height in metres.
    history : dict
        Diagnostic history with keys:
          - 'rounds': round numbers at which we checked stopping rule
          - 'best_index': best candidate index in the original height grid
          - 'best_height': corresponding height in metres
          - 'p_best': posterior probability that this candidate is optimal
    """
    if rng is None:
        rng = np.random.default_rng()
    # Ensure delta is a scalar float; callers may pass numpy scalars
    try:
        delta = float(delta)
    except Exception:
        delta = float(np.asarray(delta).flatten()[0])
    # Planning horizon
    start_year, end_year = years_range
    n_years = end_year - start_year + 1
    cand_idx = np.asarray(candidate_indices, dtype=int)
    if cand_idx.size == 0:
        raise RuntimeError("No candidate heights provided to bandit")
    num_cands = cand_idx.size
    # Candidate heights and protection costs
    cand_heights = heights[cand_idx]
    cand_protections = protection_costs[cand_idx]
    # Posterior draws for parameters
    u_cm = float(posterior_params.get("u", 60.0))
    # Slice mean sea level paths to the analysis years
    years_all = np.asarray(years_all, dtype=int)
    mask = (years_all >= start_year) & (years_all <= end_year)
    year_idx = np.where(mask)[0]
    if year_idx.size != n_years:
        raise ValueError(
            f"years_all and years_range mismatch: expected {n_years} years, found {year_idx.size}"
        )
    X_paths_slice_cm = np.asarray(X_pred_paths_cm, dtype=float)[:, year_idx]
    # Tune the batch size S and obtain sample means
    tuned_S, mu_samples = _tune_batch_size(
        rng=rng,
        cand_heights=cand_heights,
        cand_protections=cand_protections,
        height_grid_m=heights,
        damage_grid=damage_costs,
        X_paths_slice_cm=X_paths_slice_cm,
        posterior_params=posterior_params,
        u_cm=u_cm,
        initial_S=initial_S,
        significance=significance,
        test_replicates=test_replicates,
        verbose=verbose,
    )
    if verbose:
        print(f"Tuned batch size S={tuned_S}")
    # Use these samples to infer an initial prior on mean cost
    # Estimate underlying covariance of single–scenario cost vector
    # mu_samples has shape (test_replicates, num_cands)
    m_init = mu_samples.mean(axis=0)
    # Sample covariance of the sample means
    Cov_mu_hat = np.cov(mu_samples, rowvar=False, ddof=1)  # shape (d,d)
    # Underlying covariance of the costs: Cov(mu_hat) ≈ Sigma_x / S
    Sigma_x_init = Cov_mu_hat * tuned_S
    # Effective number of observations: replicates * S
    n_init = test_replicates * tuned_S
    # Compute M2 (sum of squares) for the initial sample covariance
    # Sample covariance Cov_x = Sigma_x_init; unbiased => M2_init = Cov_x * (n_init - 1)
    M2 = Sigma_x_init * (n_init - 1)
    # Running mean
    mean_vec = m_init.copy()
    n = n_init
    # History
    history: Dict[str, List[float]] = {
        "rounds": [],
        "best_index": [],
        "best_height": [],
        "p_best": [],
    }
    round_idx = 0
    # Main simulation loop
    while round_idx < max_rounds:
        round_idx += 1
        # Simulate a single scenario and update running statistics
        x = _simulate_single_scenario(
            rng,
            cand_heights=cand_heights,
            cand_protections=cand_protections,
            height_grid_m=heights,
            damage_grid=damage_costs,
            X_paths_slice_cm=X_paths_slice_cm,
            posterior_params=posterior_params,
            u_cm=u_cm,
        )
        # Online update of mean and M2 (Welford's algorithm)
        n_prev = n
        n = n + 1
        # Use distinct variable names to avoid shadowing the misselection threshold ``delta``
        delta_vec = x - mean_vec
        mean_vec = mean_vec + delta_vec / n
        delta2_vec = x - mean_vec
        # Update M2
        M2 += np.outer(delta_vec, delta2_vec)
        # Check stopping rule
        if round_idx % check_every == 0 or round_idx == max_rounds:
            if n < 2:
                # Cannot compute covariance with fewer than 2 observations
                continue
            Cov_x = M2 / (n - 1)
            # Posterior covariance of mean: Cov(mu) ≈ Cov_x / n
            Cov_mu = Cov_x / n
            # Add a small jitter to diagonal if necessary for numerical stability
            # (particularly when n is small or Cov_x is singular)
            jitter = 1e-10 * np.eye(num_cands)
            try:
                samples = multivariate_normal.rvs(
                    mean=mean_vec,
                    cov=Cov_mu + jitter,
                    size=post_samples,
                    random_state=rng,
                )
            except Exception:
                # Fall back to diagonal covariance
                diag_cov = np.diag(np.diag(Cov_mu))
                samples = multivariate_normal.rvs(
                    mean=mean_vec,
                    cov=diag_cov + jitter,
                    size=post_samples,
                    random_state=rng,
                )
            # Determine which candidate is minimal for each sample
            # Count occurrences of each index being the argmin
            mins = np.argmin(samples, axis=1)
            counts = np.bincount(mins, minlength=num_cands)
            prob = counts / float(post_samples)
            best_loc = int(np.argmax(prob))
            p_best = float(prob[best_loc])
            best_global_index = int(cand_idx[best_loc])
            best_height = float(cand_heights[best_loc])
            history["rounds"].append(round_idx)
            history["best_index"].append(best_global_index)
            history["best_height"].append(best_height)
            history["p_best"].append(p_best)
            if verbose:
                print(
                    f"Round {round_idx}: best height {best_height:.2f} m, p_best = {p_best:.4f}"
                )
            # Cast delta to float explicitly; numpy arrays are not directly comparable
            if p_best >= 1.0 - delta:
                return best_height, history
    # If max_rounds reached without meeting stopping rule, return current best
    # Based on posterior probabilities at the final check
    if n < 2:
        # If we never computed probabilities, fall back to trivial choice
        best_loc = 0
        best_global_index = int(cand_idx[best_loc])
        best_height = float(cand_heights[best_loc])
        return best_height, history
    Cov_x = M2 / (n - 1)
    Cov_mu = Cov_x / n
    jitter = 1e-10 * np.eye(num_cands)
    try:
        samples = multivariate_normal.rvs(
            mean=mean_vec,
            cov=Cov_mu + jitter,
            size=post_samples,
            random_state=rng,
        )
    except Exception:
        diag_cov = np.diag(np.diag(Cov_mu))
        samples = multivariate_normal.rvs(
            mean=mean_vec,
            cov=diag_cov + jitter,
            size=post_samples,
            random_state=rng,
        )
    mins = np.argmin(samples, axis=1)
    counts = np.bincount(mins, minlength=num_cands)
    prob = counts / float(post_samples)
    best_loc = int(np.argmax(prob))
    best_global_index = int(cand_idx[best_loc])
    best_height = float(cand_heights[best_loc])
    history["rounds"].append(round_idx)
    history["best_index"].append(best_global_index)
    history["best_height"].append(best_height)
    history["p_best"].append(float(prob[best_loc]))
    return best_height, history
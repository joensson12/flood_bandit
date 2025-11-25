import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def simulate_annual_max_pp(
    eta0: float,
    eta1: float,
    alpha0: float,
    xi: float,
    X_t_series_cm: np.ndarray,
    u_cm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate one trajectory of annual maxima (in cm) for a given parameter
    draw and mean sea level (MSL) path, using the tested Poisson–GPD model.

    This is exactly your tested code, adapted to use centimetres.

    Parameters
    ----------
    eta0, eta1 : float
        Poisson intensity regression parameters.
    alpha0 : float
        Log-scale parameter for the GPD: sigma = exp(alpha0).
    xi : float
        GPD shape parameter.
    X_t_series_cm : np.ndarray
        Latent MSL time series in centimetres for the target years.
    u_cm : float
        POT threshold in centimetres (e.g. 60.0).
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
        # Intensity for year j (lambda_t = exp(eta0 + eta1 * X_t))
        lam = np.exp(eta0 + eta1 * X_t_series_cm[j])
        N = rng.poisson(lam)
        if N <= 0:
            # No exceedances: max is at threshold (below-threshold years)
            maxima_cm[j] = u_cm
        else:
            U = rng.uniform(size=N)
            if abs(xi) < 1e-6:
                # Approximate GPD with xi ~ 0 by exponential
                Z = -sigma * np.log(1.0 - U)
            else:
                # Inverse CDF for GPD: z = sigma/xi * ((1 - U)^(-xi) - 1)
                Z = sigma / xi * ((1.0 - U) ** (-xi) - 1.0)
            maxima_cm[j] = u_cm + Z.max()

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
    w = np.asarray(exceedances_m, dtype=float)
    w[w < 0.0] = 0.0
    w[w>12.0]=12.0
    return np.interp(w, height_grid_m, damage_grid)


def run_beta_dueling_bandit_analytical(
    heights: np.ndarray,
    damage_costs: np.ndarray,
    protection_costs: np.ndarray,
    candidate_indices: Sequence[int],
    years_all: np.ndarray,
    X_pred_paths_cm: np.ndarray,
    posterior_params: Dict[str, np.ndarray],
    years_range: Tuple[int, int] = (2025, 2100),
    delta: float = 0.05,
    max_rounds: int = 100000,
    check_every: int = 1000,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> Tuple[float, Dict[str, List[float]]]:
    """
    Bayesian dueling-bandit algorithm with Beta priors on pairwise win
    probabilities p_ij = P(X_ij = 1), where X_ij = 1 if height i has
    lower total cost than height j in a random scenario.

    For each round m:
      1. Draw a posterior parameter sample (eta0, eta1, alpha0, xi).
      2. Draw a mean-sea-level path X_t for the chosen years.
      3. Simulate annual maxima with the Poisson–GPD model.
      4. Compute total cost for each candidate height (protection + damage).
      5. For each pair (i, j), update a Beta prior based on which height
         had lower total cost in this scenario.

    The posterior predictive probability that i beats j in a new scenario
    is then P(X_ij = 1 | data) = alpha_ij / (alpha_ij + beta_ij).

    Stopping rule:
      - For each candidate i, compute q_i = min_{j != i} P(X_ij = 1 | data).
      - Let i* = arg max_i q_i.
      - Stop if q_{i*} >= 1 - delta.

    Parameters
    ----------
    heights, damage_costs, protection_costs : np.ndarray
        Full grids (in metres and million EUR) from load_cost_curves.

    candidate_indices : sequence of int
        Indices into heights that survived pruning.

    years_all : np.ndarray
        1D array of years corresponding to the columns of X_pred_paths_cm.

    X_pred_paths_cm : np.ndarray
        MSL sample paths in centimetres, shape (M_pred, T_total).

    posterior_params : dict
        Keys: 'eta0', 'eta1', 'alpha0', 'xi' (and optionally 'u' for
        threshold in centimetres). Each value is a 1D array of flattened
        posterior draws.

    years_range : (int, int)
        Inclusive range of years over which to compute damages.

    delta : float
        Target misselection probability. The stopping rule uses
        q_best >= 1 - delta.

    max_rounds : int
        Maximum number of simulated scenarios.

    check_every : int
        Evaluate the stopping rule every 'check_every' rounds.

    rng : np.random.Generator, optional
        Random number generator. If None, a default generator is used.

    verbose : bool
        If True, print diagnostic information when checking.

    Returns
    -------
    best_height : float
        Selected levee height in metres.

    history : dict
        Diagnostic history with keys:
          - 'rounds': round numbers at which we checked stopping rule
          - 'best_index': best candidate index in the original height grid
          - 'best_height': corresponding height in metres
          - 'min_pair_prob': q_best = min_j P(X_best,j = 1 | data)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Planning horizon
    start_year, end_year = years_range
    n_years = end_year - start_year + 1

    cand_idx = np.asarray(candidate_indices, dtype=int)
    if cand_idx.size == 0:
        raise RuntimeError("No candidate heights provided to bandit")

    num_cands = cand_idx.size

    # Candidate heights and protection costs
    cand_heights = heights[cand_idx]          # (C,)
    cand_protections = protection_costs[cand_idx]  # (C,)

    # Posterior draws for (eta0, eta1, alpha0, xi)
    eta0_s = np.asarray(posterior_params["eta0"]).reshape(-1)
    eta1_s = np.asarray(posterior_params["eta1"]).reshape(-1)
    alpha0_s = np.asarray(posterior_params["alpha0"]).reshape(-1)
    xi_s = np.asarray(posterior_params["xi"]).reshape(-1)
    n_total = eta0_s.size

    # Threshold (cm)
    u_cm = float(posterior_params.get("u", 60.0))

    # Select relevant years from MSL paths
    years_all = np.asarray(years_all, dtype=int)
    mask = (years_all >= start_year) & (years_all <= end_year)
    year_idx = np.where(mask)[0]
    if year_idx.size != n_years:
        raise ValueError(
            f"years_all and years_range mismatch: expected {n_years} years, "
            f"found {year_idx.size}"
        )

    X_paths_slice_cm = np.asarray(X_pred_paths_cm, dtype=float)[:, year_idx]
    M_pred = X_paths_slice_cm.shape[0]

    # Beta priors for p_ij = P(X_ij = 1); use Beta(1,1) flat prior
    alpha = np.ones((num_cands, num_cands), dtype=float)
    beta = np.ones((num_cands, num_cands), dtype=float)
    # diagonal entries are not used
    np.fill_diagonal(alpha, 0.0)
    np.fill_diagonal(beta, 0.0)

    # History
    history: Dict[str, List[float]] = {
        "rounds": [],
        "best_index": [],
        "best_height": [],
        "min_pair_prob": [],
    }

    # For damage interpolation
    height_grid_m = heights
    damage_grid = damage_costs

    round_idx = 0
    while round_idx < max_rounds:
        round_idx += 1

        # --- 1. Draw posterior parameters and an MSL path ---
        i_draw = rng.integers(low=0, high=n_total)
        m_path = rng.integers(low=0, high=M_pred)

        eta0 = float(eta0_s[i_draw])
        eta1 = float(eta1_s[i_draw])
        alpha0 = float(alpha0_s[i_draw])
        xi = float(xi_s[i_draw])

        X_t_series_cm = X_paths_slice_cm[m_path]  # (n_years,)

        # --- 2. Simulate annual maxima (cm) and convert to metres ---
        maxima_cm = simulate_annual_max_pp(
            eta0=eta0,
            eta1=eta1,
            alpha0=alpha0,
            xi=xi,
            X_t_series_cm=X_t_series_cm,
            u_cm=u_cm,
            rng=rng,
        )
        maxima_m = maxima_cm * 0.01  # cm -> m

        # --- 3. Compute total costs per candidate ---
        #    total_cost = protection + sum yearly damage
        total_costs = np.empty(num_cands, dtype=float)
        for c in range(num_cands):
            h = cand_heights[c]
            exceedances = np.where(maxima_m > h, maxima_m, 0)
            yearly_damage = _interpolate_damage(exceedances, height_grid_m, damage_grid)
            total_damage = float(np.sum(yearly_damage))
            total_costs[c] = cand_protections[c] + total_damage

        # --- 4. Update Beta(a_ij, b_ij) for each ordered pair ---
        #     We use:
        #       X_ij = 1 if cost_i < cost_j
        #       X_ij = 0 if cost_i > cost_j
        #       tie -> 0.5 to each side
        for i in range(num_cands):
            for j in range(i + 1, num_cands):
                ci = total_costs[i]
                cj = total_costs[j]
                if ci < cj:
                    alpha[i, j] += 1.0
                    beta[j, i] += 1.0
                elif ci > cj:
                    alpha[j, i] += 1.0
                    beta[i, j] += 1.0
                else:
                    # tie: half-credit to each
                    alpha[i, j] += 0.5
                    beta[i, j] += 0.5
                    alpha[j, i] += 0.5
                    beta[j, i] += 0.5

        # --- 5. Check stopping rule every 'check_every' rounds ---
        if round_idx % check_every == 0 or round_idx == max_rounds:
            # Posterior predictive probability that i beats j:
            # P(X_ij = 1 | data) = alpha_ij / (alpha_ij + beta_ij)
            with np.errstate(divide="ignore", invalid="ignore"):
                p_pred = alpha / (alpha + beta)
                p_pred[np.isnan(p_pred)] = 0.5  # if alpha=beta=0 just treat as 0.5

            # For each candidate i, compute q_i = min_{j != i} P(X_ij = 1 | data)
            q = np.empty(num_cands, dtype=float)
            for i in range(num_cands):
                mask_j = np.ones(num_cands, dtype=bool)
                mask_j[i] = False
                q[i] = float(np.min(p_pred[i, mask_j]))

            best_loc = int(np.argmax(q))
            q_best = float(q[best_loc])

            best_global_index = int(cand_idx[best_loc])
            best_height = float(cand_heights[best_loc])

            history["rounds"].append(round_idx)
            history["best_index"].append(best_global_index)
            history["best_height"].append(best_height)
            history["min_pair_prob"].append(q_best)

            if verbose:
                print(
                    f"Round {round_idx}: best height {best_height:.2f} m, "
                    f"min pairwise win prob q_best = {q_best:.4f}"
                )

            # Stopping condition: q_best >= 1 - delta
            if q_best >= 1.0 - delta:
                return best_height, history

    # If max_rounds reached without meeting stopping rule, return current best
    # based on q_i.
    with np.errstate(divide="ignore", invalid="ignore"):
        p_pred = alpha / (alpha + beta)
        p_pred[np.isnan(p_pred)] = 0.5
    q = np.empty(num_cands, dtype=float)
    for i in range(num_cands):
        mask_j = np.ones(num_cands, dtype=bool)
        mask_j[i] = False
        q[i] = float(np.min(p_pred[i, mask_j]))
    best_loc = int(np.argmax(q))
    best_height = float(cand_heights[best_loc])
    return best_height, history

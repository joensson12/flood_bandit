"""
GPU‑accelerated levee bandit
===========================

This module provides a stripped‑down and GPU‑accelerated version of the
original ``optimal_levee_bandit`` implementation.  The goal is to remove
all extraneous functionality (such as deterministic pruning and the
standard CPU‑based bandit) and concentrate on a high‑throughput Monte
Carlo procedure that runs on a CUDA‑capable device using CuPy.  The
algorithm mirrors the pure exploration strategy from the accompanying
LaTeX document: given a discrete set of candidate levee heights,
simulate flood scenarios under a Poisson–GPD point‑process model and
maintain uniform confidence intervals until a single height is clearly
optimal.

Only a few functions are exposed:

* :func:`load_cost_curves` – read damage and protection cost curves
  from the published ``.tab`` files.
* :func:`simulate_annual_max_pp_batch_gpu` – vectorised simulation of
  annual maxima on the GPU for a batch of posterior parameter draws and
  latent mean sea level (MSL) paths.
* :func:`run_bandit_gpu` – the core pure exploration routine that
  repeatedly calls the simulator, computes damages for each candidate
  levee height, and stops once the empirically best arm's confidence
  interval lies below those of all competitors.

This module depends on CuPy.  If a CUDA‑capable device is not
available at run time, the functions will raise an import error.  All
computations within the bandit loop are carried out in single precision
(``cp.float32``) to reduce memory pressure; the final results are
returned to the host (NumPy) as Python floats.

Example usage (see ``run_gpu_bandit.ipynb`` for a more complete
demonstration)::

    from optimal_levee_bandit_gpu import load_cost_curves, run_bandit_gpu
    import numpy as np

    # Load cost curves for a city
    heights, damage_costs, protection_costs = load_cost_curves(
        'data/Damage_cost_curves.tab', 'data/Protection_cost_curves_high_estimate.tab', 'Halmstad'
    )
    # Load predictive posterior and latent MSL paths from NPZ
    pp = np.load('data/pp_inputs_halmsdad_pp_mixture_2025_2100.npz')
    posterior_params = {
        'eta0': pp['eta0'],
        'eta1': pp['eta1'],
        'alpha0': pp['alpha0'],
        'xi': pp['xi'],
        'u': float(pp['u']),
    }
    years_future   = pp['years_future']
    X_future_paths = pp['X_future_paths']  # shape (M_pred, n_years)

    # Use all heights as candidates (no deterministic pruning)
    candidate_indices = list(range(len(heights)))

    # Run the GPU bandit
    best_height, history = run_bandit_gpu(
        heights=heights,
        damage_costs=damage_costs,
        protection_costs=protection_costs,
        candidate_indices=candidate_indices,
        years_all=years_future,
        X_pred_paths_cm=X_future_paths,
        posterior_params=posterior_params,
        years_range=(2025, 2100),
        delta=0.05,
        max_rounds=200000,
        verbose=True,
        chunk_size=50000,
        check_every=50000,
    )
    print(f"Selected design height: {best_height:.2f} m")

"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np

try:
    import cupy as cp
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Cupy is required for optimal_levee_bandit_gpu. Ensure a CUDA‑capable device is available."
    ) from e


def load_cost_curves(
    damage_file: str,
    protection_file: str,
    city: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load damage and protection cost curves for a given city.

    This helper replicates the logic from the original
    ``optimal_levee_bandit`` implementation.  It reads tab‑separated
    files where each line corresponds to a city and the columns
    following the metadata contain costs at discrete flood heights.

    Parameters
    ----------
    damage_file : str
        Path to the damage cost curves ``.tab`` file.

    protection_file : str
        Path to the protection cost curves ``.tab`` file.

    city : str
        City name to select (case‑insensitive).  If multiple rows match,
        the first is returned.

    Returns
    -------
    heights : np.ndarray
        A one‑dimensional array of flood heights (m) at which costs are
        reported (typically 0–12 m in 0.5 m increments).

    damage_costs : np.ndarray
        Direct damage costs (million EUR) corresponding to ``heights``.

    protection_costs : np.ndarray
        Levee construction costs (million EUR) corresponding to ``heights``.

    Raises
    ------
    ValueError
        If the city is not found in either file or if the cost curves
        have different lengths.
    """

    def _parse_file(path: str) -> Dict[str, List[float]]:
        costs: Dict[str, List[float]] = {}
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("/*"):
                    continue
                parts = line.split("\t")
                if len(parts) < 6:
                    continue
                name = parts[1].strip().lower()
                try:
                    numeric = [float(x) for x in parts[5:]]
                except ValueError:
                    continue
                costs[name] = numeric
        return costs

    damage_map = _parse_file(damage_file)
    protection_map = _parse_file(protection_file)
    key = city.strip().lower()
    if key not in damage_map:
        raise ValueError(f"City '{city}' not found in damage cost file {damage_file}")
    if key not in protection_map:
        raise ValueError(f"City '{city}' not found in protection cost file {protection_file}")
    damage_costs = np.asarray(damage_map[key], dtype=float)
    protection_costs = np.asarray(protection_map[key], dtype=float)
    if damage_costs.shape[0] != protection_costs.shape[0]:
        raise ValueError(
            f"Damage and protection curves have different lengths for '{city}'"
        )
    num_levels = len(damage_costs)
    max_height = 12.0
    if num_levels > 1:
        step = max_height / (num_levels - 1)
    else:
        step = 0.0
    heights = np.linspace(0.0, max_height, num_levels)
    return heights, damage_costs, protection_costs


import cupy as cp

def simulate_annual_max_pp_batch_gpu(
    eta0_batch,
    eta1_batch,
    alpha0_batch,
    xi_batch,
    X_batch_cm,
    u_cm: float,
    max_lambda: float = 1e5,
):
    """
    Vectorised Poisson–GPD annual-max simulation for a batch of scenarios on GPU.

    Parameters
    ----------
    eta0_batch, eta1_batch, alpha0_batch, xi_batch : array-like, shape (B,)
        Posterior parameter draws for each of B scenarios.
    X_batch_cm : array-like, shape (B, T)
        Latent MSL time series in cm for each scenario (B) and year (T).
    u_cm : float
        POT threshold in cm.
    max_lambda : float
        Cap on Poisson intensity λ to avoid overflow.

    Returns
    -------
    maxima_cm : cupy.ndarray, shape (B, T)
        Annual maxima in cm for each scenario and year.
    """
    # Move everything to GPU
    eta0_batch   = cp.asarray(eta0_batch,   dtype=cp.float64)
    eta1_batch   = cp.asarray(eta1_batch,   dtype=cp.float64)
    alpha0_batch = cp.asarray(alpha0_batch, dtype=cp.float64)
    xi_batch     = cp.asarray(xi_batch,     dtype=cp.float64)
    X_batch_cm   = cp.asarray(X_batch_cm,   dtype=cp.float64)  # (B, T)

    B, T = X_batch_cm.shape

    # Broadcast sigma and xi to shape (B, T)
    sigma_bt = cp.exp(alpha0_batch)[:, None]       # (B, 1)
    xi_bt    = xi_batch[:, None]                   # (B, 1)
    sigma_bt = cp.broadcast_to(sigma_bt, (B, T))   # (B, T)
    xi_bt    = cp.broadcast_to(xi_bt,    (B, T))   # (B, T)

    # Poisson intensities λ_{b,t} = exp(η0_b + η1_b * X_{b,t})
    log_lam_bt = eta0_batch[:, None] + eta1_batch[:, None] * X_batch_cm  # (B, T)

    # Clip log λ to avoid numerical insanity
    log_lam_bt = cp.clip(log_lam_bt, a_min=-cp.inf, a_max=cp.log(max_lambda))
    lam_bt = cp.exp(log_lam_bt)  # (B, T)

    # Draw N_{b,t} ~ Poisson(λ_{b,t})
    N_bt = cp.random.poisson(lam_bt)  # (B, T)

    # Initialise maxima at the threshold u_cm
    maxima_cm = cp.full((B, T), u_cm, dtype=cp.float64)

    # Mask where there is at least one exceedance
    mask = N_bt > 0
    if not bool(mask.any()):
        return maxima_cm

    # Draw U_max ~ Beta(N, 1) via U^(1/N) with U ~ Uniform(0,1)
    U = cp.random.random(size=(B, T))
    U_max = cp.ones_like(U)
    U_max[mask] = U[mask] ** (1.0 / N_bt[mask])

    # Compute z_max via GPD inverse CDF for the maximum
    z_max = cp.zeros((B, T), dtype=cp.float64)

    # xi ≈ 0 (exponential case)
    small_mask = (cp.abs(xi_bt) < 1e-6) & mask
    if bool(small_mask.any()):
        z_max[small_mask] = -sigma_bt[small_mask] * cp.log(1.0 - U_max[small_mask])

    # xi != 0 case
    large_mask = (~small_mask) & mask
    if bool(large_mask.any()):
        # Full elementwise formula, then mask
        gpd_term = (sigma_bt / xi_bt) * (
            (1.0 - U_max) ** (-xi_bt) - 1.0
        )
        z_max[large_mask] = gpd_term[large_mask]

    # Set maxima (in cm) where there are exceedances
    maxima_cm[mask] = u_cm + z_max[mask]
    return maxima_cm



def run_bandit_gpu(
    heights: Sequence[float],
    damage_costs: Sequence[float],
    protection_costs: Sequence[float],
    candidate_indices: Sequence[int],
    years_all: Sequence[int],
    X_pred_paths_cm: Sequence[Sequence[float]],
    posterior_params: Dict[str, np.ndarray],
    years_range: Tuple[int, int] = (2025, 2100),
    delta: float = 0.05,
    max_rounds: int = 100000,
    verbose: bool = False,
    chunk_size: int = 50000,
    check_every: int = 50000,
) -> Tuple[float, Dict[str, List[float]]]:
    """Run the confidence‑based pure exploration bandit on a GPU.

    This function orchestrates the Monte Carlo sampling procedure.  It
    accepts full grids of heights, damage and protection costs, a list
    of candidate indices (which can be the full set if no pruning is
    desired), predictive posterior draws and latent MSL sample paths.
    Flood scenarios are simulated in large chunks on the GPU, and
    damages for all candidates are computed via linear interpolation on
    the GPU as well.  Sampling stops once the empirically best height
    has a confidence interval that lies strictly below those of all
    other candidates.

    Parameters
    ----------
    heights : sequence of float
        Monotonic grid of design heights (metres) corresponding to the
        cost curves.

    damage_costs : sequence of float
        Per‑flood damage costs (million EUR) on the same grid as
        ``heights``.

    protection_costs : sequence of float
        Present value of levee construction costs (million EUR) on the
        same grid as ``heights``.

    candidate_indices : sequence of int
        Indices of ``heights`` that should be considered.  For a no‑pruning
        strategy pass ``range(len(heights))``.

    years_all : sequence of int
        Array of calendar years corresponding to the columns of
        ``X_pred_paths_cm``.

    X_pred_paths_cm : array‑like, shape ``(M_pred, T_total)``
        Latent MSL sample paths in centimetres.  Only the years
        specified by ``years_range`` are extracted.

    posterior_params : dict
        Dictionary containing the posterior draws for ``eta0``, ``eta1``,
        ``alpha0`` and ``xi``, each as a one‑dimensional NumPy array of
        equal length.  Optionally include ``'u'`` to override the
        default threshold (60 cm).

    years_range : (int, int), optional
        Inclusive range of years over which to accumulate damages.

    delta : float, optional
        Desired misselection probability.  Controls the width of the
        confidence intervals via a union bound.

    max_rounds : int, optional
        Maximum number of flood scenarios to simulate.  If the stopping
        condition has not been met after this many draws, the
        algorithm returns the empirically best height.

    verbose : bool, optional
        If True, print diagnostic information at each confidence check.

    chunk_size : int, optional
        Number of scenarios to simulate in each GPU batch.  Larger
        values may improve throughput but consume more GPU memory.

    check_every : int, optional
        Number of scenarios between confidence interval checks.  Must
        divide ``chunk_size``.  When ``check_every`` is reached the
        algorithm recomputes the empirical means and radii and tests
        whether a single candidate is separated.

    Returns
    -------
    best_height : float
        Selected levee height (metres).

    history : dict
        Keys ``'rounds'``, ``'best_index'``, ``'best_height'``, and
        ``'radii'`` recording the progress of the bandit.  These
        sequences have length equal to the number of confidence checks
        performed.
    """
    # Convert scalar inputs
    heights = np.asarray(heights, dtype=float)
    damage_costs = np.asarray(damage_costs, dtype=float)
    protection_costs = np.asarray(protection_costs, dtype=float)
    years_all = np.asarray(years_all, dtype=int)
    X_pred_paths_cm = np.asarray(X_pred_paths_cm, dtype=float)

    # Determine horizon
    start_year, end_year = years_range
    n_years = end_year - start_year + 1

    cand_idx = np.asarray(candidate_indices, dtype=int)
    if cand_idx.size == 0:
        raise RuntimeError("No candidate heights provided to bandit")
    num_cands = cand_idx.size

    # Maximum per‑year damage cost for concentration bound
    d_max = float(np.max(damage_costs))
    D_max_total = n_years * d_max

    # Candidate heights and protection costs (NumPy and CuPy)
    cand_heights = heights[cand_idx]               # shape (C,)
    cand_protections = protection_costs[cand_idx]  # shape (C,)
    # Convert to GPU arrays (float32 for damage simulation, float64 for protections)
    cand_heights_cp = cp.asarray(cand_heights, dtype=cp.float64)
    cand_protections_cp = cp.asarray(cand_protections, dtype=cp.float64)

    # Posterior draws
    eta0_s = np.asarray(posterior_params["eta0"]).reshape(-1)
    eta1_s = np.asarray(posterior_params["eta1"]).reshape(-1)
    alpha0_s = np.asarray(posterior_params["alpha0"]).reshape(-1)
    xi_s = np.asarray(posterior_params["xi"]).reshape(-1)
    n_total = eta0_s.size
    if not (eta1_s.size == n_total == alpha0_s.size == xi_s.size):
        raise ValueError("Posterior parameter arrays must have the same length")

    # Threshold (cm)
    u_cm = float(posterior_params.get("u", 60.0))

    # Slice the latent MSL paths for the target years
    mask_years = (years_all >= start_year) & (years_all <= end_year)
    year_idx = np.where(mask_years)[0]
    if year_idx.size != n_years:
        raise ValueError(
            f"Expected {n_years} years between {start_year} and {end_year}, got {year_idx.size}"
        )
    # X_paths_slice_cm shape: (M_pred, n_years)
    X_paths_slice_cm_np = X_pred_paths_cm[:, year_idx]
    M_pred = X_paths_slice_cm_np.shape[0]
    # Transfer latent paths to GPU as float64
    X_paths_slice_cp = cp.asarray(X_paths_slice_cm_np, dtype=cp.float64)

    # Transfer posterior draws to GPU as float64
    cp_eta0 = cp.asarray(eta0_s, dtype=cp.float64)
    cp_eta1 = cp.asarray(eta1_s, dtype=cp.float64)
    cp_alpha0 = cp.asarray(alpha0_s, dtype=cp.float64)
    cp_xi = cp.asarray(xi_s, dtype=cp.float64)

    # Precompute damage interpolation grid on GPU
    height_grid_cp = cp.asarray(heights, dtype=cp.float64)
    damage_grid_cp = cp.asarray(damage_costs, dtype=cp.float64)

    # Bandit state on GPU
    cum_damage_cp = cp.zeros(num_cands, dtype=cp.float64)
    num_samples = 0

    # History on CPU
    history: Dict[str, List[float]] = {
        "rounds": [],
        "best_index": [],
        "best_height": [],
        "radii": [],
    }

    round_idx = 0
    # Ensure chunk_size and check_every are compatible
    if check_every > chunk_size:
        check_every = chunk_size
    if chunk_size % check_every != 0:
        raise ValueError("chunk_size must be an integer multiple of check_every")

    while round_idx < max_rounds:
        # Determine how many scenarios to simulate in this batch
        remaining = max_rounds - round_idx
        this_chunk = min(chunk_size, remaining)

        # Draw parameter and path indices on GPU
        # We sample using CuPy to avoid host‑device transfers
        draw_indices_cp = cp.random.randint(0, n_total, size=this_chunk, dtype=cp.int32)
        path_indices_cp = cp.random.randint(0, M_pred, size=this_chunk, dtype=cp.int32)

        # Gather posterior parameters for this batch
        eta0_batch = cp_eta0[draw_indices_cp]
        eta1_batch = cp_eta1[draw_indices_cp]
        alpha0_batch = cp_alpha0[draw_indices_cp]
        xi_batch = cp_xi[draw_indices_cp]

        # Gather MSL paths for this batch (shape: this_chunk x n_years)
        X_batch_cm = X_paths_slice_cp[path_indices_cp, :]

        # Simulate annual maxima in cm (GPU)
        maxima_batch_cm = simulate_annual_max_pp_batch_gpu(
            eta0_batch=eta0_batch,
            eta1_batch=eta1_batch,
            alpha0_batch=alpha0_batch,
            xi_batch=xi_batch,
            X_batch_cm=X_batch_cm,
            u_cm=u_cm,
        )  # shape: (this_chunk, n_years)
        # Convert to metres on GPU
        maxima_batch_m = maxima_batch_cm * 0.01  # cm → m

        # Compute exceedances for all candidates (shape: this_chunk, C, n_years)
        # If the flood height is below the levee, damage is zero
        exceedances = cp.where(
            maxima_batch_m[:, None, :] >= cand_heights_cp[None, :, None],
            maxima_batch_m[:, None, :],
            0.0,
        )

        # Flatten exceedances and interpolate damage on GPU
        flat_exceed = exceedances.reshape(-1)
        flat_damage = cp.interp(flat_exceed, height_grid_cp, damage_grid_cp)
        damage_matrix = flat_damage.reshape(this_chunk, num_cands, n_years)

        # Accumulate damage over time and scenarios
        # Sum over scenarios and years: result shape (C,)
        damage_sums = cp.sum(damage_matrix, axis=(0, 2))
        cum_damage_cp += damage_sums
        num_samples += this_chunk
        round_idx += this_chunk

        # Recompute confidence intervals periodically
        if round_idx % check_every == 0 or round_idx >= max_rounds:
            # Mean total cost per candidate (GPU)
            mean_total_cp = cand_protections_cp + cum_damage_cp / float(num_samples)
            # Transfer to CPU for argmin and comparisons
            mean_total = cp.asnumpy(mean_total_cp)

            # Confidence radius (Hoeffding + union bound)
            log_term = math.log(
                (2.0 * num_cands * round_idx * (round_idx + 1)) / max(delta, 1e-16)
            )
            r_S = D_max_total * math.sqrt(log_term / (2.0 * round_idx))

            # Identify empirically best candidate
            best_loc = int(np.argmin(mean_total))

            history["rounds"].append(round_idx)
            history["best_index"].append(int(cand_idx[best_loc]))
            history["best_height"].append(float(cand_heights[best_loc]))
            history["radii"].append(r_S)

            # Print all options with their current stats
            print(f"\nRound {round_idx} – all candidate options:")
            for j, (h, c) in enumerate(zip(cand_heights, mean_total)):
                lower = c - r_S
                upper = c + r_S
                print(
                    f"  option {j:2d}: idx={int(cand_idx[j])}, "
                    f"height={h:.2f} m, mean_total={c:.3f}, "
                    f"CI=[{lower:.3f}, {upper:.3f}]"
                )

            # Check separation: best CI strictly below all others
            mu_best = mean_total[best_loc]
            separated = True
            for j in range(num_cands):
                if j == best_loc:
                    continue
                mu_j = mean_total[j]
                if not (mu_best + r_S < mu_j - r_S):
                    separated = False
                    break

            if verbose:
                print(
                    f"Round {round_idx}: best height {cand_heights[best_loc]:.2f} m, "
                    f"mean cost {mu_best:.3f}, r={r_S:.3f}, separated={separated}"
                )

            if separated:
                return float(cand_heights[best_loc]), history

    # If max_rounds reached without separation
    mean_total_cp = cand_protections_cp + cum_damage_cp / float(num_samples)
    mean_total = cp.asnumpy(mean_total_cp)
    best_final = int(np.argmin(mean_total))
    return float(cand_heights[best_final]), history


__all__ = [
    "load_cost_curves",
    "simulate_annual_max_pp_batch_gpu",
    "run_bandit_gpu",
]
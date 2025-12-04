"""
optimal_levee_bandit
=====================

This module implements a risk–neutral levee height selection algorithm
based on the framework described in the accompanying LaTeX document
`mainfinal (1).tex`.  The goal is to choose an optimal levee design
height from a discrete set by combining deterministic pruning with a
confidence‑based pure exploration strategy.

The implementation proceeds in three stages:

1. **Data ingestion**:  Damage and protection cost curves are
   loaded from tab‑separated files.  Each row represents a city and
   contains flood heights (0 m, 0.5 m, … , 12 m) together with the
   corresponding damage or protection costs in million EUR.  The helper
   function :func:`load_cost_curves` reads a specified city's curves and
   returns the height grid, damage costs and protection costs.

2. **Pruning**:  Using only the cost curves and the length of the
   planning horizon `n`, the algorithm derives distribution‑free
   lower bounds on cost differences between adjacent heights.  Heights
   that are dominated irrespective of the flood distribution are
   removed.  See :func:`prune_candidates` for details.

3. **Pure exploration bandit**:  After pruning, the remaining heights
   are treated as arms in a pure exploration setting.  At each round
   a flood scenario is sampled and the damage is evaluated at all
   candidate heights simultaneously.  Uniform confidence intervals are
   maintained for the mean cost of each height using Hoeffding's
   inequality.  Sampling stops once the confidence interval of the
   empirically best height lies strictly below those of all others.
   The function :func:`run_bandit` orchestrates this procedure.

This file is self‑contained and does not depend on external tools.  All
data files are assumed to reside in the same directory as this script.

Example usage::

    from optimal_levee_bandit import run_bandit
    best_height, history = run_bandit(
        damage_file='data/Damage_cost_curves.tab',
        protection_file='data/Protection_cost_curves_high_estimate.tab',
        city='Halmstad',
        years_range=(2025, 2100),
        delta=0.05,
        max_rounds=10000,
    )
    print(f"Selected design height: {best_height:.2f} m")

The variable `history` contains intermediate estimates that can be used
for diagnostic plots or further analysis.

Note
----
The sampling of flood scenarios in :func:`_sample_predictive_maxima`
relies on the file ``varberg_sl_annual_2010_2200.npz``.  This file
contains 20 000 sample paths of mean sea level (MSL) in millimetres
over the years 2010–2100.  For demonstration purposes, we convert
these values to metres and treat negative values as zero.  In a more
realistic setting you would replace this function with a call to a
predictive model (e.g. the Poisson–GPD simulator provided in the
exercise description).

"""

from __future__ import annotations

import os
import math
from typing import Callable, Iterable, List, Sequence, Tuple, Dict, Optional

import numpy as np

def load_cost_curves(
    damage_file: str,
    protection_file: str,
    city: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load damage and protection cost curves for a given city.

    Parameters
    ----------
    damage_file : str
        Path to the damage cost curves ``.tab`` file.  Each line has the
        structure::

            ID<TAB>Name<TAB>Country<TAB>Latitude<TAB>Longitude<TAB>\
            Damage(0 m) <TAB> Damage(0.5 m) <TAB> ... <TAB> Damage(12 m)

    protection_file : str
        Path to the protection cost curves ``.tab`` file.  Each line has
        the same structure but with protection costs instead of damages.

    city : str
        Name of the city to select.  The comparison is case‑insensitive
        and matches the ``Name`` column in the files.  If multiple rows
        match, the first is taken.

    Returns
    -------
    heights : np.ndarray
        A one‑dimensional array of flood heights in metres at which costs
        are reported.  For the published curves this ranges from 0.0 to
        12.0 m in 0.5 m increments.

    damage_costs : np.ndarray
        An array of direct damage costs (in million EUR) corresponding to
        the heights in ``heights``.

    protection_costs : np.ndarray
        An array of levee construction costs (in million EUR) corresponding
        to the heights in ``heights``.

    Raises
    ------
    ValueError
        If the specified city is not found in either file or if the
        numbers of cost entries differ between the damage and protection
        curves.
    """
    def _parse_file(path: str) -> Dict[str, List[float]]:
        """Parse a cost curve file and return a mapping from city to costs.

        The keys are lower‑case city names; the values are lists of
        floating‑point numbers representing costs at successive heights.
        """
        costs: Dict[str, List[float]] = {}
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('/*'):
                    # Skip comments and empty lines
                    continue
                parts = line.split('\t')
                if len(parts) < 6:
                    # Unexpected format
                    continue
                name = parts[1].strip().lower()
                # The cost entries start from column 5 (index 5) onwards
                try:
                    numeric = [float(x) for x in parts[5:]]
                except ValueError:
                    # Skip lines with non‑numeric entries
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
    # Construct the height grid: assume equally spaced from 0 to maximum height
    num_levels = len(damage_costs)
    # The published data uses 0.5 m increments over 0 – 12 m inclusive: 25 points
    # Compute step size accordingly.
    max_height = 12.0
    if num_levels > 1:
        step = max_height / (num_levels - 1)
    else:
        step = 0.0
    heights = np.linspace(0.0, max_height, num_levels)
    return heights, damage_costs, protection_costs


def prune_candidates(
    heights: Sequence[float],
    protection_costs: Sequence[float],
    damage_costs: Sequence[float],
    n_years: int,
) -> List[int]:
    """Eliminate design heights that are deterministically suboptimal.

    The pruning algorithm follows Section 3.2 of the LaTeX document.
    Given the one‑step margins

        m_k^{(+)} = (C(h_{k+1}) - C(h_k)) - n * d(h_{k+1}),

    any height ``h_{k+1}`` with ``m_k^{(+)} > 0`` cannot be optimal for any
    flood distribution.  In addition, if the sum of margins along a
    multi‑step chain from ``h_j`` to ``h_k`` is positive, then ``h_k`` is
    dominated by ``h_j`` and may also be removed.

    Parameters
    ----------
    heights : Sequence[float]
        Array of design heights (metres) sorted in increasing order.

    protection_costs : Sequence[float]
        Present value of the construction/maintenance cost for each
        corresponding height in ``heights``.

    damage_costs : Sequence[float]
        Per‑flood damage cost corresponding to the flood height equal to
        each value in ``heights``.

    n_years : int
        Length of the planning horizon (number of flood scenarios to
        accumulate damage over).

    Returns
    -------
    List[int]
        Indices of heights that remain after pruning.  These heights
        require Monte Carlo exploration.
    """
    K = len(heights)
    # Compute the upward margins between adjacent heights
    m_plus = np.zeros(K - 1, dtype=float)
    for k in range(K - 1):
        delta_cost = protection_costs[k + 1] - protection_costs[k]
        m_plus[k] = delta_cost - n_years * damage_costs[k + 1]
    # Start with all heights active
    active = np.ones(K, dtype=bool)
    # Local one‑step elimination: if m_k^{(+)} > 0 remove h_{k+1}
    for k in range(K - 1):
        if m_plus[k] > 0.0:
            active[k + 1] = False
    # Multi‑step chain elimination.  Iterate until convergence.
    changed = True
    while changed:
        changed = False
        for j in range(K - 1):
            if not active[j]:
                continue
            sum_margin = 0.0
            for k in range(j, K - 1):
                # Only sum margins if both h_k and h_{k+1} are still active
                if active[k] and active[k + 1]:
                    sum_margin += m_plus[k]
                    if sum_margin > 0.0 and active[k + 1]:
                        active[k + 1] = False
                        changed = True
                # If the target height becomes inactive, break the chain
                if not active[k + 1]:
                    break
    return [i for i, flag in enumerate(active) if flag]

def prune_candidates_exact(
    heights, protection_costs, damage_costs, n_years
):
    K = len(heights)
    m_plus = np.zeros(K - 1)
    for k in range(K - 1):
        c_k = protection_costs[k + 1] - protection_costs[k]
        m_plus[k] = c_k - n_years * damage_costs[k + 1]
        print(m_plus[k])
    active = np.ones(K, dtype=bool)

    changed = True
    while changed:
        changed = False

        # 1) local one-step
        for k in range(K - 1):
            if active[k + 1] and m_plus[k] > 0:
                active[k + 1] = False
                changed = True

        # 2) multi-step: all pairs j<k
        for j in range(K - 1):
            if not active[j]:
                continue
            for k in range(j + 1, K):
                if not active[k]:
                    continue
                # sum_{r=j}^{k-1} m_r^{(+)} over ORIGINAL margins
                sum_margin = float(np.sum(m_plus[j:k]))
                if sum_margin > 0:
                    active[k] = False
                    changed = True

    return [i for i, flag in enumerate(active) if flag]

def _load_msl_data(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load mean sea level sample paths from a NumPy archive.

    The archive must contain keys ``'years'`` and ``'sl'``.  The years
    array should be one‑dimensional and sorted, and the ``sl`` array
    should have shape ``(n_paths, n_years)``.  Values in ``sl`` are
    interpreted as millimetres relative to a common datum and are
    converted to metres.

    Returns
    -------
    years : np.ndarray
        Array of years corresponding to the second dimension of ``sl``.

    sl_m : np.ndarray
        Array of sample paths converted to metres.  Negative values are
        clipped at zero as flood heights cannot be below the datum.
    """
    data = np.load(npz_path)
    years = data['years']
    sl_mm = data['sl'].astype(float)
    # Convert from millimetres to metres
    sl_m = sl_mm / 1000.0
    # Clip negative mean sea levels to zero
    sl_m[sl_m < 0.0] = 0.0
    return years, sl_m


def _sample_predictive_maxima(
    sl_m: np.ndarray,
    years: np.ndarray,
    year_range: Tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a random annual maximum flood scenario.

    For demonstration purposes we treat the MSL sample paths stored in
    ``varberg_sl_annual_2010_2200.npz`` as proxies for the annual maximum
    water levels.  In a real application this function should be
    replaced by a simulator that produces annual maxima conditional on
    mean sea level, such as the Poisson–GPD point‑process model provided
    elsewhere in the project.

    Parameters
    ----------
    sl_m : np.ndarray
        Two‑dimensional array of MSL sample paths in metres with shape
        ``(n_paths, n_years)``.

    years : np.ndarray
        One‑dimensional array of calendar years corresponding to the
        columns of ``sl_m``.

    year_range : tuple of (int, int)
        Inclusive range of years to extract.  For example, ``(2025,2100)``
        extracts 76 years (2025 through 2100 inclusive).

    rng : np.random.Generator
        Random number generator from NumPy.

    Returns
    -------
    maxima : np.ndarray
        One‑dimensional array of length equal to the number of years in
        ``year_range``.  Values are water levels in metres for each year.
    """
    start_year, end_year = year_range
    # Find the indices corresponding to the desired years
    mask = (years >= start_year) & (years <= end_year)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError(f"No years found in range {year_range}")
    # Select a random sample path
    path_index = rng.integers(low=0, high=sl_m.shape[0])
    sample_path = sl_m[path_index]
    return sample_path[idx]


def _interpolate_damage(
    exceedances: np.ndarray,
    height_grid: np.ndarray,
    damage_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate damage costs for arbitrary exceedance heights.

    Given a damage cost curve ``damage_grid`` defined on ``height_grid``,
    return the damage cost for each exceedance in ``exceedances``.  Values
    are linearly interpolated between grid points and clipped at the
    endpoints.

    Parameters
    ----------
    exceedances : np.ndarray
        Flood exceedance heights (metres) at which to evaluate the damage
        function.  Any negative values are treated as zero (i.e. no
        damage).

    height_grid : np.ndarray
        Monotonic grid of heights in metres corresponding to the
        ``damage_grid`` values.

    damage_grid : np.ndarray
        Damage costs (in million EUR) corresponding to ``height_grid``.

    Returns
    -------
    np.ndarray
        Damage costs for each exceedance height.
    """
    # Ensure non‑negative exceedances
    w = np.asarray(exceedances, dtype=float)
    w[w < 0.0] = 0.0
    # Use numpy interpolation; values outside the range are clipped
    return np.interp(w, height_grid, damage_grid)


import os
import math
from typing import Tuple, Dict, List, Optional

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
    Simulate one trajectory of annual maxima (in cm) for a given
    parameter draw and latent MSL path, using the Poisson–GPD PP model.

    Parameters
    ----------
    eta0, eta1 : float
        Poisson intensity regression parameters.
    alpha0 : float
        Log-scale parameter for the GPD: sigma = exp(alpha0).
    xi : float
        GPD shape parameter.
    X_t_series_cm : np.ndarray
        Latent MSL time series in centimeters for the target years.
    u_cm : float
        POT threshold in centimeters (e.g. 60.0).
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    maxima_cm : np.ndarray
        Annual maxima in centimeters, same length as X_t_series_cm.
    """
    sigma = np.exp(alpha0)
    T = X_t_series_cm.shape[0]
    maxima_cm = np.empty(T, dtype=float)

    for j in range(T):
        # Intensity for year j (λ_t = exp(η0 + η1 X_t))
        lam = np.exp(eta0 + eta1 * X_t_series_cm[j])
        N = rng.poisson(lam)

        if N <= 0:
            # No exceedances: max is below u; represent as exactly u
            maxima_cm[j] = u_cm
        else:
            U = rng.uniform(size=N)
            if abs(xi) < 1e-6:
                # Approximate GPD with ξ ≈ 0 by exponential
                Z = -sigma * np.log(1.0 - U)
            else:
                # Inverse CDF for GPD: z = σ/ξ * ((1 - U)^(-ξ) - 1)
                Z = sigma / xi * ((1.0 - U) ** (-xi) - 1.0)
            maxima_cm[j] = u_cm + Z.max()

    return maxima_cm

def run_bandit(
    damage_file: str,
    protection_file: str,
    city: str,
    years_range: Tuple[int, int] = (2025, 2100),
    delta: float = 0.05,
    max_rounds: int = 10000,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> Tuple[float, Dict[str, List[float]]]:
    """Run the confidence-based pure exploration algorithm for levee heights
    using the Poisson–GPD predictive posterior model.

    This function orchestrates the three stages of the procedure:

    1. Load cost curves for the specified city and construct the height grid.
    2. Apply deterministic pruning to eliminate obviously suboptimal heights.
    3. Perform Monte Carlo sampling using your Poisson–GPD point-process
       model to estimate the expected total cost of each remaining height.
       Sampling stops when the empirically best height is separated from
       all others by the confidence intervals.

    The predictive posterior (η0, η1, α0, ξ) and latent MSL paths are
    loaded from the NPZ file ``varberg_sl_annual_2010_2200.npz`` located
    in the same directory as the cost curve files. Adjust the key names
    below if your NPZ uses different ones.

    Parameters
    ----------
    damage_file : str
        Path to the damage cost curve data file.

    protection_file : str
        Path to the protection cost curve data file.

    city : str
        City name for which to perform the analysis.

    years_range : (int, int), optional
        Inclusive range of years over which to compute damages.  Defaults
        to (2025, 2100), which yields a horizon of 76 years.

    delta : float, optional
        Desired misselection probability.  The algorithm guarantees
        ``P(h_hat != h_star) <= delta`` asymptotically.  Defaults to 0.05.

    max_rounds : int, optional
        Upper bound on the number of Monte Carlo rounds to perform.  If
        the stopping condition has not been met by this point the
        procedure terminates early and returns the empirically best
        height.  Defaults to 10 000.

    rng : np.random.Generator, optional
        Random number generator.  If ``None`` a default generator is
        created using entropy from the OS.

    verbose : bool, optional
        If True, prints diagnostic information each round.

    Returns
    -------
    best_height : float
        The selected levee height in metres.

    history : Dict[str, List[float]]
        Dictionary containing sampling history.  The keys are
        ``'rounds'`` (round numbers), ``'best_index'`` (index of the
        empirically best height at each round), ``'best_height'``
        (corresponding height in metres) and ``'radii'`` (confidence
        radii).  These sequences have length equal to the number of
        rounds executed.
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- Stage 1: load cost curves ---
    heights, damage_costs, protection_costs = load_cost_curves(
        damage_file, protection_file, city
    )

    # Determine number of years in the planning horizon
    start_year, end_year = years_range
    n_years = end_year - start_year + 1

    # --- Stage 2: prune deterministically dominated heights ---
    candidate_indices = prune_candidates(
        heights=heights,
        protection_costs=protection_costs,
        damage_costs=damage_costs,
        n_years=n_years,
    )
    if not candidate_indices:
        raise RuntimeError("No candidate heights remain after pruning")
    print(
        f"Pruned from {len(heights)} to {len(candidate_indices)} candidate heights."
    )
    num_candidates = len(candidate_indices)

    # Maximum per-year damage cost (used for concentration bound)
    d_max = float(np.max(damage_costs))
    D_max_total = n_years * d_max

    # Precompute fixed protection costs and heights for candidates
    candidate_heights = heights[candidate_indices]
    candidate_protections = protection_costs[candidate_indices]

    # --- NEW: Load predictive posterior + latent MSL paths from NPZ ---
    npz_path = os.path.join(
        os.path.dirname(damage_file),
        "varberg_sl_annual_2010_2200.npz",
    )
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Cannot locate posterior/MSL data file at '{npz_path}'. "
            "It should reside in the same directory as the cost files."
        )

    data = np.load(npz_path)

    # Adjust these keys if your NPZ uses different names
    years_all = data["years"]                  # shape (n_years_total,)
    X_pred_paths_cm = data["X_pred_paths_cm"]  # shape (M_pred, n_years_total)
    eta0_s = data["eta0_s"].reshape(-1)
    eta1_s = data["eta1_s"].reshape(-1)
    alpha0_s = data["alpha0_s"].reshape(-1)
    xi_s = data["xi_s"].reshape(-1)

    n_total = eta0_s.size
    M_pred = X_pred_paths_cm.shape[0]

    # Indices of the target years
    mask = (years_all >= start_year) & (years_all <= end_year)
    year_idx = np.where(mask)[0]
    if year_idx.size == 0:
        raise ValueError(f"No years available in the posterior/MSL data covering {years_range}")

    # --- Bandit state ---
    cum_damage = np.zeros(num_candidates, dtype=float)
    num_samples = 0

    history: Dict[str, List[float]] = {
        "rounds": [],
        "best_index": [],
        "best_height": [],
        "radii": [],
    }

    # Interpolation grid for damage costs (heights in metres)
    height_grid = heights
    damage_grid = damage_costs

    # Threshold used in PP model (cm) – must match your PP fit
    u_cm = 60.0

    # --- Stage 3: Monte Carlo bandit loop ---
    for round_idx in range(1, max_rounds + 1):
        # Sample one posterior draw and one latent MSL path
        i = rng.integers(low=0, high=n_total)
        m = rng.integers(low=0, high=M_pred)

        eta0 = float(eta0_s[i])
        eta1 = float(eta1_s[i])
        alpha0 = float(alpha0_s[i])
        xi = float(xi_s[i])

        # Latent MSL path for desired years (cm)
        X_t_series_cm = X_pred_paths_cm[m, year_idx]

        # Simulate annual maxima in cm, then convert to metres
        maxima_cm = simulate_annual_max_pp(
            eta0=eta0,
            eta1=eta1,
            alpha0=alpha0,
            xi=xi,
            X_t_series_cm=X_t_series_cm,
            u_cm=u_cm,
            rng=rng,
        )
        maxima_m = maxima_cm / 100.0  # cm → m

        # Evaluate damage at each candidate height
        for idx_c, h in enumerate(candidate_heights):
            exceedances = maxima_m - h
            yearly_damage = _interpolate_damage(exceedances, height_grid, damage_grid)
            total_damage = float(np.sum(yearly_damage))
            cum_damage[idx_c] += total_damage

        num_samples += 1

        if round_idx % 10000 == 0:
                    # Empirical mean total cost for each candidate
            mean_total_cost = candidate_protections + cum_damage / num_samples

            # Confidence radius (Hoeffding + union bound, as in your LaTeX)
            log_term = math.log(
                (2.0 * num_candidates * round_idx * (round_idx + 1)) / max(delta, 1e-16)
            )
            r_S = D_max_total * math.sqrt(log_term / (2.0 * round_idx))

            # Empirically best height
            best_idx_local = int(np.argmin(mean_total_cost))

            # Record history (index in original height grid)
            history["rounds"].append(round_idx)
            history["best_index"].append(candidate_indices[best_idx_local])
            history["best_height"].append(float(candidate_heights[best_idx_local]))
            history["radii"].append(r_S)

            # Only check stopping condition every 1000 rounds
            separated = False  # default
            # Stopping condition: CI of best strictly below all others
            mu_best = mean_total_cost[best_idx_local]
            separated = True
            for j in range(num_candidates):
                if j == best_idx_local:
                    continue
                mu_j = mean_total_cost[j]
                if not (mu_best + r_S < mu_j - r_S):
                    separated = False
                    break

            if verbose:
                print(
                    f"Round {round_idx}: best height {candidate_heights[best_idx_local]:.2f} m, "
                    f"mean cost {mu_best:.3f}, r={r_S:.3f}, separated={separated}"
                )

            if separated:
                return float(candidate_heights[best_idx_local]), history

    # Fall back if max_rounds reached without separation
    best_final = int(np.argmin(mean_total_cost))
    return float(candidate_heights[best_final]), history

def run_bandit_from_candidates(
    heights: np.ndarray,
    damage_costs: np.ndarray,
    protection_costs: np.ndarray,
    candidate_indices: Sequence[int],
    years_all: np.ndarray,
    X_pred_paths_cm: np.ndarray,
    posterior_params: Dict[str, np.ndarray],
    years_range: Tuple[int, int] = (2025, 2100),
    delta: float = 0.05,
    max_rounds: int = 10000,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> Tuple[float, Dict[str, List[float]]]:
    """
    Bandit core: assumes you already did pruning and pass in candidate_indices,
    plus predictive posterior parameters and latent MSL paths.

    Parameters
    ----------
    heights, damage_costs, protection_costs : np.ndarray
        Full grids from `load_cost_curves`.

    candidate_indices : sequence of int
        Indices (into `heights`) that survived pruning.

    years_all : np.ndarray
        1D array of years corresponding to the columns of X_pred_paths_cm.

    X_pred_paths_cm : np.ndarray
        Latent MSL sample paths in centimetres, shape (M_pred, n_years_total).

    posterior_params : dict
        Dictionary with keys 'eta0', 'eta1', 'alpha0', 'xi' (and optionally 'u'),
        each a 1D array of flattened posterior draws.

    years_range, delta, max_rounds, rng, verbose :
        Same meaning as before.

    Returns
    -------
    best_height : float
        Selected levee height in metres.

    history : dict
        Keys: 'rounds', 'best_index', 'best_height', 'radii'.
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- Basic setup ---
    start_year, end_year = years_range
    n_years = end_year - start_year + 1

    candidate_indices = np.asarray(candidate_indices, dtype=int)
    if candidate_indices.size == 0:
        raise RuntimeError("No candidate heights provided to bandit")

    num_candidates = candidate_indices.size

    # Max per-year damage cost (for concentration bound)
    d_max = float(np.max(damage_costs))
    D_max_total = n_years * d_max

    # Candidate heights and protection costs
    candidate_heights = heights[candidate_indices]          # shape (C,)
    candidate_protections = protection_costs[candidate_indices]  # shape (C,)

    # --- Predictive posterior draws ---
    eta0_s   = np.asarray(posterior_params["eta0"]).reshape(-1)
    eta1_s   = np.asarray(posterior_params["eta1"]).reshape(-1)
    alpha0_s = np.asarray(posterior_params["alpha0"]).reshape(-1)
    xi_s     = np.asarray(posterior_params["xi"]).reshape(-1)
    n_total  = eta0_s.size

    # Optional: threshold from posterior_params if provided
    u_cm = float(posterior_params.get("u", 60.0))

    # --- MSL paths and year slicing ---
    X_pred_paths_cm = np.asarray(X_pred_paths_cm, dtype=float)  # (M_pred, T_all)
    M_pred = X_pred_paths_cm.shape[0]

    mask = (years_all >= start_year) & (years_all <= end_year)
    year_idx = np.where(mask)[0]
    if year_idx.size == 0:
        raise ValueError(
            f"No years available in the MSL data covering {years_range}"
        )

    # Slice once to avoid indexing inside the loop
    X_paths_slice_cm = X_pred_paths_cm[:, year_idx]  # shape (M_pred, n_years)

    # --- Bandit state ---
    cum_damage = np.zeros(num_candidates, dtype=float)
    num_samples = 0

    history: Dict[str, List[float]] = {
        "rounds": [],
        "best_index": [],
        "best_height": [],
        "radii": [],
    }

    height_grid = heights
    damage_grid = damage_costs

    # --- Monte Carlo bandit loop ---
    for round_idx in range(1, max_rounds + 1):
        # Sample one posterior draw and one MSL path
        i = rng.integers(low=0, high=n_total)
        m = rng.integers(low=0, high=M_pred)

        eta0   = float(eta0_s[i])
        eta1   = float(eta1_s[i])
        alpha0 = float(alpha0_s[i])
        xi     = float(xi_s[i])

        X_t_series_cm = X_paths_slice_cm[m]  # shape (n_years,)

        # Simulate annual maxima in cm, then convert to metres
        maxima_cm = simulate_annual_max_pp(
            eta0=eta0,
            eta1=eta1,
            alpha0=alpha0,
            xi=xi,
            X_t_series_cm=X_t_series_cm,
            u_cm=u_cm,
            rng=rng,
        )
        maxima_m = maxima_cm / 100.0  # cm → m, shape (n_years,)

        # --- Accumulate damage for all candidates (vectorised) ---
        # exceedances: shape (C, n_years)
        exceedances = np.where(
    maxima_m[None, :] >= candidate_heights[:, None],
    maxima_m[None, :],
    0
)
        exceedances = np.where(
    maxima_m[None, :] < 12,
    12,
    maxima_m[None, :]
)



        # Flatten, interpolate, then reshape
        flat_exceed = exceedances.ravel()
        flat_damage = _interpolate_damage(flat_exceed, height_grid, damage_grid)
        damage_matrix = flat_damage.reshape(num_candidates, -1)  # (C, n_years)

        cum_damage += damage_matrix.sum(axis=1)
        num_samples += 1

        # Only recompute CI / stopping every 10 000 rounds (change if you like)
        if round_idx % 1000 == 0:
            mean_total_cost = candidate_protections + cum_damage / num_samples

            # Confidence radius
            log_term = math.log(
                (2.0 * num_candidates * round_idx * (round_idx + 1)) / max(delta, 1e-16)
            )
            r_S = D_max_total * math.sqrt(log_term / (2.0 * round_idx))

            best_idx_local = int(np.argmin(mean_total_cost))

            history["rounds"].append(round_idx)
            history["best_index"].append(int(candidate_indices[best_idx_local]))
            history["best_height"].append(float(candidate_heights[best_idx_local]))
            history["radii"].append(r_S)

            # Stopping condition: CI of best strictly below all others
            separated = True
            mu_best = mean_total_cost[best_idx_local]
            for j in range(num_candidates):
                if j == best_idx_local:
                    continue
                mu_j = mean_total_cost[j]
                if not (mu_best + r_S < mu_j - r_S):
                    separated = False
                    break

            if verbose:
                print(
                    f"Round {round_idx}: best height {candidate_heights[best_idx_local]:.2f} m, "
                    f"mean cost {mu_best:.3f}, r={r_S:.3f}, separated={separated}"
                )

            if separated:
                return float(candidate_heights[best_idx_local]), history

    # Fall back if max_rounds reached
    mean_total_cost = candidate_protections + cum_damage / num_samples
    best_final = int(np.argmin(mean_total_cost))
    return float(candidate_heights[best_final]), history

def simulate_annual_max_pp_eff(
    eta0: float,
    eta1: float,
    alpha0: float,
    xi: float,
    X_t_series_cm: np.ndarray,
    u_cm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate one trajectory of annual maxima (in cm) using the Poisson–GPD PP model,
    but draw the maximum of N uniform exceedances via a Beta distribution for efficiency.

    Parameters
    ----------
    eta0, eta1, alpha0, xi : float
        Posterior parameter draws for intensity and GPD tail.
    X_t_series_cm : np.ndarray
        Latent MSL time series in centimetres for the target years.
    u_cm : float
        Threshold in centimetres.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    maxima_cm : np.ndarray
        Annual maxima in centimetres, same length as X_t_series_cm.
    """
    sigma = math.exp(alpha0)
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
                # Approximate GPD with xi ~ 0 by exponential
                Z = -sigma * np.log(1.0 - U)
            else:
                # Inverse CDF for GPD: z = sigma/xi * ((1 - U)^(-xi) - 1)
                Z = sigma / xi * ((1.0 - U) ** (-xi) - 1.0)
            maxima_cm[j] = u_cm + Z.max()

    return maxima_cm

def simulate_annual_max_pp_batch_eff(
    eta0_batch: np.ndarray,
    eta1_batch: np.ndarray,
    alpha0_batch: np.ndarray,
    xi_batch: np.ndarray,
    X_batch_cm: np.ndarray,
    u_cm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Vectorised Poisson–GPD annual-max simulation for a batch of scenarios.

    Parameters
    ----------
    eta0_batch, eta1_batch, alpha0_batch, xi_batch : np.ndarray, shape (B,)
        Posterior parameter draws for each of B scenarios.
    X_batch_cm : np.ndarray, shape (B, T)
        Latent MSL time series in cm for each scenario (B) and year (T).
    u_cm : float
        POT threshold in cm.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    maxima_cm : np.ndarray, shape (B, T)
        Annual maxima in cm for each scenario and year.
    """
    eta0_batch   = np.asarray(eta0_batch, dtype=float)
    eta1_batch   = np.asarray(eta1_batch, dtype=float)
    alpha0_batch = np.asarray(alpha0_batch, dtype=float)
    xi_batch     = np.asarray(xi_batch, dtype=float)
    X_batch_cm   = np.asarray(X_batch_cm, dtype=float)  # (B, T)

    B, T = X_batch_cm.shape

    # Broadcast sigma and xi to shape (B, T)
    sigma_bt = np.exp(alpha0_batch)[:, None]            # (B, 1)
    xi_bt    = xi_batch[:, None]                        # (B, 1)
    sigma_bt = np.broadcast_to(sigma_bt, (B, T))        # (B, T)
    xi_bt    = np.broadcast_to(xi_bt,    (B, T))        # (B, T)

    # Poisson intensities λ_{b,t} = exp(η0_b + η1_b * X_{b,t})
    # work on the log scale and clip to avoid overflow / huge λ
    log_lam_bt = eta0_batch[:, None] + eta1_batch[:, None] * X_batch_cm  # (B, T)

    # Maximum *plausible* expected number of storms per year.
    # Adjust this if you want a different physical cap.
    max_lam = 1e5
    log_lam_bt = np.clip(log_lam_bt, a_min=-np.inf, a_max=max_lam)

    lam_bt = np.exp(log_lam_bt)  # (B, T), now safely bounded

    # Draw N_{b,t} ~ Poisson(λ_{b,t})
    N_bt = rng.poisson(lam_bt)  # (B, T)


    # Initialise maxima at the threshold u_cm
    maxima_cm = np.full((B, T), u_cm, dtype=float)

    # Mask where there is at least one exceedance
    mask = N_bt > 0
    if not np.any(mask):
        return maxima_cm

    # Draw U_max ~ Beta(N, 1) via U^(1/N) with U ~ Uniform(0,1)
    U = rng.random(size=(B, T))
    U_max = np.ones_like(U)
    U_max[mask] = U[mask] ** (1.0 / N_bt[mask])

    # Compute z_max via GPD inverse for the maximum
    z_max = np.zeros((B, T), dtype=float)

    # xi ≈ 0 (exponential case)
    small_mask = (np.abs(xi_bt) < 1e-6) & mask
    if np.any(small_mask):
        z_max[small_mask] = -sigma_bt[small_mask] * np.log(1.0 - U_max[small_mask])

    # xi != 0 case
    large_mask = (~small_mask) & mask
    if np.any(large_mask):
        z_max[large_mask] = (sigma_bt[large_mask] / xi_bt[large_mask]) * (
            (1.0 - U_max[large_mask]) ** (-xi_bt[large_mask]) - 1.0
        )

    maxima_cm[mask] = u_cm + z_max[mask]
    return maxima_cm



def run_bandit_from_candidates_eff(
    heights: np.ndarray,
    damage_costs: np.ndarray,
    protection_costs: np.ndarray,
    candidate_indices: Sequence[int],
    years_all: np.ndarray,
    X_pred_paths_cm: np.ndarray,
    posterior_params: Dict[str, np.ndarray],
    years_range: Tuple[int, int] = (2025, 2100),
    delta: float = 0.05,
    max_rounds: int = 10000,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> Tuple[float, Dict[str, List[float]]]:
    """
    Efficient bandit algorithm using chunked simulation and vectorised damage computation.
    Assumes pruning has already been done (candidate_indices).
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- Basic setup ---
    start_year, end_year = years_range
    n_years = end_year - start_year + 1

    cand_idx = np.asarray(candidate_indices, dtype=int)
    if cand_idx.size == 0:
        raise RuntimeError("No candidate heights provided to bandit")

    num_cands = cand_idx.size

    # Max per-year damage cost (for Hoeffding bound)
    d_max = float(np.max(damage_costs))
    D_max_total = n_years * d_max

    # Candidate heights and protection costs
    cand_heights = heights[cand_idx]
    cand_protections = protection_costs[cand_idx]

    # Posterior draws
    eta0_s   = np.asarray(posterior_params["eta0"]).reshape(-1)
    eta1_s   = np.asarray(posterior_params["eta1"]).reshape(-1)
    alpha0_s = np.asarray(posterior_params["alpha0"]).reshape(-1)
    xi_s     = np.asarray(posterior_params["xi"]).reshape(-1)
    n_total  = eta0_s.size

    # Threshold in cm (use stored value if present)
    u_cm = float(posterior_params.get("u", 60.0))

    # Slice MSL paths for the target years once
    mask = (years_all >= start_year) & (years_all <= end_year)
    year_idx = np.where(mask)[0]
    if year_idx.size == 0:
        raise ValueError(f"No years in `years_all` covering {years_range}")

    X_paths_slice_cm = np.asarray(X_pred_paths_cm, dtype=float)[:, year_idx]  # (M_pred, n_years)
    M_pred = X_paths_slice_cm.shape[0]

    # Bandit state
    cum_damage = np.zeros(num_cands, dtype=float)
    num_samples = 0

    history: Dict[str, List[float]] = {
        "rounds": [],
        "best_index": [],
        "best_height": [],
        "radii": [],
    }

    height_grid = heights
    damage_grid = damage_costs

    # --- Chunked simulation parameters ---
    chunk_size = 100000       # number of scenarios per batch
    check_every = 100000     # how often to recompute CI / stopping rule

    round_idx = 0
    while round_idx < max_rounds:
        # How many rounds to simulate in this chunk?
        remaining = max_rounds - round_idx
        this_chunk = min(chunk_size, remaining)

        # Sample posterior draw and MSL path indices for all rounds in this chunk
        draw_indices = rng.integers(low=0, high=n_total, size=this_chunk)
        path_indices = rng.integers(low=0, high=M_pred, size=this_chunk)

        # Container for all simulated maxima (meters)
        # --- Vectorised simulation for this chunk ---
        # Parameters for all scenarios in the chunk
        eta0_batch   = eta0_s[draw_indices]
        eta1_batch   = eta1_s[draw_indices]
        alpha0_batch = alpha0_s[draw_indices]
        xi_batch     = xi_s[draw_indices]

        # Latent MSL series for all selected paths in the chunk: shape (this_chunk, n_years)
        X_batch_cm = X_paths_slice_cm[path_indices, :]

        # Vectorised Poisson–GPD simulation: shape (this_chunk, n_years), in cm
        maxima_batch_cm = simulate_annual_max_pp_batch_eff(
            eta0_batch=eta0_batch,
            eta1_batch=eta1_batch,
            alpha0_batch=alpha0_batch,
            xi_batch=xi_batch,
            X_batch_cm=X_batch_cm,
            u_cm=u_cm,
            rng=rng,
        )
        maxima_batch_m = maxima_batch_cm * 0.01  # convert cm → m


        # Vectorised damage for all scenarios and candidates:
        # exceedances shape: (this_chunk, C, n_years)
        exceedances = np.where(
        maxima_batch_m[:, None, :] >= height_grid[None, :, None],
        maxima_batch_m[:, None, :],
        0.0,
        )


        # Flatten, interpolate damage for all entries at once, and reshape
        flat = exceedances.reshape(-1)
        flat_damage = _interpolate_damage(flat, height_grid, damage_grid)
        damage_matrix = flat_damage.reshape(this_chunk, num_cands, n_years)

        # Sum damage over time and scenarios: accumulate into cum_damage
        cum_damage += damage_matrix.sum(axis=(0, 2))
        num_samples += this_chunk
        round_idx += this_chunk

        # Recompute CI and stopping every `check_every` rounds
        if round_idx % check_every == 0:
            mean_total = cand_protections + cum_damage / num_samples

            if verbose:
                print(f"\nRound {round_idx} – mean total cost per candidate height:")
                for h, c in zip(cand_heights, mean_total):
                    print(f"  height = {h:.2f} m, mean total cost = {c:.3f}")

            log_term = math.log(
                (2.0 * num_cands * round_idx * (round_idx + 1)) / max(delta, 1e-16)
            )
            r_S = D_max_total * math.sqrt(log_term / (2.0 * round_idx))

            best_loc = int(np.argmin(mean_total))

            history["rounds"].append(round_idx)
            history["best_index"].append(int(cand_idx[best_loc]))
            history["best_height"].append(float(cand_heights[best_loc]))
            history["radii"].append(r_S)

            # Stop when best CI is strictly below all others
            mu_best = mean_total[best_loc]
            separated = True
            for j in range(num_cands):
                if j == best_loc:
                    continue
                if not (mu_best + r_S < mean_total[j] - r_S):
                    separated = False
                    break

            if verbose:
                print(
                    f"Round {round_idx}: best height {cand_heights[best_loc]:.2f} m, "
                    f"mean cost {mu_best:.3f}, r={r_S:.3f}, separated={separated}"
                )

            if separated:
                return float(cand_heights[best_loc]), history

    # If max_rounds reached, return current best
    mean_total = cand_protections + cum_damage / num_samples
    best_final = int(np.argmin(mean_total))
    return float(cand_heights[best_final]), history

__all__ = [
    'load_cost_curves',
    'prune_candidates',
    'run_bandit',
]
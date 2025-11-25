import os
import numpy as np
from beta_dueling_bandit_analytical import (
    simulate_annual_max_pp,
    _interpolate_damage,
)
from optimal_levee_bandit import load_cost_curves

# ---------------------------
# 1. Paths and data loading
# ---------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))

damage_file     = os.path.join(script_dir, "Damage_cost_curves.tab")
protection_file = os.path.join(script_dir, "Protection_cost_curves_high_estimate.tab")
pp_file         = os.path.join(script_dir, "pp_inputs_halmsdad_pp_mixture_2025_2100.npz")

print("Using paths:")
print("  damage_file     =", damage_file)
print("  protection_file =", protection_file)
print("  pp_file         =", pp_file)

# City name (was missing in your snippet)
city = "Halmstad"

# Load cost curves
heights, damage_costs, protection_costs = load_cost_curves(
    damage_file,
    protection_file,
    city,
)

# Load posterior draws and mean sea level paths
pp = np.load(pp_file)
posterior_params = {
    "eta0":   pp["eta0"],
    "eta1":   pp["eta1"],
    "alpha0": pp["alpha0"],
    "xi":     pp["xi"],
    "u":      float(pp["u"]),  # threshold in cm
}
years_future   = pp["years_future"]   # 1D array of years
X_future_paths = pp["X_future_paths"] # shape (M_pred, T_future) in cm

# Analysis horizon
years_range = (2025, 2100)


# -----------------------------------------------
# 2. Monte Carlo estimator for a fixed height
# -----------------------------------------------

def estimate_mean_total_cost_for_height(
    target_height: float,
    heights: np.ndarray,
    damage_costs: np.ndarray,
    protection_costs: np.ndarray,
    years_all: np.ndarray,
    X_pred_paths_cm: np.ndarray,
    posterior_params: dict,
    years_range=(2025, 2100),
    n_replications: int = 5000,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Monte Carlo estimator of E[total_cost | height = target_height],
    using the same Poisson–GPD + MSL model as the dueling bandit.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Find index of the target height in the full grid
    i_h_matches = np.where(heights == target_height)[0]
    if i_h_matches.size == 0:
        raise ValueError(f"target_height {target_height} not found in heights grid.")
    i_h = int(i_h_matches[0])

    # Basic setup
    start_year, end_year = years_range
    n_years = end_year - start_year + 1

    years_all = np.asarray(years_all, dtype=int)
    mask = (years_all >= start_year) & (years_all <= end_year)
    year_idx = np.where(mask)[0]
    if year_idx.size != n_years:
        raise ValueError(
            f"years_all and years_range mismatch: expected {n_years} years, "
            f"found {year_idx.size} (check years_future and years_range)."
        )

    # Slice paths to the analysis years
    X_paths_slice_cm = np.asarray(X_pred_paths_cm, dtype=float)[:, year_idx]
    M_pred = X_paths_slice_cm.shape[0]

    # Flatten posterior parameter draws
    eta0_s   = np.asarray(posterior_params["eta0"]).reshape(-1)
    eta1_s   = np.asarray(posterior_params["eta1"]).reshape(-1)
    alpha0_s = np.asarray(posterior_params["alpha0"]).reshape(-1)
    xi_s     = np.asarray(posterior_params["xi"]).reshape(-1)
    n_total  = eta0_s.size

    u_cm = float(posterior_params.get("u", 60.0))

    # Precompute some constants
    h_m           = float(target_height)
    prot_cost     = float(protection_costs[i_h])
    height_grid_m = heights
    damage_grid   = damage_costs

    total_cost_sum = 0.0

    for r in range(n_replications):
        # 1. Draw parameter index and MSL path index
        i_draw = rng.integers(low=0, high=n_total)
        m_path = rng.integers(low=0, high=M_pred)

        eta0   = float(eta0_s[i_draw])
        eta1   = float(eta1_s[i_draw])
        alpha0 = float(alpha0_s[i_draw])
        xi     = float(xi_s[i_draw])

        X_t_series_cm = X_paths_slice_cm[m_path]  # (n_years,)

        # 2. Simulate annual maxima (cm) and convert to metres
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

        # 3. Compute total cost at this height
        exceedances   = np.where(maxima_m > h_m, maxima_m, 0.0)
        yearly_damage = _interpolate_damage(exceedances, height_grid_m, damage_grid)
        total_damage  = float(np.sum(yearly_damage))

        total_cost_sum += total_damage

        # Progress print every 10k reps
        if (r + 1) % 10_000 == 0:
            approx_mean = prot_cost + total_cost_sum / (r + 1)
            print(f"[Estimation] Replication {r+1}, running mean ≈ {approx_mean:.3f}")

    mean_cost = prot_cost + total_cost_sum / n_replications
    return mean_cost


# ---------------------------
# 3. Entry point
# ---------------------------

def main():
    best_height = 2.5
    print(f"Starting Monte Carlo estimation for height {best_height:.2f} m")

    rng = np.random.default_rng(2024)
    mean_best = estimate_mean_total_cost_for_height(
        target_height=best_height,
        heights=heights,
        damage_costs=damage_costs,
        protection_costs=protection_costs,
        years_all=years_future,
        X_pred_paths_cm=X_future_paths,
        posterior_params=posterior_params,
        years_range=years_range,
        n_replications=200_000,
        rng=rng,
    )

    print(f"[Estimation] Mean total cost for best height {best_height:.2f} m ≈ {mean_best:.3f}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print("ERROR OCCURRED:", repr(e))
        traceback.print_exc()
        input("Press Enter to exit...")

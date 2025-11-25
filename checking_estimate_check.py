import os
import numpy as np
import psutil
from beta_dueling_bandit_analytical import (
    _interpolate_damage,
)
from optimal_levee_bandit import load_cost_curves

# ---------------------------
# 0. Process handle for memory diagnostics
# ---------------------------

process = psutil.Process(os.getpid())


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

# --- Log shapes and approximate sizes of big arrays ---
print("\n[Diagnostics] Array shapes and approximate sizes")
print("  heights.shape:", heights.shape)
print("  damage_costs.shape:", damage_costs.shape)
print("  protection_costs.shape:", protection_costs.shape)
print("  years_future.shape:", years_future.shape)
print("  X_future_paths.shape:", X_future_paths.shape)
print("  X_future_paths approx size (MB):",
      X_future_paths.size * 8 / 1e6)

for name, arr in posterior_params.items():
    a = np.asarray(arr)
    print(f"  {name}.shape: {a.shape}, approx size (MB): {a.size * 8 / 1e6:.2f}")

print("[Diagnostics] Initial RSS (MB):", process.memory_info().rss / 1e6)


# -----------------------------------------------
# 2. Capped version of simulate_annual_max_pp
# -----------------------------------------------

def simulate_annual_max_pp_capped(
    eta0: float,
    eta1: float,
    alpha0: float,
    xi: float,
    X_t_series_cm: np.ndarray,
    u_cm: float,
    rng: np.random.Generator,
    lam_cap: float = 10_000.0,
    N_max: int = 100_000,
) -> np.ndarray:
    """
    Safe wrapper for the Poisson–GPD point process simulator.

    - Caps the intensity: lam = min(exp(eta0 + eta1 * X_t), lam_cap)
    - Caps the number of exceedances per year: N <= N_max

    This prevents occasional insane Poisson draws from making a single
    simulation run take forever.
    """
    X_t_series_cm = np.asarray(X_t_series_cm, dtype=float)
    T = X_t_series_cm.size
    maxima_cm = np.empty(T, dtype=float)

    # Convert parameters to floats
    eta0   = float(eta0)
    eta1   = float(eta1)
    alpha0 = float(alpha0)
    xi     = float(xi)
    u_cm   = float(u_cm)

    for j in range(T):
        lam_raw = np.exp(eta0 + eta1 * X_t_series_cm[j])
        lam = min(lam_raw, lam_cap)

        N = rng.poisson(lam)
        if N > N_max:
            # Optional: log occasionally
            # print(f"[simulate_capped] Clipping N from {N} to {N_max} (lam_raw={lam_raw:.2e}, lam={lam:.2e})")
            N = N_max

        if N <= 0:
            maxima_cm[j] = u_cm
        else:
            # Standard GPD exceedance simulation
            U = rng.uniform(size=N)
            # sigma_t = exp(alpha0 + alpha1 * X_t); here alpha1=0 in your model
            sigma = np.exp(alpha0)
            if abs(xi) < 1e-8:
                Z = -sigma * np.log(1.0 - U)
            else:
                Z = sigma / xi * ((1.0 - U) ** (-xi) - 1.0)
            maxima_cm[j] = u_cm + Z.max()

    return maxima_cm


# -----------------------------------------------
# 3. Monte Carlo estimator for a fixed height
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
    using the *capped* Poisson–GPD + MSL model.
    Prints memory diagnostics every 10k replications.
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

    print("\n[Diagnostics] X_paths_slice_cm.shape:", X_paths_slice_cm.shape)
    print("[Diagnostics] X_paths_slice_cm approx size (MB):",
          X_paths_slice_cm.size * 8 / 1e6)

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
        maxima_cm = simulate_annual_max_pp_capped(
            eta0=eta0,
            eta1=eta1,
            alpha0=alpha0,
            xi=xi,
            X_t_series_cm=X_t_series_cm,
            u_cm=u_cm,
            rng=rng,
            lam_cap=10_000_000.0,
            N_max=100_000,
        )
        maxima_m = maxima_cm * 0.01  # cm -> m

        # 3. Compute total cost at this height
        exceedances   = np.where(maxima_m > h_m, maxima_m, 0.0)
        yearly_damage = _interpolate_damage(exceedances, height_grid_m, damage_grid)
        total_damage  = float(np.sum(yearly_damage))

        total_cost_sum += total_damage

        # Progress + memory diagnostics every 10k reps
        if (r + 1) % 10_000 == 0 or (r + 1) == n_replications:
            approx_mean = prot_cost + total_cost_sum / (r + 1)
            rss_mb = process.memory_info().rss / 1e6
            print(
                f"[Estimation] Rep {r+1}/{n_replications}, "
                f"running mean ≈ {approx_mean:.3f}, "
                f"RSS ≈ {rss_mb:.1f} MB"
            )

    mean_cost = prot_cost + total_cost_sum / n_replications
    return mean_cost


# ---------------------------
# 4. Entry point
# ---------------------------

def main():
    best_height = 2.5
    print(f"\nStarting Monte Carlo estimation for height {best_height:.2f} m")

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
        n_replications=1_000_000,
        rng=rng,
    )

    print(f"\n[Estimation] Mean total cost for best height {best_height:.2f} m ≈ {mean_best:.3f}")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print("ERROR OCCURRED:", repr(e))
        traceback.print_exc()
        input("Press Enter to exit...")

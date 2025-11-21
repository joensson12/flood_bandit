import numpy as np

class LeveeHeightSelector:
    """
    Implements the Confidence-Based Pure Exploration algorithm for risk-neutral
    levee height selection as described in Section 4 of the paper.
    """
    def __init__(self, candidates, protection_costs, d_max, delta):
        """
        Args:
            candidates (list): List of candidate height identifiers (e.g., indices or floats).
                               Corresponds to H_eff.
            protection_costs (np.array): Deterministic costs C_n(h) for each candidate.
            d_max (float): The known upper bound of damage D_max.
            delta (float): The error tolerance parameter (probability of error).
        """
        self.candidates = candidates
        self.num_candidates = len(candidates) # |H_eff|
        self.protection_costs = np.array(protection_costs)
        self.d_max = d_max
        self.delta = delta
        
        # Tracking simulation state
        self.s = 0  # Sample size S
        self.sum_damages = np.zeros(self.num_candidates) # Sum of Y_{s,h}
        self.empirical_means = np.zeros(self.num_candidates) # mu_hat(S)

    def calculate_radius(self, s):
        """
        Calculates the confidence radius r(S) based on Equation (11).
        
        Formula: r(S) = D_max * sqrt( (1/2S) * log( (2 * |H| * S * (S+1)) / delta ) )
        """
        # Avoid division by zero or log(0) if called with s=0
        if s < 1:
            return float('inf')

        numerator = 2 * self.num_candidates * s * (s + 1)
        log_term = np.log(numerator / self.delta)
        
        radius = self.d_max * np.sqrt((1 / (2 * s)) * log_term)
        return radius

    def update(self, damages):
        """
        Updates the internal state with a new vector of damages from one flood scenario.
        
        Args:
            damages (np.array): Vector of damages D(F_s | h) for all candidates.
                                Must have length equal to num_candidates.
        """
        self.s += 1
        self.sum_damages += damages
        
        # Calculate mu_hat(S) = C_n(h) + (1/S) * sum(Damages)
        self.empirical_means = self.protection_costs + (self.sum_damages / self.s)

    def check_stopping_condition(self):
        """
        Checks if the stopping rule from Equation (13) is met.
        
        Returns:
            tuple: (bool, int/None) -> (should_stop, best_candidate_index)
        """
        r_s = self.calculate_radius(self.s)
        
        # Identify the empirical best arm: h_hat(S)
        best_idx = np.argmin(self.empirical_means)
        mu_best = self.empirical_means[best_idx]
        
        # Check condition: mu_best + r(S) < mu_h - r(S) for all h != best
        # Equivalently: mu_best + 2*r(S) < mu_h
        
        # Get the minimum mean among all competitors (exclude the best)
        # We can do this efficiently by masking
        mask = np.ones(self.num_candidates, dtype=bool)
        mask[best_idx] = False
        min_competitor_mu = np.min(self.empirical_means[mask])
        
        # The Stopping Condition (Eq 13)
        if (mu_best + r_s) < (min_competitor_mu - r_s):
            return True, best_idx
            
        return False, None

# -------------------------------------------------------
# Simulation Driver with Synthetic Data
# -------------------------------------------------------

def run_simulation():
    # 1. Setup Synthetic Problem
    # Let's assume 5 candidate heights. 
    # Height 2 (index 2) will be the true optimal.
    candidates = [0.0, 1.0, 2.0, 3.0, 4.0] 
    
    # Deterministic Protection Costs (increasing with height)
    protection_costs = np.array([10, 20, 35, 60, 90])
    
    # Load real damage curve data
    data = np.load('varberg_sl_annual_2010_2200.npz')
    sl = data['sl']  # shape (N_years, N_heights)

    # Use all columns as candidates (or select a subset if desired)
    num_candidates = sl.shape[1]
    candidates = list(range(num_candidates))

    # Protection costs: must be provided for each candidate
    # For demonstration, use a simple increasing cost (replace with real data if available)
    protection_costs = np.linspace(10, 90, num_candidates)

    # Parameters
    d_max = np.max(sl)  # Known upper bound on damage
    delta = 0.05   # 95% confidence

    # Initialize Selector
    selector = LeveeHeightSelector(candidates, protection_costs, d_max, delta)

    print(f"Starting simulation with {len(candidates)} candidates.")
    print(f"Target Delta: {delta}, D_max: {d_max}")
    print("-" * 50)

    # Monte Carlo Loop
    max_iterations = 10000000
    np.random.seed(42)

    for s in range(1, max_iterations + 1):
        # Sample a random year/scenario from the damage curve
        damages = sl[np.random.randint(sl.shape[0])]
        damages = np.clip(damages, 0, d_max)  # Enforce boundedness assumption

        # Update Algorithm
        selector.update(damages)

        # Check Stopping Rule
        should_stop, best_idx = selector.check_stopping_condition()

        if should_stop:
            print(f"\nStopping condition met at sample size S = {s}")
            print(f"Selected Height Index: {best_idx}")
            print(f"Selected Height Value: {candidates[best_idx]}")
            print(f"Estimated Means: {np.round(selector.empirical_means, 2)}")
            print(f"Final Confidence Radius r(S): {selector.calculate_radius(s):.4f}")
            break

        if s % 5000 == 0:
            # Print status occasionally
            r = selector.calculate_radius(s)
            best = np.argmin(selector.empirical_means)
            print(f"Step {s}: Best Idx {best}, r(S) = {r:.2f}")

if __name__ == "__main__":
    run_simulation()
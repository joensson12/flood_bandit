import numpy as np
import pandas as pd
from LeveeHeightSelector import LeveeHeightSelector

def load_varberg_data():
    """
    Loads Varberg damage and protection cost curves, and sea level scenarios.
    """
    # 1. Load Cost Curves
    # The files are tab-separated. We know Varberg is on line 378 (1-based) or we can search by ID 340.
    # We'll read the whole file and filter.
    
    # Column names based on the header inspection
    # ID, Name, Country, Lat, Lon, Cost_0.0, Cost_0.5, ... Cost_12.0
    flood_heights = np.arange(0.0, 12.5, 0.5) # 0.0 to 12.0
    
    # Helper to parse the tab files
    def parse_curve_file(filepath, city_id="340"):
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Find the line with the city_id
        city_line = None
        for line in lines:
            if line.startswith(city_id + "\t"):
                city_line = line
                break
        
        if city_line is None:
            raise ValueError(f"City ID {city_id} not found in {filepath}")
            
        parts = city_line.strip().split('\t')
        # The costs start from the 6th column (index 5)
        # ID(0), Name(1), Country(2), Lat(3), Lon(4), Cost_0.0(5)...
        costs = [float(x) for x in parts[5:]]
        return np.array(costs)

    damage_curve = parse_curve_file("Damage_cost_curves.tab")
    protection_curve = parse_curve_file("Protection_cost_curves_low_estimate.tab")
    
    # 2. Load Sea Level Scenarios
    data = np.load('varberg_sl_annual_2010_2200.npz')
    sl_scenarios = data['sl'] # Shape (20000, 91)
    # Convert cm to meters
    sl_scenarios_m = sl_scenarios / 100.0
    
    return flood_heights, damage_curve, protection_curve, sl_scenarios_m

def calculate_damage(flood_heights, damage_curve, water_levels, levee_height):
    """
    Calculates total damage for a sequence of water levels given a levee height.
    
    Args:
        flood_heights (np.array): The x-values of the damage curve (0.0, 0.5, ...).
        damage_curve (np.array): The y-values (damage in M€).
        water_levels (np.array): A sequence of water levels (meters) for one scenario.
        levee_height (float): The design height of the levee.
        
    Returns:
        float: Total damage over the scenario.
    """
    # Damage occurs only if water_level > levee_height
    # We interpolate the damage curve at the water levels
    
    # Filter for floods that exceed the levee
    exceedances = water_levels[water_levels > levee_height]
    
    if len(exceedances) == 0:
        return 0.0
        
    # Interpolate damage
    # We assume damage is 0 below 0m (which is handled by the curve usually starting at 0 or low)
    # and we extrapolate or clip if above 12m (though 12m is very high)
    damages = np.interp(exceedances, flood_heights, damage_curve, left=0.0, right=damage_curve[-1])
    
    return np.sum(damages)

def run_varberg_simulation():
    print("Loading Varberg data...")
    flood_heights, damage_curve, protection_curve, sl_scenarios = load_varberg_data()
    
    print(f"Loaded {len(sl_scenarios)} scenarios.")
    print(f"Damage curve max: {np.max(damage_curve)} M€")
    print(f"Protection curve max: {np.max(protection_curve)} M€")
    
    # Setup Bandit Problem
    candidates = flood_heights # We can choose from the discrete heights 0.0, 0.5, ... 12.0
    protection_costs = protection_curve
    
    # Estimate D_max
    # Max possible damage per year is roughly max(damage_curve).
    # Over 91 years, D_max could be 91 * max(damage_curve).
    # This is a very loose bound. We can refine it.
    # Let's look at the max water level in the data.
    max_sl = np.max(sl_scenarios)
    print(f"Max Sea Level in data: {max_sl:.2f} m")
    
    # Calculate max possible damage for a single event
    max_event_damage = np.interp(max_sl, flood_heights, damage_curve, right=damage_curve[-1])
    print(f"Max single event damage: {max_event_damage:.2f} M€")
    
    # D_max for the horizon (sum of damages)
    # In the worst case, every year floods (unlikely but possible bound).
    # A tighter bound might be needed for efficiency, but let's be safe.
    # Let's assume a max of 10 catastrophic floods in 91 years for the bound? 
    # Or just use the theoretical max: 91 * max_event_damage.
    d_max = 91 * max_event_damage
    print(f"Using D_max (horizon): {d_max:.2f} M€")
    
    delta = 0.05
    
    selector = LeveeHeightSelector(candidates, protection_costs, d_max, delta)
    
    print("-" * 50)
    print("Starting Pure Exploration...")
    
    # Shuffle scenarios to simulate random sampling
    num_scenarios = len(sl_scenarios)
    indices = np.arange(num_scenarios)
    np.random.shuffle(indices)
    
    for i, idx in enumerate(indices):
        scenario = sl_scenarios[idx] # A sequence of 91 annual max water levels
        
        # Calculate damage for ALL candidates for this SAME scenario (CRN)
        damages = []
        for h in candidates:
            d = calculate_damage(flood_heights, damage_curve, scenario, h)
            damages.append(d)
        
        damages = np.array(damages)
        
        # Update Bandit
        selector.update(damages)
        
        # Check Stopping
        should_stop, best_idx = selector.check_stopping_condition()
        
        if should_stop:
            print(f"\nStopping condition met at sample size S = {selector.s}")
            print(f"Selected Height: {candidates[best_idx]} m")
            print(f"Estimated Total Cost: {selector.empirical_means[best_idx]:.2f} M€")
            print(f"Confidence Radius: {selector.calculate_radius(selector.s):.2f}")
            break
            
        if (i + 1) % 100 == 0:
             best = np.argmin(selector.empirical_means)
             r = selector.calculate_radius(selector.s)
             print(f"Step {i+1}: Best Height {candidates[best]} m, Cost {selector.empirical_means[best]:.2f}, r(S)={r:.2f}")

    if not should_stop:
        print("\nExhausted all available scenarios without meeting stopping condition.")
        best_idx = np.argmin(selector.empirical_means)
        print(f"Best guess: {candidates[best_idx]} m")

if __name__ == "__main__":
    run_varberg_simulation()

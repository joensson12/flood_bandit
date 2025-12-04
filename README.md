# Flood Bandit Project

This project implements a risk-neutral levee height selection algorithm using confidence-based pure exploration and dueling bandits.

## Project Structure

### `src/` - Source Code
Contains the core logic and reusable modules.
- `bandit_cpu.py`: CPU-based implementation of the bandit algorithm (formerly `optimal_levee_bandit.py`).
- `bandit_gpu.py`: GPU-accelerated implementation (formerly `optimal_levee_bandit_gpu.py`).
- `bandit_dueling.py`: Beta-Dueling Bandit implementation (formerly `beta_dueling_bandit_analytical.py`).

### `notebooks/` - Experiments & Analysis
Jupyter notebooks for running experiments and analyzing data.
- `01_hazard_modeling.ipynb`: AR6 data fetching and latent covariate processing (formerly `fitting_simulator.ipynb`).
- `02_data_prep.ipynb`: Loading and parsing cost curves (formerly `reading_data.ipynb`).
- `03_main_experiment_gpu.ipynb`: Main GPU-accelerated experiment (formerly `run_gpu_bandit.ipynb`).
- `04_dueling_experiment.ipynb`: Dueling Bandits comparative analysis (formerly `duelingclean.ipynb`).

### `data/` - Datasets
Contains input data files (cost curves, sea level projections).

### `archive/` - Prototypes & Scripts
Old prototypes and helper scripts.
- `prototype_cpu_bandit.ipynb`: Superseded CPU prototype.
- `prototype_grid_pruning.ipynb`: Development scratchpad for pruning.
- `script_check_estimate.py`: Verification scripts.

### `papers/`
LaTeX source and PDFs for the project paper/presentation.

## Usage

To run the main experiment:
1. Ensure you have the required dependencies installed.
2. Navigate to `notebooks/`.
3. Open `03_main_experiment_gpu.ipynb`.

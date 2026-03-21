# Microbial Fuel Cell Genetic Algorithm Simulation

This repository contains a Python Tkinter program that:

1. creates an experimental-style microbial fuel cell dataset,
2. fits a higher-order polynomial surrogate model,
3. runs a genetic algorithm over glucose concentration and temperature, and
4. animates the optimisation in Tkinter with a surrogate heatmap and convergence view.

## Features

- **Larger synthetic dataset** with repeated trials and small random noise.
- **Higher-order surrogate model** including cubic and interaction terms.
- **Genetic algorithm kept intact** with selection, crossover, mutation, and elitism.
- **Tkinter simulation retained** so you can watch the GA move across the fitted search surface.
- **Clear visual feedback** showing the surrogate heatmap, current population, best solution, and convergence curve.
- **Comments that explain the modelling idea**: the GA optimises the fitted surrogate, not the raw data points.

## Requirements

- Python 3.11+
- Tkinter support in your Python installation

## Run

```bash
python3 main.py
```

If you run the program in a headless environment without a display server, the optimisation will still execute in the terminal, but the Tkinter window will not open.

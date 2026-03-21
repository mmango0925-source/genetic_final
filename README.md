# Microbial Fuel Cell Genetic Algorithm Simulation

This repository contains a Python Tkinter program that:

1. creates a small microbial fuel cell dataset,
2. fits a quadratic surrogate model,
3. runs a genetic algorithm over glucose concentration and temperature, and
4. animates every generation so users can watch each entity move through the search space.

## Features

- **Cleaned-up codebase** based on the provided script.
- **Generation-by-generation animation** in Tkinter.
- **Circle colours mapped to glucose concentration**.
- **Best entity highlighted** in every generation.
- **Play, pause, next-generation, and restart controls**.

## Requirements

- Python 3.11+
- Tkinter support in your Python installation

## Run

```bash
python3 main.py
```

If you run the program in a headless environment without a display server, the optimisation will still execute in the terminal, but the Tkinter window will not open.

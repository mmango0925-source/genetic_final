import math
import random
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import NamedTuple


# =============================================================================
# Microbial fuel cell optimisation project
# -----------------------------------------------------------------------------
# Workflow:
#   1. Create an experimental-style dataset.
#   2. Fit a polynomial regression model to the dataset.
#   3. Treat that regression equation as a surrogate model.
#   4. Use a genetic algorithm (GA) to optimise the surrogate model.
#   5. Animate the GA in Tkinter so the search can be watched generation by
#      generation.
#
# Important idea:
#   The GA is NOT applied directly to the raw data points.
#   Instead, the data is used to build a surrogate model, and the GA searches
#   that model for the best glucose concentration and temperature.
# =============================================================================

MIN_GLUCOSE = 0.00
MAX_GLUCOSE = 0.20
MIN_TEMPERATURE = 20.0
MAX_TEMPERATURE = 40.0

POPULATION_SIZE = 60
GENERATIONS = 100
MUTATION_RATE = 0.20
CROSSOVER_RATE = 0.85
TOURNAMENT_SIZE = 3
ELITE_COUNT = 2
GLUCOSE_MUTATION_STEP = 0.010
TEMPERATURE_MUTATION_STEP = 1.50
RANDOM_SEED = 7
ANIMATION_DELAY_MS = 300

GLUCOSE_CENTRE = 0.10
GLUCOSE_SCALE = 0.08
TEMPERATURE_CENTRE = 30.0
TEMPERATURE_SCALE = 8.0


class Observation(NamedTuple):
    """One row of experimental-style data."""

    glucose_concentration: float
    temperature: float
    voltage: float


class SurrogateModel(NamedTuple):
    """Stores the fitted polynomial model and simple fit statistics."""

    coefficients: list[float]
    feature_names: list[str]
    r_squared: float
    rmse: float


@dataclass(frozen=True)
class GenerationSnapshot:
    """Stores one generation of GA results for reporting and animation."""

    generation_index: int
    population: list[tuple[float, float]]
    best_individual: tuple[float, float]
    best_voltage: float
    average_voltage: float


# -----------------------------------------------------------------------------
# 1. Dataset and surrogate model.
# -----------------------------------------------------------------------------


def true_biological_response(glucose: float, temperature: float) -> float:
    """Create a realistic hidden response used only to generate sample data."""
    main_peak = 0.95 * math.exp(-((glucose - 0.11) / 0.045) ** 2 - ((temperature - 31.5) / 4.8) ** 2)
    secondary_peak = 0.16 * math.exp(-((glucose - 0.17) / 0.022) ** 2 - ((temperature - 27.0) / 3.8) ** 2)
    stress_penalty = 0.16 * ((glucose - 0.11) / 0.09) ** 4 + 0.14 * ((temperature - 31.0) / 7.5) ** 4
    metabolic_wobble = 0.03 * math.sin(26.0 * glucose + 0.35 * temperature)

    voltage = 0.24 + main_peak + secondary_peak + metabolic_wobble - stress_penalty
    return max(0.05, voltage)



def create_sample_dataset() -> list[Observation]:
    """Create a larger dataset with repeated trials and experimental noise."""
    glucose_levels = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
    temperature_levels = [22.0, 25.0, 28.0, 31.0, 34.0, 37.0, 40.0]
    trial_count = 3
    rng = random.Random(RANDOM_SEED)

    dataset: list[Observation] = []
    for glucose in glucose_levels:
        for temperature in temperature_levels:
            baseline_voltage = true_biological_response(glucose, temperature)
            for _ in range(trial_count):
                experimental_noise = rng.uniform(-0.028, 0.028)
                measured_voltage = max(0.05, baseline_voltage + experimental_noise)
                dataset.append(Observation(glucose, temperature, measured_voltage))

    return dataset



def build_feature_row(glucose: float, temperature: float) -> list[float]:
    """Create one regression feature row for the surrogate model.

    The inputs are centred and scaled first so the least-squares calculation is
    more stable. The resulting surrogate is still a cubic polynomial with the
    higher-order and interaction terms requested in the task.
    """
    g = (glucose - GLUCOSE_CENTRE) / GLUCOSE_SCALE
    t = (temperature - TEMPERATURE_CENTRE) / TEMPERATURE_SCALE

    return [
        1.0,
        g,
        t,
        g * g,
        t * t,
        g * t,
        g ** 3,
        t ** 3,
        (g ** 2) * t,
        g * (t ** 2),
    ]



def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]

    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row: abs(augmented[row][pivot_index]))
        augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        pivot_value = augmented[pivot_index][pivot_index]
        if abs(pivot_value) < 1e-12:
            raise ValueError("The regression system is singular and cannot be solved.")

        for column in range(pivot_index, size + 1):
            augmented[pivot_index][column] /= pivot_value

        for row in range(size):
            if row == pivot_index:
                continue
            factor = augmented[row][pivot_index]
            for column in range(pivot_index, size + 1):
                augmented[row][column] -= factor * augmented[pivot_index][column]

    return [augmented[row][-1] for row in range(size)]



def fit_surrogate_model(dataset: list[Observation]) -> SurrogateModel:
    """Fit the polynomial surrogate model using least squares normal equations."""
    rows = [build_feature_row(item.glucose_concentration, item.temperature) for item in dataset]
    targets = [item.voltage for item in dataset]
    feature_count = len(rows[0])

    xtx = [[0.0 for _ in range(feature_count)] for _ in range(feature_count)]
    xty = [0.0 for _ in range(feature_count)]

    for row, target in zip(rows, targets):
        for left in range(feature_count):
            xty[left] += row[left] * target
            for right in range(feature_count):
                xtx[left][right] += row[left] * row[right]

    coefficients = solve_linear_system(xtx, xty)
    predictions = [sum(value * coefficient for value, coefficient in zip(row, coefficients)) for row in rows]

    mean_voltage = sum(targets) / len(targets)
    residuals = [target - prediction for target, prediction in zip(targets, predictions)]
    ss_res = sum(residual * residual for residual in residuals)
    ss_tot = sum((target - mean_voltage) ** 2 for target in targets)
    r_squared = 1.0 - ss_res / ss_tot
    rmse = math.sqrt(ss_res / len(targets))

    return SurrogateModel(
        coefficients=coefficients,
        feature_names=[
            "Intercept",
            "g (scaled glucose)",
            "t (scaled temp)",
            "g^2",
            "t^2",
            "g*t",
            "g^3",
            "t^3",
            "g^2*t",
            "g*t^2",
        ],
        r_squared=r_squared,
        rmse=rmse,
    )



def predict_voltage(glucose: float, temperature: float, model: SurrogateModel) -> float:
    """Predict voltage from glucose and temperature using the surrogate model."""
    features = build_feature_row(glucose, temperature)
    return sum(value * coefficient for value, coefficient in zip(features, model.coefficients))



def print_model_summary(model: SurrogateModel, dataset: list[Observation]) -> None:
    """Print a concise summary of the fitted surrogate model."""
    print("Fitted surrogate model")
    print("=" * 78)
    print("Using centred variables: g = (G - 0.10) / 0.08 and t = (T - 30.0) / 8.0")
    print(
        "V = b0 + b1*g + b2*t + b3*g^2 + b4*t^2 + b5*g*t + "
        "b6*g^3 + b7*t^3 + b8*g^2*t + b9*g*t^2"
    )
    print()
    for name, coefficient in zip(model.feature_names, model.coefficients):
        print(f"{name:<24} = {coefficient: .6f}")
    print()
    print(f"Dataset rows : {len(dataset)}")
    print(f"Model R^2    : {model.r_squared:.4f}")
    print(f"Model RMSE   : {model.rmse:.4f} V")
    print("=" * 78)


# -----------------------------------------------------------------------------
# 2. Genetic algorithm.
# -----------------------------------------------------------------------------


def clamp_glucose(value: float) -> float:
    return max(MIN_GLUCOSE, min(MAX_GLUCOSE, value))



def clamp_temperature(value: float) -> float:
    return max(MIN_TEMPERATURE, min(MAX_TEMPERATURE, value))



def create_individual() -> tuple[float, float]:
    return (
        random.uniform(MIN_GLUCOSE, MAX_GLUCOSE),
        random.uniform(MIN_TEMPERATURE, MAX_TEMPERATURE),
    )



def create_population(size: int) -> list[tuple[float, float]]:
    return [create_individual() for _ in range(size)]



def fitness(individual: tuple[float, float], model: SurrogateModel) -> float:
    return predict_voltage(individual[0], individual[1], model)



def tournament_selection(population: list[tuple[float, float]], model: SurrogateModel) -> tuple[float, float]:
    competitors = random.sample(population, TOURNAMENT_SIZE)
    return max(competitors, key=lambda individual: fitness(individual, model))



def crossover(
    parent1: tuple[float, float], parent2: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    if random.random() < CROSSOVER_RATE:
        alpha = random.random()
        child1 = (
            alpha * parent1[0] + (1.0 - alpha) * parent2[0],
            alpha * parent1[1] + (1.0 - alpha) * parent2[1],
        )
        child2 = (
            alpha * parent2[0] + (1.0 - alpha) * parent1[0],
            alpha * parent2[1] + (1.0 - alpha) * parent1[1],
        )
        return child1, child2
    return parent1, parent2



def mutate(individual: tuple[float, float]) -> tuple[float, float]:
    glucose, temperature = individual

    if random.random() < MUTATION_RATE:
        glucose += random.uniform(-GLUCOSE_MUTATION_STEP, GLUCOSE_MUTATION_STEP)
    if random.random() < MUTATION_RATE:
        temperature += random.uniform(-TEMPERATURE_MUTATION_STEP, TEMPERATURE_MUTATION_STEP)

    return clamp_glucose(glucose), clamp_temperature(temperature)



def run_genetic_algorithm(
    model: SurrogateModel,
) -> tuple[list[GenerationSnapshot], tuple[float, float], float]:
    """Run the GA and store every generation for reporting and animation."""
    population = create_population(POPULATION_SIZE)
    history: list[GenerationSnapshot] = []

    for generation in range(GENERATIONS):
        population.sort(key=lambda individual: fitness(individual, model), reverse=True)
        new_population = population[:ELITE_COUNT]

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, model)
            parent2 = tournament_selection(population, model)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(child2))

        population = new_population
        best = max(population, key=lambda individual: fitness(individual, model))
        best_voltage = fitness(best, model)
        average_voltage = sum(fitness(individual, model) for individual in population) / len(population)
        history.append(
            GenerationSnapshot(
                generation_index=generation + 1,
                population=list(population),
                best_individual=best,
                best_voltage=best_voltage,
                average_voltage=average_voltage,
            )
        )

        print(
            f"Generation {generation + 1:2d}: "
            f"best glucose = {best[0]:.4f} mol dm^-3, "
            f"best temperature = {best[1]:.2f} °C, "
            f"predicted voltage = {best_voltage:.4f} V, "
            f"population average = {average_voltage:.4f} V"
        )

    final_snapshot = history[-1]
    return history, final_snapshot.best_individual, final_snapshot.best_voltage


# -----------------------------------------------------------------------------
# 3. Tkinter visualisation.
# -----------------------------------------------------------------------------


def colour_from_ratio(ratio: float) -> str:
    """Map a 0-1 ratio to a readable blue-green-yellow-red colour scale."""
    ratio = max(0.0, min(1.0, ratio))

    if ratio < 0.33:
        blend = ratio / 0.33
        red = int(30 + 20 * blend)
        green = int(60 + 150 * blend)
        blue = int(170 + 40 * blend)
    elif ratio < 0.66:
        blend = (ratio - 0.33) / 0.33
        red = int(50 + 190 * blend)
        green = int(210 + 20 * blend)
        blue = int(210 - 160 * blend)
    else:
        blend = (ratio - 0.66) / 0.34
        red = 240
        green = int(230 - 150 * blend)
        blue = int(50 - 20 * blend)

    return f"#{red:02x}{green:02x}{blue:02x}"



def create_prediction_grid(
    model: SurrogateModel,
    glucose_steps: int = 30,
    temperature_steps: int = 24,
) -> tuple[list[float], list[float], list[list[float]], float, float]:
    """Create a grid of surrogate predictions for the Tkinter heatmap."""
    glucose_values = [
        MIN_GLUCOSE + step * (MAX_GLUCOSE - MIN_GLUCOSE) / (glucose_steps - 1)
        for step in range(glucose_steps)
    ]
    temperature_values = [
        MIN_TEMPERATURE + step * (MAX_TEMPERATURE - MIN_TEMPERATURE) / (temperature_steps - 1)
        for step in range(temperature_steps)
    ]

    voltage_grid: list[list[float]] = []
    all_values: list[float] = []
    for temperature in temperature_values:
        row = [predict_voltage(glucose, temperature, model) for glucose in glucose_values]
        voltage_grid.append(row)
        all_values.extend(row)

    return glucose_values, temperature_values, voltage_grid, min(all_values), max(all_values)


class GeneticAlgorithmViewer:
    """Tkinter application that animates the GA over the surrogate model."""

    def __init__(self, model: SurrogateModel, history: list[GenerationSnapshot]) -> None:
        self.model = model
        self.history = history
        self.current_index = 0
        self.is_playing = True
        self.after_id: str | None = None

        self.root = tk.Tk()
        self.root.title("Microbial Fuel Cell GA Simulation")
        self.root.geometry("1180x900")
        self.root.configure(bg="#f4f6f8")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.main_canvas_width = 960
        self.main_canvas_height = 500
        self.chart_canvas_width = 960
        self.chart_canvas_height = 220

        self.margin_left = 90
        self.margin_right = 50
        self.margin_top = 40
        self.margin_bottom = 70
        self.plot_width = self.main_canvas_width - self.margin_left - self.margin_right
        self.plot_height = self.main_canvas_height - self.margin_top - self.margin_bottom

        self.info_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.generation_var = tk.StringVar()

        self.grid_glucose, self.grid_temperature, self.grid_voltage, self.grid_min, self.grid_max = create_prediction_grid(model)

        self._build_layout()
        self._draw_static_main_canvas()
        self._draw_static_chart_canvas()
        self.show_generation(0)
        self.schedule_next_frame()

    def _build_layout(self) -> None:
        title = ttk.Label(
            self.root,
            text="Microbial Fuel Cell Genetic Algorithm Simulation",
            font=("Arial", 20, "bold"),
        )
        title.pack(pady=(16, 4))

        subtitle = ttk.Label(
            self.root,
            text=(
                "Heatmap = surrogate model prediction, circles = GA population, "
                "star = current best solution"
            ),
            font=("Arial", 11),
        )
        subtitle.pack(pady=(0, 10))

        self.main_canvas = tk.Canvas(
            self.root,
            width=self.main_canvas_width,
            height=self.main_canvas_height,
            bg="white",
            highlightthickness=1,
            highlightbackground="#c5ccd3",
        )
        self.main_canvas.pack()

        self.chart_canvas = tk.Canvas(
            self.root,
            width=self.chart_canvas_width,
            height=self.chart_canvas_height,
            bg="white",
            highlightthickness=1,
            highlightbackground="#c5ccd3",
        )
        self.chart_canvas.pack(pady=(10, 0))

        controls = ttk.Frame(self.root, padding=(10, 12))
        controls.pack(fill="x")

        ttk.Button(controls, text="Play", command=self.play).pack(side="left", padx=4)
        ttk.Button(controls, text="Pause", command=self.pause).pack(side="left", padx=4)
        ttk.Button(controls, text="Next generation", command=self.next_generation).pack(side="left", padx=4)
        ttk.Button(controls, text="Restart", command=self.restart).pack(side="left", padx=4)

        ttk.Label(controls, textvariable=self.generation_var, font=("Arial", 11, "bold")).pack(side="left", padx=16)
        ttk.Label(controls, textvariable=self.status_var, font=("Arial", 10)).pack(side="left", padx=4)

        info_panel = ttk.Frame(self.root, padding=(10, 0, 10, 14))
        info_panel.pack(fill="x")
        ttk.Label(info_panel, textvariable=self.info_var, font=("Courier New", 10), justify="left").pack(anchor="w")

    def x_to_canvas(self, glucose: float) -> float:
        scale = (glucose - MIN_GLUCOSE) / (MAX_GLUCOSE - MIN_GLUCOSE)
        return self.margin_left + scale * self.plot_width

    def y_to_canvas(self, temperature: float) -> float:
        scale = (temperature - MIN_TEMPERATURE) / (MAX_TEMPERATURE - MIN_TEMPERATURE)
        return self.main_canvas_height - self.margin_bottom - scale * self.plot_height

    def chart_x(self, generation: int) -> float:
        chart_left = 70
        chart_right = self.chart_canvas_width - 35
        width = chart_right - chart_left
        return chart_left + (generation - 1) * width / max(1, GENERATIONS - 1)

    def chart_y(self, voltage: float, min_voltage: float, max_voltage: float) -> float:
        chart_top = 25
        chart_bottom = self.chart_canvas_height - 35
        height = chart_bottom - chart_top
        if abs(max_voltage - min_voltage) < 1e-9:
            return chart_bottom - height / 2
        ratio = (voltage - min_voltage) / (max_voltage - min_voltage)
        return chart_bottom - ratio * height

    def _draw_static_main_canvas(self) -> None:
        self.main_canvas.create_rectangle(
            self.margin_left,
            self.margin_top,
            self.main_canvas_width - self.margin_right,
            self.main_canvas_height - self.margin_bottom,
            fill="#fbfcfd",
            outline="#d0d7de",
        )

        self._draw_heatmap()
        self._draw_axes()
        self._draw_heatmap_legend()

    def _draw_heatmap(self) -> None:
        for row_index in range(len(self.grid_temperature) - 1):
            for column_index in range(len(self.grid_glucose) - 1):
                left = self.x_to_canvas(self.grid_glucose[column_index])
                right = self.x_to_canvas(self.grid_glucose[column_index + 1])
                bottom = self.y_to_canvas(self.grid_temperature[row_index])
                top = self.y_to_canvas(self.grid_temperature[row_index + 1])
                voltage = self.grid_voltage[row_index][column_index]
                ratio = (voltage - self.grid_min) / (self.grid_max - self.grid_min)
                colour = colour_from_ratio(ratio)
                self.main_canvas.create_rectangle(
                    left,
                    top,
                    right,
                    bottom,
                    fill=colour,
                    outline=colour,
                    tags="background",
                )

    def _draw_axes(self) -> None:
        self.main_canvas.create_line(
            self.margin_left,
            self.margin_top,
            self.margin_left,
            self.main_canvas_height - self.margin_bottom,
            width=2,
        )
        self.main_canvas.create_line(
            self.margin_left,
            self.main_canvas_height - self.margin_bottom,
            self.main_canvas_width - self.margin_right,
            self.main_canvas_height - self.margin_bottom,
            width=2,
        )

        for tick_count in range(5):
            tick = MIN_GLUCOSE + tick_count * (MAX_GLUCOSE - MIN_GLUCOSE) / 4
            x_position = self.x_to_canvas(tick)
            self.main_canvas.create_line(
                x_position,
                self.main_canvas_height - self.margin_bottom,
                x_position,
                self.main_canvas_height - self.margin_bottom + 7,
                width=2,
            )
            self.main_canvas.create_text(
                x_position,
                self.main_canvas_height - self.margin_bottom + 24,
                text=f"{tick:.2f}",
                font=("Arial", 10),
            )

        for tick_count in range(5):
            tick = MIN_TEMPERATURE + tick_count * (MAX_TEMPERATURE - MIN_TEMPERATURE) / 4
            y_position = self.y_to_canvas(tick)
            self.main_canvas.create_line(self.margin_left - 7, y_position, self.margin_left, y_position, width=2)
            self.main_canvas.create_text(self.margin_left - 34, y_position, text=f"{tick:.0f}", font=("Arial", 10))
            self.main_canvas.create_line(
                self.margin_left,
                y_position,
                self.main_canvas_width - self.margin_right,
                y_position,
                fill="#edf1f5",
            )

        self.main_canvas.create_text(
            self.main_canvas_width / 2,
            self.main_canvas_height - 24,
            text="Glucose concentration (mol dm^-3)",
            font=("Arial", 12, "bold"),
        )
        self.main_canvas.create_text(
            32,
            self.main_canvas_height / 2,
            text="Temperature (°C)",
            angle=90,
            font=("Arial", 12, "bold"),
        )
        self.main_canvas.create_text(
            self.main_canvas_width / 2,
            18,
            text="GA search on the fitted surrogate model",
            font=("Arial", 13, "bold"),
        )

    def _draw_heatmap_legend(self) -> None:
        legend_left = self.main_canvas_width - 250
        legend_top = 52
        self.main_canvas.create_rectangle(
            legend_left,
            legend_top,
            self.main_canvas_width - 35,
            184,
            fill="white",
            outline="#8c959f",
        )
        self.main_canvas.create_text(self.main_canvas_width - 143, 72, text="Legend", font=("Arial", 11, "bold"))
        self.main_canvas.create_text(
            self.main_canvas_width - 143,
            94,
            text="Background = predicted voltage",
            font=("Arial", 9),
        )
        self.main_canvas.create_text(
            self.main_canvas_width - 143,
            112,
            text="Circles = current GA population",
            font=("Arial", 9),
        )
        self.main_canvas.create_text(
            self.main_canvas_width - 143,
            130,
            text="Star = best individual",
            font=("Arial", 9),
        )

        gradient_left = legend_left + 18
        gradient_top = 145
        gradient_width = 140
        for index in range(70):
            ratio = index / 69
            colour = colour_from_ratio(ratio)
            self.main_canvas.create_rectangle(
                gradient_left + index * 2,
                gradient_top,
                gradient_left + (index + 1) * 2,
                gradient_top + 14,
                fill=colour,
                outline=colour,
            )
        self.main_canvas.create_text(
            gradient_left,
            gradient_top + 25,
            text=f"{self.grid_min:.2f} V",
            anchor="w",
            font=("Arial", 8),
        )
        self.main_canvas.create_text(
            gradient_left + gradient_width,
            gradient_top + 25,
            text=f"{self.grid_max:.2f} V",
            anchor="e",
            font=("Arial", 8),
        )

    def _draw_static_chart_canvas(self) -> None:
        chart_left = 70
        chart_top = 25
        chart_right = self.chart_canvas_width - 35
        chart_bottom = self.chart_canvas_height - 35

        self.chart_canvas.create_rectangle(
            chart_left,
            chart_top,
            chart_right,
            chart_bottom,
            fill="#fbfcfd",
            outline="#d0d7de",
        )
        self.chart_canvas.create_text(
            self.chart_canvas_width / 2,
            12,
            text="Convergence curve (best and average predicted voltage)",
            font=("Arial", 12, "bold"),
        )

        for tick_count in range(6):
            generation_value = 1 + tick_count * (GENERATIONS - 1) / 5
            x_position = chart_left + tick_count * (chart_right - chart_left) / 5
            self.chart_canvas.create_line(x_position, chart_bottom, x_position, chart_bottom + 6, width=1)
            self.chart_canvas.create_text(x_position, chart_bottom + 18, text=f"{generation_value:.0f}", font=("Arial", 9))

        self.chart_canvas.create_text(
            self.chart_canvas_width / 2,
            self.chart_canvas_height - 10,
            text="Generation",
            font=("Arial", 10, "bold"),
        )
        self.chart_canvas.create_text(20, self.chart_canvas_height / 2, text="Voltage", angle=90, font=("Arial", 10, "bold"))

    def _draw_convergence_curves(self, current_generation: int) -> None:
        self.chart_canvas.delete("dynamic_chart")

        best_values = [snapshot.best_voltage for snapshot in self.history]
        average_values = [snapshot.average_voltage for snapshot in self.history]
        min_voltage = min(average_values)
        max_voltage = max(best_values)

        chart_left = 70
        chart_top = 25
        chart_right = self.chart_canvas_width - 35
        chart_bottom = self.chart_canvas_height - 35

        for tick_count in range(5):
            ratio = tick_count / 4
            value = min_voltage + ratio * (max_voltage - min_voltage)
            y_position = self.chart_y(value, min_voltage, max_voltage)
            self.chart_canvas.create_line(chart_left, y_position, chart_right, y_position, fill="#edf1f5", tags="dynamic_chart")
            self.chart_canvas.create_text(48, y_position, text=f"{value:.2f}", font=("Arial", 9), tags="dynamic_chart")

        best_points: list[float] = []
        average_points: list[float] = []
        for generation in range(1, current_generation + 1):
            best_points.extend([
                self.chart_x(generation),
                self.chart_y(best_values[generation - 1], min_voltage, max_voltage),
            ])
            average_points.extend([
                self.chart_x(generation),
                self.chart_y(average_values[generation - 1], min_voltage, max_voltage),
            ])

        if len(best_points) >= 4:
            self.chart_canvas.create_line(*average_points, fill="#1f77b4", width=2, smooth=True, tags="dynamic_chart")
            self.chart_canvas.create_line(*best_points, fill="#d62728", width=2.5, smooth=True, tags="dynamic_chart")

        current_x = self.chart_x(current_generation)
        self.chart_canvas.create_line(current_x, chart_top, current_x, chart_bottom, fill="#444444", dash=(4, 3), tags="dynamic_chart")
        self.chart_canvas.create_text(
            chart_right - 110,
            chart_top + 14,
            text="Red = best",
            fill="#d62728",
            font=("Arial", 9, "bold"),
            tags="dynamic_chart",
        )
        self.chart_canvas.create_text(
            chart_right - 104,
            chart_top + 32,
            text="Blue = average",
            fill="#1f77b4",
            font=("Arial", 9, "bold"),
            tags="dynamic_chart",
        )

    def show_generation(self, index: int) -> None:
        snapshot = self.history[index]
        self.main_canvas.delete("entity")

        for glucose, temperature in snapshot.population:
            x_position = self.x_to_canvas(glucose)
            y_position = self.y_to_canvas(temperature)
            predicted_voltage = fitness((glucose, temperature), self.model)
            ratio = (predicted_voltage - self.grid_min) / (self.grid_max - self.grid_min)
            colour = colour_from_ratio(ratio)
            radius = 7
            self.main_canvas.create_oval(
                x_position - radius,
                y_position - radius,
                x_position + radius,
                y_position + radius,
                fill=colour,
                outline="#1f2328",
                width=1.0,
                tags="entity",
            )

        best = snapshot.best_individual
        best_x = self.x_to_canvas(best[0])
        best_y = self.y_to_canvas(best[1])
        self.main_canvas.create_text(best_x, best_y - 18, text="Best", font=("Arial", 9, "bold"), tags="entity")
        self.main_canvas.create_text(best_x, best_y, text="★", fill="#111111", font=("Arial", 20, "bold"), tags="entity")

        self._draw_convergence_curves(snapshot.generation_index)

        self.generation_var.set(f"Generation {snapshot.generation_index} / {len(self.history)}")
        self.status_var.set("Playing" if self.is_playing else "Paused")
        self.info_var.set(
            "\n".join(
                [
                    f"Best glucose concentration : {best[0]:.4f} mol dm^-3",
                    f"Best temperature           : {best[1]:.2f} °C",
                    f"Predicted best voltage     : {snapshot.best_voltage:.4f} V",
                    f"Average population voltage : {snapshot.average_voltage:.4f} V",
                    f"Model R^2 / RMSE           : {self.model.r_squared:.4f} / {self.model.rmse:.4f} V",
                ]
            )
        )

    def schedule_next_frame(self) -> None:
        if self.is_playing:
            self.after_id = self.root.after(ANIMATION_DELAY_MS, self.advance_frame)

    def advance_frame(self) -> None:
        if not self.is_playing:
            return
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
        else:
            self.current_index = 0
        self.show_generation(self.current_index)
        self.schedule_next_frame()

    def play(self) -> None:
        if self.is_playing:
            return
        self.is_playing = True
        self.status_var.set("Playing")
        self.schedule_next_frame()

    def pause(self) -> None:
        self.is_playing = False
        self.status_var.set("Paused")
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def next_generation(self) -> None:
        self.pause()
        self.current_index = (self.current_index + 1) % len(self.history)
        self.show_generation(self.current_index)

    def restart(self) -> None:
        self.pause()
        self.current_index = 0
        self.show_generation(self.current_index)

    def on_close(self) -> None:
        self.pause()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# -----------------------------------------------------------------------------
# 4. Main program.
# -----------------------------------------------------------------------------


def main() -> None:
    random.seed(RANDOM_SEED)

    dataset = create_sample_dataset()
    model = fit_surrogate_model(dataset)
    print_model_summary(model, dataset)

    history, best_solution, best_voltage = run_genetic_algorithm(model)

    print("\nFinal optimisation result")
    print("=" * 78)
    print(f"Optimal glucose concentration : {best_solution[0]:.4f} mol dm^-3")
    print(f"Optimal temperature           : {best_solution[1]:.2f} °C")
    print(f"Predicted maximum voltage     : {best_voltage:.4f} V")
    print("=" * 78)

    try:
        viewer = GeneticAlgorithmViewer(model, history)
        viewer.run()
    except tk.TclError as error:
        print(
            "Tkinter window could not be opened. "
            "This usually means no graphical display is available "
            f"or Tk is not configured correctly.\nDetails: {error}"
        )


if __name__ == "__main__":
    main()

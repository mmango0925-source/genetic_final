import math
import os
import random
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import NamedTuple


# =============================================================================
# Microbial fuel cell optimisation project
# -----------------------------------------------------------------------------
# Workflow:
#   1. Create a small experimental-style dataset.
#   2. Fit a quadratic surrogate model to the dataset.
#   3. Use a genetic algorithm (GA) to optimise the fitted model.
#   4. Animate each GA generation in Tkinter so users can watch the search.
# =============================================================================

MIN_GLUCOSE = 0.00
MAX_GLUCOSE = 0.20
MIN_TEMPERATURE = 20.0
MAX_TEMPERATURE = 40.0

POPULATION_SIZE = 40
GENERATIONS = 60
MUTATION_RATE = 0.20
CROSSOVER_RATE = 0.85
TOURNAMENT_SIZE = 3
ELITE_COUNT = 2
GLUCOSE_MUTATION_STEP = 0.010
TEMPERATURE_MUTATION_STEP = 1.50
RANDOM_SEED = 7
ANIMATION_DELAY_MS = 450


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
    """Stores one generation of GA results for animation."""

    generation_index: int
    population: list[tuple[float, float]]
    best_individual: tuple[float, float]
    best_voltage: float


# -----------------------------------------------------------------------------
# 1. Dataset and surrogate model.
# -----------------------------------------------------------------------------


def create_sample_dataset() -> list[Observation]:
    """Create a realistic sample dataset with a non-linear optimum region."""
    return [
        Observation(0.02, 22.0, 0.22),
        Observation(0.02, 22.0, 0.24),
        Observation(0.02, 30.0, 0.43),
        Observation(0.02, 30.0, 0.45),
        Observation(0.02, 38.0, 0.27),
        Observation(0.02, 38.0, 0.29),
        Observation(0.06, 24.0, 0.51),
        Observation(0.06, 24.0, 0.49),
        Observation(0.06, 32.0, 0.78),
        Observation(0.06, 32.0, 0.80),
        Observation(0.06, 40.0, 0.46),
        Observation(0.06, 40.0, 0.44),
        Observation(0.10, 26.0, 0.88),
        Observation(0.10, 26.0, 0.91),
        Observation(0.10, 34.0, 1.20),
        Observation(0.10, 34.0, 1.22),
        Observation(0.10, 38.0, 0.99),
        Observation(0.10, 38.0, 1.01),
        Observation(0.14, 24.0, 0.93),
        Observation(0.14, 24.0, 0.95),
        Observation(0.14, 32.0, 1.34),
        Observation(0.14, 32.0, 1.37),
        Observation(0.14, 36.0, 1.24),
        Observation(0.14, 36.0, 1.27),
        Observation(0.18, 22.0, 0.58),
        Observation(0.18, 22.0, 0.56),
        Observation(0.18, 30.0, 0.92),
        Observation(0.18, 30.0, 0.95),
        Observation(0.18, 38.0, 0.74),
        Observation(0.18, 38.0, 0.72),
    ]



def build_feature_row(glucose: float, temperature: float) -> list[float]:
    """Create one quadratic-regression feature row."""
    return [
        1.0,
        glucose,
        temperature,
        glucose * glucose,
        temperature * temperature,
        glucose * temperature,
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
    """Fit a quadratic regression model using the normal equations."""
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
            "Glucose",
            "Temperature",
            "Glucose^2",
            "Temperature^2",
            "Glucose*Temperature",
        ],
        r_squared=r_squared,
        rmse=rmse,
    )



def predict_voltage(glucose: float, temperature: float, model: SurrogateModel) -> float:
    """Predict voltage from glucose and temperature using the fitted model."""
    features = build_feature_row(glucose, temperature)
    return sum(value * coefficient for value, coefficient in zip(features, model.coefficients))



def print_model_summary(model: SurrogateModel) -> None:
    """Print a concise summary of the fitted model."""
    print("Fitted surrogate model")
    print("=" * 68)
    print("V = b0 + b1*G + b2*T + b3*G^2 + b4*T^2 + b5*G*T")
    print()
    for name, coefficient in zip(model.feature_names, model.coefficients):
        print(f"{name:<22} = {coefficient: .6f}")
    print()
    print(f"Model R^2  : {model.r_squared:.4f}")
    print(f"Model RMSE : {model.rmse:.4f} V")
    print("=" * 68)


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
    """Run the GA and store every generation for Tkinter playback."""
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
        history.append(
            GenerationSnapshot(
                generation_index=generation + 1,
                population=list(population),
                best_individual=best,
                best_voltage=best_voltage,
            )
        )

        print(
            f"Generation {generation + 1:2d}: "
            f"best glucose = {best[0]:.4f} mol dm^-3, "
            f"best temperature = {best[1]:.2f} °C, "
            f"predicted voltage = {best_voltage:.4f} V"
        )

    final_snapshot = history[-1]
    return history, final_snapshot.best_individual, final_snapshot.best_voltage


# -----------------------------------------------------------------------------
# 3. Tkinter visualisation.
# -----------------------------------------------------------------------------


def glucose_to_colour(glucose: float) -> str:
    """Map glucose concentration to a blue-purple-red gradient."""
    ratio = (glucose - MIN_GLUCOSE) / (MAX_GLUCOSE - MIN_GLUCOSE)
    ratio = max(0.0, min(1.0, ratio))

    red = int(40 + 215 * ratio)
    green = int(70 + 50 * (1.0 - abs(ratio - 0.5) * 2.0))
    blue = int(255 - 180 * ratio)
    return f"#{red:02x}{green:02x}{blue:02x}"


class GeneticAlgorithmViewer:
    """Tkinter application that animates every GA generation."""

    def __init__(self, model: SurrogateModel, history: list[GenerationSnapshot]) -> None:
        self.model = model
        self.history = history
        self.current_index = 0
        self.is_playing = True
        self.after_id: str | None = None

        self.root = tk.Tk()
        self.root.title("Microbial Fuel Cell GA Simulation")
        self.root.geometry("1020x760")
        self.root.configure(bg="#f4f6f8")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.canvas_width = 860
        self.canvas_height = 520
        self.margin_left = 90
        self.margin_right = 50
        self.margin_top = 40
        self.margin_bottom = 80
        self.plot_width = self.canvas_width - self.margin_left - self.margin_right
        self.plot_height = self.canvas_height - self.margin_top - self.margin_bottom

        self.info_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.generation_var = tk.StringVar()

        self._build_layout()
        self._draw_static_canvas_elements()
        self.show_generation(0)
        self.schedule_next_frame()

    def _build_layout(self) -> None:
        title = ttk.Label(
            self.root,
            text="Microbial Fuel Cell Genetic Algorithm Simulation",
            font=("Arial", 19, "bold"),
        )
        title.pack(pady=(18, 6))

        subtitle = ttk.Label(
            self.root,
            text="Each circle is one entity in the current generation. Colour shows glucose concentration.",
            font=("Arial", 11),
        )
        subtitle.pack(pady=(0, 12))

        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="white",
            highlightthickness=1,
            highlightbackground="#c5ccd3",
        )
        self.canvas.pack()

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
        return self.canvas_height - self.margin_bottom - scale * self.plot_height

    def _draw_static_canvas_elements(self) -> None:
        self.canvas.create_rectangle(
            self.margin_left,
            self.margin_top,
            self.canvas_width - self.margin_right,
            self.canvas_height - self.margin_bottom,
            fill="#fbfcfd",
            outline="#d0d7de",
        )

        self.canvas.create_line(
            self.margin_left,
            self.margin_top,
            self.margin_left,
            self.canvas_height - self.margin_bottom,
            width=2,
        )
        self.canvas.create_line(
            self.margin_left,
            self.canvas_height - self.margin_bottom,
            self.canvas_width - self.margin_right,
            self.canvas_height - self.margin_bottom,
            width=2,
        )

        for tick_count in range(5):
            tick = MIN_GLUCOSE + tick_count * (MAX_GLUCOSE - MIN_GLUCOSE) / 4
            x_position = self.x_to_canvas(tick)
            self.canvas.create_line(
                x_position,
                self.canvas_height - self.margin_bottom,
                x_position,
                self.canvas_height - self.margin_bottom + 7,
                width=2,
            )
            self.canvas.create_text(
                x_position,
                self.canvas_height - self.margin_bottom + 24,
                text=f"{tick:.2f}",
                font=("Arial", 10),
            )

        for tick_count in range(5):
            tick = MIN_TEMPERATURE + tick_count * (MAX_TEMPERATURE - MIN_TEMPERATURE) / 4
            y_position = self.y_to_canvas(tick)
            self.canvas.create_line(self.margin_left - 7, y_position, self.margin_left, y_position, width=2)
            self.canvas.create_text(self.margin_left - 34, y_position, text=f"{tick:.0f}", font=("Arial", 10))
            self.canvas.create_line(
                self.margin_left,
                y_position,
                self.canvas_width - self.margin_right,
                y_position,
                fill="#edf1f5",
            )

        self.canvas.create_text(
            self.canvas_width / 2,
            self.canvas_height - 28,
            text="Glucose concentration (mol dm^-3)",
            font=("Arial", 12, "bold"),
        )
        self.canvas.create_text(
            30,
            self.canvas_height / 2,
            text="Temperature (°C)",
            angle=90,
            font=("Arial", 12, "bold"),
        )
        self.canvas.create_text(
            self.canvas_width / 2,
            18,
            text="Generation-by-generation search in the design space",
            font=("Arial", 13, "bold"),
        )

        self._draw_legend()

    def _draw_legend(self) -> None:
        legend_left = self.canvas_width - 235
        legend_top = 48
        self.canvas.create_rectangle(
            legend_left,
            legend_top,
            self.canvas_width - 35,
            172,
            fill="white",
            outline="#8c959f",
        )
        self.canvas.create_text(self.canvas_width - 135, 67, text="Legend", font=("Arial", 11, "bold"))
        self.canvas.create_text(
            self.canvas_width - 135,
            89,
            text="Circle position = glucose + temperature",
            font=("Arial", 9),
        )
        self.canvas.create_text(
            self.canvas_width - 135,
            108,
            text="Circle colour = glucose concentration",
            font=("Arial", 9),
        )

        gradient_left = legend_left + 18
        gradient_top = 122
        gradient_width = 120
        for index in range(60):
            ratio = index / 59
            glucose = MIN_GLUCOSE + ratio * (MAX_GLUCOSE - MIN_GLUCOSE)
            colour = glucose_to_colour(glucose)
            self.canvas.create_rectangle(
                gradient_left + index * 2,
                gradient_top,
                gradient_left + (index + 1) * 2,
                gradient_top + 14,
                fill=colour,
                outline=colour,
            )
        self.canvas.create_text(gradient_left, gradient_top + 25, text=f"{MIN_GLUCOSE:.2f}", anchor="w", font=("Arial", 8))
        self.canvas.create_text(gradient_left + gradient_width, gradient_top + 25, text=f"{MAX_GLUCOSE:.2f}", anchor="e", font=("Arial", 8))
        self.canvas.create_text(gradient_left + gradient_width / 2, gradient_top + 25, text="glucose", font=("Arial", 8))

    def show_generation(self, index: int) -> None:
        snapshot = self.history[index]
        self.canvas.delete("entity")

        for glucose, temperature in snapshot.population:
            x_position = self.x_to_canvas(glucose)
            y_position = self.y_to_canvas(temperature)
            colour = glucose_to_colour(glucose)
            radius = 7
            self.canvas.create_oval(
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
        self.canvas.create_oval(
            best_x - 12,
            best_y - 12,
            best_x + 12,
            best_y + 12,
            outline="#111111",
            width=3,
            tags="entity",
        )
        self.canvas.create_text(best_x, best_y - 18, text="Best", font=("Arial", 9, "bold"), tags="entity")

        self.generation_var.set(f"Generation {snapshot.generation_index} / {len(self.history)}")
        self.status_var.set("Playing" if self.is_playing else "Paused")
        self.info_var.set(
            "\n".join(
                [
                    f"Best glucose concentration : {best[0]:.4f} mol dm^-3",
                    f"Best temperature           : {best[1]:.2f} °C",
                    f"Predicted voltage          : {snapshot.best_voltage:.4f} V",
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
        self.current_index = (self.current_index + 1) % len(self.history)
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

    model = fit_surrogate_model(create_sample_dataset())
    print_model_summary(model)

    history, best_solution, best_voltage = run_genetic_algorithm(model)

    print("\nFinal optimisation result")
    print("=" * 68)
    print(f"Optimal glucose concentration : {best_solution[0]:.4f} mol dm^-3")
    print(f"Optimal temperature           : {best_solution[1]:.2f} °C")
    print(f"Predicted maximum voltage     : {best_voltage:.4f} V")
    print("=" * 68)

    if os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print(
            "Tkinter window not opened because no graphical display is available. "
            "Run this program on a local desktop environment to view the animation."
        )
        return

    viewer = GeneticAlgorithmViewer(model, history)
    viewer.run()


if __name__ == "__main__":
    main()

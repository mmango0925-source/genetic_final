import math
import random
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Callable, NamedTuple


# =============================================================================
# Microbial fuel cell optimisation project
# -----------------------------------------------------------------------------
# Workflow:
#   1. Create an experimental-style dataset.
#   2. Fit a symbolic regression model to the dataset.
#   3. Treat that discovered equation as a surrogate model.
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
SURFACE_PLOT_FILE = "surface_plot.png"

GLUCOSE_CENTRE = 0.10
GLUCOSE_SCALE = 0.08
TEMPERATURE_CENTRE = 30.0
TEMPERATURE_SCALE = 8.0


class Observation(NamedTuple):
    """One row of experimental-style data."""

    glucose_concentration: float
    temperature: float
    voltage: float


@dataclass(frozen=True)
class ExpressionNode:
    """One node in a symbolic expression tree used by genetic programming."""

    node_type: str
    value: str | float
    left: 'ExpressionNode | None' = None
    right: 'ExpressionNode | None' = None


@dataclass(frozen=True)
class SurrogateModel:
    """Stores the best symbolic-regression expression and its fit statistics."""

    expression_tree: ExpressionNode
    best_expression: str
    output_scale: float
    output_bias: float
    best_fitness: float
    r_squared: float
    rmse: float
    complexity: int
    history: tuple[float, ...]


@dataclass(frozen=True)
class GenerationSnapshot:
    """Stores one generation of GA results for reporting and animation."""

    generation_index: int
    population: list[tuple[float, float]]
    best_individual: tuple[float, float]
    best_voltage: float
    average_voltage: float


# -----------------------------------------------------------------------------
# 1. Dataset and symbolic surrogate model.
# -----------------------------------------------------------------------------

SYMBOLIC_POPULATION_SIZE = 180
SYMBOLIC_GENERATIONS = 50
SYMBOLIC_TOURNAMENT_SIZE = 4
SYMBOLIC_MUTATION_RATE = 0.25
SYMBOLIC_CROSSOVER_RATE = 0.80
SYMBOLIC_ELITE_COUNT = 3
SYMBOLIC_INITIAL_MAX_DEPTH = 5
SYMBOLIC_MAX_DEPTH = 7
SYMBOLIC_COMPLEXITY_PENALTY = 0.0009
SYMBOLIC_OUTPUT_CLAMP = 2.5
SYMBOLIC_INVALID_FITNESS = 1_000_000.0
CONSTANT_MUTATION_STD = 0.40

BINARY_OPERATORS = ('+', '-', '*', '/')
UNARY_OPERATORS = ('sin', 'cos', 'exp', 'log')



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



def protected_divide(numerator: float, denominator: float, epsilon: float = 1e-6) -> float:
    """Protected division to keep evolved equations numerically safe."""
    if abs(denominator) < epsilon:
        denominator = epsilon if denominator >= 0.0 else -epsilon
    return numerator / denominator



def protected_log(value: float, epsilon: float = 1e-6) -> float:
    """Protected logarithm using log(|x| + eps)."""
    return math.log(abs(value) + epsilon)



def protected_exp(value: float) -> float:
    """Protected exponential with clipping so the GP stays CPU-friendly."""
    return math.exp(max(-12.0, min(12.0, value)))



def scale_inputs(glucose: float, temperature: float) -> tuple[float, float]:
    """Create centred/scaled aliases used by the symbolic GP for stability."""
    g = (glucose - GLUCOSE_CENTRE) / GLUCOSE_SCALE
    t = (temperature - TEMPERATURE_CENTRE) / TEMPERATURE_SCALE
    return g, t



def clip_numeric(value: float, limit: float = 1_000_000.0) -> float:
    """Clamp extreme intermediate values so invalid trees do not explode."""
    if not math.isfinite(value):
        return float('nan')
    return max(-limit, min(limit, value))



def clamp_prediction(value: float) -> float:
    """Clamp final predictions to a physically reasonable range."""
    if not math.isfinite(value):
        return float('nan')
    return max(-0.20, min(SYMBOLIC_OUTPUT_CLAMP, value))



def random_constant(rng: random.Random) -> float:
    """Create an ephemeral random constant for a new expression tree."""
    reusable_constants = [
        -2.0,
        -1.0,
        -0.5,
        -0.25,
        0.0,
        0.02,
        0.05,
        0.10,
        0.12,
        0.16,
        0.20,
        0.25,
        0.5,
        1.0,
        2.0,
        5.0,
        8.0,
        22.0,
        25.0,
        30.0,
        31.0,
        34.0,
        37.0,
        40.0,
    ]
    if rng.random() < 0.55:
        return float(rng.choice(reusable_constants))
    if rng.random() < 0.50:
        return round(rng.uniform(-2.0, 2.0), 4)
    if rng.random() < 0.50:
        return round(rng.uniform(MIN_GLUCOSE, MAX_GLUCOSE), 4)
    return round(rng.uniform(MIN_TEMPERATURE, MAX_TEMPERATURE), 4)



def make_constant_node(rng: random.Random) -> ExpressionNode:
    return ExpressionNode('constant', random_constant(rng))



def make_variable_node(rng: random.Random) -> ExpressionNode:
    variable_choices = ('glucose', 'temperature', 'g', 't', 'g', 't')
    return ExpressionNode('variable', rng.choice(variable_choices))



def clone_expression(node: ExpressionNode) -> ExpressionNode:
    """Deep-copy an expression tree."""
    return ExpressionNode(
        node_type=node.node_type,
        value=node.value,
        left=clone_expression(node.left) if node.left is not None else None,
        right=clone_expression(node.right) if node.right is not None else None,
    )



def create_random_terminal(rng: random.Random) -> ExpressionNode:
    """Create a variable or constant terminal."""
    if rng.random() < 0.55:
        return make_variable_node(rng)
    return make_constant_node(rng)



def create_random_expression(
    rng: random.Random,
    max_depth: int,
    current_depth: int = 0,
    force_function: bool = False,
) -> ExpressionNode:
    """Create a random expression tree using a standard GP grow strategy."""
    at_max_depth = current_depth >= max_depth
    choose_terminal = at_max_depth or (not force_function and current_depth > 0 and rng.random() < 0.28)
    if choose_terminal:
        return create_random_terminal(rng)

    if rng.random() < 0.35:
        operator = rng.choice(UNARY_OPERATORS)
        child = create_random_expression(rng, max_depth, current_depth + 1)
        return ExpressionNode('unary', operator, left=child)

    operator = rng.choice(BINARY_OPERATORS)
    left = create_random_expression(rng, max_depth, current_depth + 1)
    right = create_random_expression(rng, max_depth, current_depth + 1)
    return ExpressionNode('binary', operator, left=left, right=right)



def expression_depth(node: ExpressionNode) -> int:
    """Return the maximum tree depth."""
    if node.node_type in {'constant', 'variable'}:
        return 1
    if node.node_type == 'unary' and node.left is not None:
        return 1 + expression_depth(node.left)
    if node.left is not None and node.right is not None:
        return 1 + max(expression_depth(node.left), expression_depth(node.right))
    return 1



def expression_complexity(node: ExpressionNode) -> int:
    """Count nodes as a simple equation-complexity measure."""
    if node.node_type in {'constant', 'variable'}:
        return 1
    if node.node_type == 'unary' and node.left is not None:
        return 1 + expression_complexity(node.left)
    if node.left is not None and node.right is not None:
        return 1 + expression_complexity(node.left) + expression_complexity(node.right)
    return 1



def expression_to_string(node: ExpressionNode) -> str:
    """Convert an expression tree to a readable infix string."""
    if node.node_type == 'constant':
        return f'{float(node.value):.4f}'.rstrip('0').rstrip('.')
    if node.node_type == 'variable':
        return str(node.value)
    if node.node_type == 'unary' and node.left is not None:
        child = expression_to_string(node.left)
        if node.value == 'log':
            return f'protected_log({child})'
        return f'{node.value}({child})'
    if node.left is not None and node.right is not None:
        left = expression_to_string(node.left)
        right = expression_to_string(node.right)
        if node.value == '/':
            return f'protected_div({left}, {right})'
        return f'({left} {node.value} {right})'
    return '0.0'



def evaluate_expression(node: ExpressionNode, glucose: float, temperature: float) -> float:
    """Safely evaluate one expression for one input row."""
    try:
        if node.node_type == 'constant':
            return float(node.value)
        if node.node_type == 'variable':
            g, t = scale_inputs(glucose, temperature)
            if node.value == 'glucose':
                return glucose
            if node.value == 'temperature':
                return temperature
            if node.value == 'g':
                return g
            return t
        if node.node_type == 'unary' and node.left is not None:
            child_value = evaluate_expression(node.left, glucose, temperature)
            if not math.isfinite(child_value):
                return float('nan')
            if node.value == 'sin':
                return clip_numeric(math.sin(child_value))
            if node.value == 'cos':
                return clip_numeric(math.cos(child_value))
            if node.value == 'exp':
                return clip_numeric(protected_exp(child_value))
            if node.value == 'log':
                return clip_numeric(protected_log(child_value))
        if node.left is not None and node.right is not None:
            left_value = evaluate_expression(node.left, glucose, temperature)
            right_value = evaluate_expression(node.right, glucose, temperature)
            if not math.isfinite(left_value) or not math.isfinite(right_value):
                return float('nan')
            if node.value == '+':
                return clip_numeric(left_value + right_value)
            if node.value == '-':
                return clip_numeric(left_value - right_value)
            if node.value == '*':
                return clip_numeric(left_value * right_value)
            if node.value == '/':
                return clip_numeric(protected_divide(left_value, right_value))
    except (OverflowError, ValueError):
        return float('nan')
    return float('nan')



def all_subtree_paths(node: ExpressionNode, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Return all subtree paths in the tree so crossover/mutation can target them."""
    paths = [prefix]
    if node.left is not None:
        paths.extend(all_subtree_paths(node.left, prefix + ('left',)))
    if node.right is not None:
        paths.extend(all_subtree_paths(node.right, prefix + ('right',)))
    return paths



def get_subtree(node: ExpressionNode, path: tuple[str, ...]) -> ExpressionNode:
    """Read a subtree by path."""
    current = node
    for step in path:
        current = current.left if step == 'left' else current.right
        if current is None:
            raise ValueError('Invalid subtree path.')
    return current



def replace_subtree(node: ExpressionNode, path: tuple[str, ...], new_subtree: ExpressionNode) -> ExpressionNode:
    """Return a new tree with one subtree replaced."""
    if not path:
        return clone_expression(new_subtree)

    step = path[0]
    remaining = path[1:]
    if step == 'left' and node.left is not None:
        return ExpressionNode(node.node_type, node.value, replace_subtree(node.left, remaining, new_subtree), clone_expression(node.right) if node.right is not None else None)
    if step == 'right' and node.right is not None:
        return ExpressionNode(node.node_type, node.value, clone_expression(node.left) if node.left is not None else None, replace_subtree(node.right, remaining, new_subtree))
    return clone_expression(node)



def mutate_constant_value(value: float, rng: random.Random) -> float:
    """Perturb an evolved constant while keeping the mutation scale reasonable."""
    scale = max(0.05, abs(value) * 0.15)
    return round(value + rng.gauss(0.0, max(CONSTANT_MUTATION_STD, scale)), 4)



def mutate_expression(node: ExpressionNode, rng: random.Random, max_depth: int) -> ExpressionNode:
    """Apply subtree mutation, operator mutation, or constant mutation."""
    base_tree = clone_expression(node)
    target_path = rng.choice(all_subtree_paths(base_tree))
    target = get_subtree(base_tree, target_path)

    if target.node_type == 'constant' and rng.random() < 0.55:
        replacement = ExpressionNode('constant', mutate_constant_value(float(target.value), rng))
    elif target.node_type == 'variable' and rng.random() < 0.25:
        replacement = ExpressionNode('variable', 'temperature' if target.value == 'glucose' else 'glucose')
    elif target.node_type == 'unary' and rng.random() < 0.50:
        choices = [operator for operator in UNARY_OPERATORS if operator != target.value]
        replacement = ExpressionNode('unary', rng.choice(choices), left=clone_expression(target.left) if target.left is not None else create_random_terminal(rng))
    elif target.node_type == 'binary' and rng.random() < 0.50:
        choices = [operator for operator in BINARY_OPERATORS if operator != target.value]
        replacement = ExpressionNode(
            'binary',
            rng.choice(choices),
            left=clone_expression(target.left) if target.left is not None else create_random_terminal(rng),
            right=clone_expression(target.right) if target.right is not None else create_random_terminal(rng),
        )
    else:
        replacement = create_random_expression(rng, max_depth=max(1, min(3, max_depth - len(target_path))), force_function=False)

    mutated = replace_subtree(base_tree, target_path, replacement)
    if expression_depth(mutated) > max_depth:
        return clone_expression(node)
    return mutated



def subtree_crossover(
    parent1: ExpressionNode,
    parent2: ExpressionNode,
    rng: random.Random,
    max_depth: int,
) -> tuple[ExpressionNode, ExpressionNode]:
    """Swap random subtrees between two parent expressions."""
    left_parent = clone_expression(parent1)
    right_parent = clone_expression(parent2)

    left_path = rng.choice(all_subtree_paths(left_parent))
    right_path = rng.choice(all_subtree_paths(right_parent))
    left_subtree = clone_expression(get_subtree(left_parent, left_path))
    right_subtree = clone_expression(get_subtree(right_parent, right_path))

    left_child = replace_subtree(left_parent, left_path, right_subtree)
    right_child = replace_subtree(right_parent, right_path, left_subtree)

    if expression_depth(left_child) > max_depth:
        left_child = clone_expression(parent1)
    if expression_depth(right_child) > max_depth:
        right_child = clone_expression(parent2)
    return left_child, right_child



def calculate_fit_metrics(targets: list[float], predictions: list[float]) -> tuple[float, float, float, float]:
    """Return SSE, MSE, R², and RMSE for one set of predictions."""
    residuals = [target - prediction for target, prediction in zip(targets, predictions)]
    ss_res = sum(residual * residual for residual in residuals)
    mse = ss_res / len(targets)
    mean_voltage = sum(targets) / len(targets)
    ss_tot = sum((target - mean_voltage) ** 2 for target in targets)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    rmse = math.sqrt(mse)
    return ss_res, mse, r_squared, rmse



def evaluate_symbolic_candidate(expression: ExpressionNode, dataset: list[Observation]) -> tuple[float, float, float, float, float, float]:
    """Score one expression tree on the whole dataset."""
    raw_outputs: list[float] = []
    targets = [item.voltage for item in dataset]

    for item in dataset:
        raw_value = evaluate_expression(expression, item.glucose_concentration, item.temperature)
        if not math.isfinite(raw_value):
            return SYMBOLIC_INVALID_FITNESS, SYMBOLIC_INVALID_FITNESS, -1.0, SYMBOLIC_INVALID_FITNESS, 0.0, 0.0
        raw_outputs.append(raw_value)

    mean_output = sum(raw_outputs) / len(raw_outputs)
    mean_target = sum(targets) / len(targets)
    variance_output = sum((value - mean_output) ** 2 for value in raw_outputs)
    if variance_output < 1e-12:
        output_scale = 0.0
        output_bias = mean_target
    else:
        covariance = sum((value - mean_output) * (target - mean_target) for value, target in zip(raw_outputs, targets))
        output_scale = covariance / variance_output
        output_bias = mean_target - output_scale * mean_output

    predictions = [clamp_prediction(output_scale * value + output_bias) for value in raw_outputs]
    if any(not math.isfinite(prediction) for prediction in predictions):
        return SYMBOLIC_INVALID_FITNESS, SYMBOLIC_INVALID_FITNESS, -1.0, SYMBOLIC_INVALID_FITNESS, 0.0, 0.0

    _, mse, r_squared, rmse = calculate_fit_metrics(targets, predictions)
    complexity = expression_complexity(expression)
    fitness = mse + SYMBOLIC_COMPLEXITY_PENALTY * complexity
    return fitness, mse, r_squared, rmse, output_scale, output_bias



def tournament_select(
    population: list[ExpressionNode],
    fitness_lookup: dict[int, float],
    rng: random.Random,
) -> ExpressionNode:
    """Tournament selection for symbolic-regression individuals."""
    competitors = rng.sample(population, SYMBOLIC_TOURNAMENT_SIZE)
    winner = min(competitors, key=lambda candidate: fitness_lookup[id(candidate)])
    return clone_expression(winner)



def initialise_symbolic_population(rng: random.Random) -> list[ExpressionNode]:
    """Create a diverse starting population using ramped tree depths."""
    population: list[ExpressionNode] = []
    for index in range(SYMBOLIC_POPULATION_SIZE):
        depth = rng.randint(1, SYMBOLIC_INITIAL_MAX_DEPTH)
        force_function = index % 2 == 0
        population.append(create_random_expression(rng, max_depth=depth, force_function=force_function))
    return population



def fit_symbolic_model(dataset: list[Observation]) -> SurrogateModel:
    """Fit a symbolic-regression surrogate using genetic programming.

    Each population member is an explicit expression tree. Selection, subtree
    crossover, subtree mutation, and elitism are used to evolve equations that
    predict voltage while keeping complexity under control.
    """
    rng = random.Random(RANDOM_SEED)
    population = initialise_symbolic_population(rng)

    best_tree = clone_expression(population[0])
    best_fitness = SYMBOLIC_INVALID_FITNESS
    best_r_squared = -1.0
    best_rmse = SYMBOLIC_INVALID_FITNESS
    best_complexity = expression_complexity(best_tree)
    best_output_scale = 1.0
    best_output_bias = 0.0
    best_history: list[float] = []

    for generation in range(SYMBOLIC_GENERATIONS):
        scored_population: list[tuple[ExpressionNode, float, float, float, float, float, float]] = []
        fitness_lookup: dict[int, float] = {}

        for candidate in population:
            fitness, mse, r_squared, rmse, output_scale, output_bias = evaluate_symbolic_candidate(candidate, dataset)
            scored_population.append((candidate, fitness, mse, r_squared, rmse, output_scale, output_bias))
            fitness_lookup[id(candidate)] = fitness

        scored_population.sort(key=lambda entry: entry[1])
        generation_best, generation_fitness, _, generation_r_squared, generation_rmse, generation_scale, generation_bias = scored_population[0]
        best_history.append(generation_fitness)

        if generation_fitness < best_fitness:
            best_tree = clone_expression(generation_best)
            best_fitness = generation_fitness
            best_r_squared = generation_r_squared
            best_rmse = generation_rmse
            best_complexity = expression_complexity(generation_best)
            best_output_scale = generation_scale
            best_output_bias = generation_bias

        print(
            f'Symbolic generation {generation + 1:2d}: '
            f'best fitness = {generation_fitness:.6f}, '
            f'R^2 = {generation_r_squared:.4f}, '
            f'RMSE = {generation_rmse:.4f}, '
            f'complexity = {expression_complexity(generation_best)}'
        )

        elites = [clone_expression(entry[0]) for entry in scored_population[:SYMBOLIC_ELITE_COUNT]]
        next_population = elites[:]

        while len(next_population) < SYMBOLIC_POPULATION_SIZE:
            parent1 = tournament_select(population, fitness_lookup, rng)
            parent2 = tournament_select(population, fitness_lookup, rng)

            if rng.random() < SYMBOLIC_CROSSOVER_RATE:
                child1, child2 = subtree_crossover(parent1, parent2, rng, SYMBOLIC_MAX_DEPTH)
            else:
                child1, child2 = clone_expression(parent1), clone_expression(parent2)

            if rng.random() < SYMBOLIC_MUTATION_RATE:
                child1 = mutate_expression(child1, rng, SYMBOLIC_MAX_DEPTH)
            if rng.random() < SYMBOLIC_MUTATION_RATE:
                child2 = mutate_expression(child2, rng, SYMBOLIC_MAX_DEPTH)

            next_population.append(child1)
            if len(next_population) < SYMBOLIC_POPULATION_SIZE:
                next_population.append(child2)

        population = next_population

    return SurrogateModel(
        expression_tree=best_tree,
        best_expression=expression_to_string(best_tree),
        output_scale=best_output_scale,
        output_bias=best_output_bias,
        best_fitness=best_fitness,
        r_squared=best_r_squared,
        rmse=best_rmse,
        complexity=best_complexity,
        history=tuple(best_history),
    )



def predict_symbolic(glucose: float, temperature: float, model: SurrogateModel) -> float:
    """Predict voltage from glucose and temperature using the evolved expression tree."""
    prediction = evaluate_expression(model.expression_tree, glucose, temperature)
    prediction = clamp_prediction(model.output_scale * prediction + model.output_bias)
    if not math.isfinite(prediction):
        return 0.0
    return prediction



def predict_voltage(glucose: float, temperature: float, model: SurrogateModel) -> float:
    """Compatibility wrapper so the existing optimisation pipeline stays unchanged."""
    return predict_symbolic(glucose, temperature, model)



def print_model_summary(model: SurrogateModel, dataset: list[Observation]) -> None:
    """Print a concise summary of the evolved symbolic-regression model."""
    print('Fitted symbolic regression surrogate')
    print('=' * 78)
    print('Input aliases         : g = (glucose - 0.10) / 0.08, t = (temperature - 30.0) / 8.0')
    print(f'Discovered equation  : V ≈ ({model.output_scale:.4f} * ({model.best_expression})) + {model.output_bias:.4f}')
    print(f'Dataset rows         : {len(dataset)}')
    print(f'Best fitness         : {model.best_fitness:.6f}')
    print(f'Model R^2            : {model.r_squared:.4f}')
    print(f'Model RMSE           : {model.rmse:.4f} V')
    print(f'Expression complexity: {model.complexity}')
    print()
    print('Comparison note      : the previous polynomial surrogate fixed the')
    print('                       equation structure first and only fit the')
    print('                       coefficients. This version evolves the equation')
    print('                       tree itself, so it searches for structure and')
    print('                       constants together.')
    print('=' * 78)



def plot_predicted_vs_actual(dataset: list[Observation], model: SurrogateModel) -> None:
    """Optional scatter plot for actual vs predicted voltages."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print('Predicted-vs-actual scatter plot was skipped because matplotlib is not installed.')
        return

    actual = [item.voltage for item in dataset]
    predicted = [predict_symbolic(item.glucose_concentration, item.temperature, model) for item in dataset]

    figure, axis = plt.subplots(figsize=(6.5, 6.0))
    axis.scatter(actual, predicted, alpha=0.75, color='#1f77b4', edgecolors='black', linewidths=0.4)
    low = min(actual + predicted)
    high = max(actual + predicted)
    axis.plot([low, high], [low, high], '--', color='black', linewidth=1.2)
    axis.set_xlabel('Actual voltage (V)')
    axis.set_ylabel('Predicted voltage (V)')
    axis.set_title('Symbolic regression: predicted vs actual')
    figure.tight_layout()
    plt.show()



def plot_symbolic_history(model: SurrogateModel) -> None:
    """Optional best-fitness-per-generation plot for the symbolic GP run."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print('Symbolic fitness plot was skipped because matplotlib is not installed.')
        return

    generations = list(range(1, len(model.history) + 1))
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.plot(generations, list(model.history), color='#d62728', linewidth=2)
    axis.set_xlabel('Symbolic regression generation')
    axis.set_ylabel('Best fitness')
    axis.set_title('GA-based symbolic regression training history')
    figure.tight_layout()
    plt.show()


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


def plot_surface_3d(
    model: SurrogateModel,
    final_population: list[tuple[float, float]],
    best_solution: tuple[float, float],
) -> None:
    """Plot the fitted surrogate model as a 3D surface.

    The surface uses the already-fitted symbolic model. The final GA
    population is overlaid as scatter points so the search result can be
    compared directly with the surrogate landscape.
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ModuleNotFoundError:
        print(
            "3D surface plot was skipped because numpy and/or matplotlib are not "
            "installed in this environment."
        )
        return

    # Create a regular grid over the design space using numpy.meshgrid.
    glucose_values = np.linspace(MIN_GLUCOSE, MAX_GLUCOSE, 60)
    temperature_values = np.linspace(MIN_TEMPERATURE, MAX_TEMPERATURE, 60)
    glucose_grid, temperature_grid = np.meshgrid(glucose_values, temperature_values)

    # Use the existing prediction function to calculate the fitted voltage at
    # each point on the grid.
    voltage_grid = np.empty_like(glucose_grid)
    for row_index in range(glucose_grid.shape[0]):
        for column_index in range(glucose_grid.shape[1]):
            voltage_grid[row_index, column_index] = predict_voltage(
                float(glucose_grid[row_index, column_index]),
                float(temperature_grid[row_index, column_index]),
                model,
            )

    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")

    surface = axis.plot_surface(
        glucose_grid,
        temperature_grid,
        voltage_grid,
        cmap="viridis",
        edgecolor="none",
        alpha=0.88,
        antialiased=True,
    )

    population_x = [individual[0] for individual in final_population]
    population_y = [individual[1] for individual in final_population]
    population_z = [predict_voltage(individual[0], individual[1], model) for individual in final_population]
    best_z = predict_voltage(best_solution[0], best_solution[1], model)

    axis.scatter(
        population_x,
        population_y,
        population_z,
        c="red",
        s=28,
        depthshade=True,
        label="Final GA population",
    )
    axis.scatter(
        [best_solution[0]],
        [best_solution[1]],
        [best_z],
        c="yellow",
        edgecolors="black",
        s=120,
        marker="*",
        depthshade=False,
        label="Best solution",
    )

    axis.set_xlabel("Glucose concentration (mol dm^-3)", labelpad=12)
    axis.set_ylabel("Temperature (°C)", labelpad=12)
    axis.set_zlabel("Predicted voltage (V)", labelpad=10)
    axis.set_title("3D Surface Plot of the Fitted Symbolic Model", pad=18)
    axis.view_init(elev=28, azim=-130)
    axis.legend(loc="upper left")

    colourbar = figure.colorbar(surface, ax=axis, shrink=0.70, pad=0.10)
    colourbar.set_label("Predicted voltage (V)")

    figure.tight_layout()
    figure.savefig(SURFACE_PLOT_FILE, dpi=300)
    plt.show()


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
    model = fit_symbolic_model(dataset)
    print_model_summary(model, dataset)
    plot_predicted_vs_actual(dataset, model)
    plot_symbolic_history(model)

    history, best_solution, best_voltage = run_genetic_algorithm(model)

    print("\nFinal optimisation result")
    print("=" * 78)
    print(f"Optimal glucose concentration : {best_solution[0]:.4f} mol dm^-3")
    print(f"Optimal temperature           : {best_solution[1]:.2f} °C")
    print(f"Predicted maximum voltage     : {best_voltage:.4f} V")
    print("=" * 78)

    plot_surface_3d(model, history[-1].population, best_solution)

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

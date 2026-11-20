# Adult Household Meal Planning with a Python Genetic Algorithm

Requires Python 3.10+ and uses PyTorch for batch nutrition and fitness calculations. The algorithm generates one day of staple foods, side dishes, and fruit for a household of adults. It scores each member's nutrient intake against individual targets and optimizes plans through four-way tournament selection, crossover, mutation, and elitism. The repository's original name does not affect how this project runs.

## Running the Project

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
python3 meal_ga.py --seed 42
python3 -m unittest -v test_all
```

All tests are in `test_all.py`, covering CNF data loading, genetic operators, meal constraints, and PyTorch scoring. You can also run them directly with `python3 test_all.py -v`.

Select `.venv/bin/python` as your IDE's Python interpreter, or activate the virtual environment before running the commands above.

The default device is CPU. With a CUDA-enabled PyTorch installation and an available GPU, use `--device cuda` or `--device cuda:0`. Use `--device auto` to select CUDA when available and CPU otherwise. A CPU-only PyTorch build cannot use CUDA; see the [official PyTorch installation instructions](https://docs.pytorch.org/get-started/locally/) for GPU setup. CUDA tests are skipped when CUDA is unavailable.

By default, the program reads **CNF 2026 food and nutrient CSV data** from `dataset/`, selects 30 foods through `dataset/candidates.json`, and reads household members from `data/people.json`. It evolves a population of 100 individuals for 50 generations and writes `results.json`. Including generation 0, the output contains 51 sets of statistics and up to three distinct preferred plans collected across generations. Plans follow the UL comparison rules, so the preferred plan may differ from the plan with the highest base score alone.

```bash
python3 meal_ga.py --foods dataset --people data/people.json \
  --population 200 --generations 50 --seed 123 --output my_results.json
```

The data source is [Health Canada's CNF 2026](https://open.canada.ca/data/en/dataset/1b6139bd-ed7e-4043-bc28-ff00e10f3109). The original local food name, source, and group tables are supplemented with the official `nutrient_name.csv` and `nutrient_amount.csv` tables. Records are joined by code, and nutrient values come from the dataset rather than demonstration estimates. See [dataset/README.md](dataset/README.md) for food selection, download links, and integrity rules. The fixed daily fruit allowance of 250 g refers to edible weight. The program does not predict energy requirements.

Edit `dataset/candidates.json` to change the candidate pool, or use `--candidates my_candidates.json` to provide another list. You can also use `--foods my_foods.json` to load custom food data in the format described below.

## Resolving Conflicting or Unspecified Requirements

- The default population size follows the final workflow specification: 100 individuals with two elites. It can be changed to the earlier value of 200 through a command-line argument.
- Carbohydrate, protein, and fat targets are calculated as `E * 0.55 / 4`, `0.8 * body_weight_kg`, and `E * 0.275 / 9`, respectively. `E` is reference energy in kcal/day. The illustrative values of 280 g, 70 g, and 55 g do not override these formulas.
- The household base score is the arithmetic mean of individual scores, with a maximum of 100. Optimization retains fractional scores. Displayed component and total scores use rounding to the nearest integer, with halves rounded up. The total is rounded after summing unrounded components, so displayed component scores may not sum to the displayed total.
- The excess penalty uses an additional engineering parameter: `fitness = household_base_score - 10 * household_normalized_UL_excess / member_count`. Comparison prioritizes UL status and excess magnitude before fitness. A fitness advantage cannot place a known over-limit plan above a fully assessed plan with no excess.
- The comparison order is: fully assessed with no excess; no known excess but incomplete assessment; known excess. Among over-limit plans, compare the sum of `max(actual / UL - 1, 0)` across members and UL fields, then fitness. Unknown fields are never treated as passed checks.
- UL values are fixed project parameters and are not recalculated by age. Only adults aged 18 or older are accepted, and age is retained in the output. Real assessments for other age groups require the corresponding age-specific targets and UL rules.
- Crossover operates on whole dishes: each corresponding side-dish slot, meal staple, and fruit module independently inherits from either parent with 50% probability. Each parent pair produces two independent children. Parents come from two independent four-way tournaments. `Config.crossover_probability` defaults to 1; lowering it allows parent copying instead of crossover.
- By default, staple types change only during initialization and crossover. Their weights are derived from the formula and are not independent genes. Use `--staple-mutation 0.05` to enable an additional, independent staple-type mutation; changing the type automatically changes its derived weight. The default probability is 0 because the assignment does not specify one. Staples do not participate in the 40% side-dish/fruit ingredient-mutation events or the 30% portion-mutation events.
- A fruit transfer moves 10 g per member between meals, or `10 * member_count` g for the household. Crossover repair normalizes each member's three meal weights and projects them onto the constraints of 250 g/day and 50-150 g/meal.
- An infeasible ingredient-mutation event leaves the individual unchanged without drawing a different event. Portion mutation selects uniformly among the currently executable operation categories.
- Plan deduplication ignores dish and ingredient ordering and rounds weights to two decimal places. Distinct plans differ in structure or portions; they are not guaranteed to have substantially different menus. Diversity is the number of unique plans under this rule divided by population size.

## Household Member Input

`data/people.json` contains a JSON array, for example:

```json
[
  {"name": "Adult A", "age": 30, "reference_group": "male", "weight_kg": 75, "energy_kcal": 2400},
  {"name": "Adult B", "age": 28, "reference_group": "female", "weight_kg": 60, "energy_kcal": 2000}
]
```

`reference_group` selects the nutrient reference targets and must be `male` or `female`. Member names must be unique. Reference energy and body weight must be positive, finite numbers.

## Food Database

The default CNF directory requires `food_name.csv`, `food_source.csv`, `cnf_food_group.csv`, `nutrient_name.csv`, `nutrient_amount.csv`, and `candidates.json`. The loader dynamically resolves nutrient codes by standard `Tagname` first, then by exact English nutrient name, and validates units. Numeric nutrient codes are not hardcoded. Nutrients are read per 100 g of edible food. CNF `Food_Code` values become food IDs, and the output retains original descriptions, preparation states, and sources. Vitamin E UL assessment remains incomplete when applicable sources cannot be identified.

Custom JSON input is also supported, with the top-level structure `{"metadata": {...}, "foods": [...]}`. Each database must contain 25-40 foods with unique IDs, exactly three staples representing cooked rice, bread, and cooked egg noodles, and at least two fruits. Each entry in `foods` uses the following fields:

| Field | Meaning |
| --- | --- |
| `id`, `name` | Unique identifier and display name |
| `category` | `vegetable`: non-starchy vegetables; `starch`: starchy foods; `protein`: protein foods; `fruit`: fruit; `other`: other foods |
| `role` | `staple`: staple food; `fruit`: fruit; `mixed`: suitable for mixing; `standalone`: served separately |
| `mix_group` | Mixing compatibility group, defaulting to `savory`; different groups cannot be mixed, and each group needs at least four foods |
| `per100g` | Nutrient values per 100 g of edible food |

Same-category replacement also requires compatible roles. Foods marked `standalone`, such as milk and yogurt, are always served separately. Categories do not restrict nutrient calculations: every nutrient field from every consumed food contributes to intake.

| `per100g` key | Unit and purpose |
| --- | --- |
| `energy_kcal` | kcal; reported actual energy, excluded from the 100-point score |
| `carbohydrate_g`, `protein_g`, `fat_g` | g |
| `vitamin_a_rae_ug` | Micrograms RAE; vitamin A adequacy scoring |
| `vitamin_c_mg` | mg |
| `vitamin_d_ug` | Micrograms |
| `vitamin_e_mg` | mg alpha-tocopherol; adequacy scoring |
| `folate_dfe_ug` | Micrograms DFE; folate adequacy scoring |
| `calcium_mg`, `iron_mg`, `potassium_mg` | mg |
| `preformed_a_ug` | Micrograms of preformed vitamin A; UL checks only |
| `ul_applicable_e_mg` | mg of vitamin E in forms and sources covered by the assignment's UL; UL checks only |
| `synthetic_folic_acid_ug` | Micrograms of synthetic folic acid; UL checks only |

The last three UL-specific fields may be `null` or omitted, indicating an incomplete assessment. Missing values in other fields cause an error. Use 0 only for a known zero. If a member consumes a food with an unknown UL field, the corresponding daily total is `null`; partial data are not presented as complete. The check normally receives `not_assessed`. If the known intake lower bound already exceeds the UL, the check receives `exceeded`, uses that lower bound to calculate excess, and retains `fully_assessed: false`. RAE cannot substitute for preformed vitamin A, DFE cannot substitute for synthetic folic acid, and total vitamin E cannot substitute for vitamin E from UL-applicable sources.

## Encoding and Algorithm

Genetic operations are separated by responsibility:

| File | Responsibility and entry points |
| --- | --- |
| `genetic_operators/survivor_selection.py` | `rank_population()` orders plans by UL priority; `select_survivors()` deep-copies elites |
| `genetic_operators/parent_selection.py` | `select_parent()` runs one tournament; `select_parents()` runs two independent tournaments |
| `genetic_operators/crossover.py` | `crossover()` inherits corresponding dishes, staples, and fruit modules, then repairs fruit totals |
| `genetic_operators/mutation.py` | `mutate_kind()` changes ingredients; `mutate_portions()` changes portions; `mutate_staple()` optionally changes staple types |
| `genetic_operators/evolution.py` | `evolve()` manages generations, the plan archive, and statistics, calling the genetic operators directly |
| `ga_models.py` | Shared food, member, individual, configuration, evaluation, and nutrient definitions |
| `device_utils.py` | `resolve_device()` parses CPU/CUDA/auto settings and checks CUDA availability and GPU indices |
| `meal_ga.py` | Data loading, initialization, validation, PyTorch batch scoring, the execution entry point, and output |

Selection modules receive the population, evaluations, and random number generator. Crossover and mutation receive a `MealGA` context to access compatibility rules, meal weight bounds, and repair functions. All operators share the instance's `rng`. Modify the corresponding module to extend a strategy. `MealGA.run()` is the common entry point and calls `evolution.evolve(ga)`. Selection, crossover, and mutation are called directly by the evolution module rather than through forwarding methods on `MealGA`.

To execute operators individually:

```python
from genetic_operators import parent_selection, mutation
from genetic_operators.crossover import crossover

a, b = parent_selection.select_parents(population, evaluations, ga.rng)
child = crossover(ga, a, b)
mutation.mutate_kind(ga, child)
mutation.mutate_portions(ga, child)
ga.validate(child)
```

Existing data-type imports such as `from meal_ga import Config, Individual` remain supported.

Each generation is encoded as an edible-weight tensor with shape `[population_size, member_count, food_count]`. Multiplication by the per-100-g nutrient matrix, shaped `[food_count, nutrient_field_count]`, produces daily nutrient totals for every member in the generation. Target ratios, scores, deficits, normalized UL excess, and fitness are calculated in PyTorch batches using `torch.float64` and `torch.inference_mode()`, without constructing an autograd graph. See the [PyTorch matrix multiplication documentation](https://docs.pytorch.org/docs/stable/generated/torch.matmul) for dimension rules.

Missing UL fields use a separate mask, propagating unknown status only from foods actually consumed. Placeholder zeros in the numeric matrix are used solely to calculate known intake lower bounds; they do not indicate passed checks. Single-individual `evaluate()` and batch `evaluate_population()` share the same scoring logic. Actual weights are encoded again each generation, without reusing parent nutrition results.

Individuals retain the variable-length dish encoding described below. Selection, crossover, mutation, and feasibility repair run in Python. Random choices use an independent `random.Random(seed)` instance; tensor scoring itself uses no randomness. Runs are reproducible with the same inputs, seed, PyTorch version, and device. Floating-point rounding may produce different evolutionary trajectories across devices. CPU/GPU performance depends on population and household size; small workloads are not guaranteed to run faster on a GPU.

`Dish` stores food IDs, integer proportion units representing tenths, and the household dish weight. `Fruit` stores food IDs, proportions, and each member's meal weight. `Meal` contains side dishes, a staple ID, and fruit. `Individual` contains three meals.

Breakfast, lunch, and dinner have `n`, `n+1`, and `n` side-dish slots, respectively. Household side-dish budgets are `150/250/200 * sum(E/2000)` g for the three meals. Dividing each budget by its slot count gives the baseline weight `B`. Initial weights range from `0.8B` to `1.2B`, with mutation bounds of `0.5B` to `1.5B`. Mixed dishes contain 1-4 unique ingredients, sampled by rounding a truncated normal draw from `N(2.5, 1)`; standalone items contain one ingredient. Shared dishes are divided among members in proportion to `E/2000`, and all nutrients are calculated separately from each member's actual allocated weights.

Each non-elite child draws one mutually exclusive ingredient-mutation event: 25% same-category replacement, 5% copying an ingredient ID from another compatible module, 5% deletion, 5% addition, and 60% no change. Proportions use increments of 10 percentage points, with a minimum of 10% for side-dish ingredients. Two-fruit modules require 20%-80% per fruit, and adding a second fruit assigns it 20%. Deletion and addition preserve the module's total weight.

Portion mutation is triggered independently with 30% probability and selects from executable changes to side-dish proportions, side-dish total weights, fruit proportions, or fruit allocation between meals. Fruit proportion changes transfer 10 or 20 percentage points, with equal probability when both are feasible. Every child is validated and its nutrition is recalculated. Elites are deep-copied and do not undergo crossover or mutation.

## Output

`results.json` contains:

- `config`, `seed`, `members`, and source data metadata.
- `backend`: the PyTorch version, actual device, and floating-point precision.
- `history`: generation 0 through 50 by default, including the preferred plan's base score and fitness, the generation's highest base score, mean score, known-excess fraction, incomplete-assessment fraction among plans without known excess, and diversity. The preferred base score follows UL ordering and may differ from the highest base score.
- `plans`: up to three distinct plans, with household dish weights, individual ingredient weights/proportions, and per-member staple and fruit weights for each meal.
- Per-member `portions` list actual food weights by meal and ingredient; `totals` include actual energy and all nutrient totals; `targets`, `scores`, and `deficits` contain targets, scores, and shortfalls; `ul_checks` contain intake, limits, normalized excess, and assessment status.

A score of 100 means only that the nutrients selected by this model reach their targets. The model does not penalize energy, carbohydrate, or fat intake above target and does not calculate sodium. The score is not a complete assessment of diet quality or safety.

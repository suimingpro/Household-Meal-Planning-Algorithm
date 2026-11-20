# Adult Household Three-Meal Ingredient Planning: Python Genetic Algorithm

Python 3.10+, supporting NumPy CPU and the original PyTorch batch-scoring backend. Generates one day of staples, side-dish combinations, and fruit for several adults, scores them against individual nutrient targets, and optimizes through four-way tournaments, crossover, mutation, and elitism. The original repository name does not affect this project's execution.

## Running

See the [project README](../README.md) for installation and execution details. NumPy CPU can be used directly by default. If PyTorch is installed, specify `--backend torch`, and specify `--device cuda` in an environment that supports CUDA. The archived experiments for Sections 4 and 5 explicitly use NumPy CPU.

By default, reads the **CNF 2026 real food and nutrient CSVs** in `dataset/`, selects 30 foods through `dataset/candidates.json`, and still reads members from `data/people.json`. Runs 100 individuals for 50 evolutionary updates and generates `results.json`. Including generation 0, there are 51 statistics records, and up to three distinct preferred plans across generations are returned. Results prioritize UL rules and may not be the plans with the highest base scores alone.

```bash
python3 meal_ga.py --foods dataset --people data/people.json \
  --population 200 --generations 50 --seed 123 --output my_results.json
```

The data source is [Health Canada's CNF 2026](https://open.canada.ca/data/en/dataset/1b6139bd-ed7e-4043-bc28-ff00e10f3109). The original local name, source, and group tables have been supplemented with the official `nutrient_name.csv` and `nutrient_amount.csv`, joined by code; nutrient values no longer use demonstration estimates. See [dataset/README.md](dataset/README.md) for record selection, download links, and integrity rules. The fixed 250 g of fruit refers to edible weight. The script does not predict energy requirements itself.

Edit `dataset/candidates.json` to change the candidate pool, or use `--candidates my_candidates.json` to specify another list. Custom JSON food data in the format below can also be used through `--foods my_foods.json`.

## Handling Conflicts and Undefined Parts in the Requirements

- The population size uses 100 from the final execution workflow, retaining two elites; a parameter can change it to the earlier value of 200.
- Carbohydrate, protein, and fat are calculated strictly as `E×0.55/4`, `0.8×body weight`, and `E×0.275/9`, respectively. E is reference energy in kcal/day, not 280 g of carbohydrate; 280 g, 70 g, and 55 g do not override the formulas.
- The household base score is the arithmetic mean of member base scores, with a maximum of 100. Each score retains decimal places during optimization; output displays components and totals rounded to integers using ordinary rounding with halves rounded up. The total is rounded after summing unrounded components, so the displayed integer components may not sum to the displayed total.
- The excess penalty uses an additional engineering parameter: `fitness = household base score − 10 × sum of household normalized excess ratios / member count`. Comparison uses UL status and excess level first, then fitness; the penalty cannot let a clearly over-limit plan outrank a fully checked plan with no excess.
- The sorting order is: fully checked with no excess, no known excess but incomplete checks, and known excess. Among over-limit plans, first compare the sum of `max(actual/UL−1, 0)` for all members and UL fields, then fitness. Fields that cannot be checked are not considered passed.
- ULs use the fixed model limits given in the assignment and are not recalculated by age; only adults aged 18 or older are accepted. Age is retained in the results. For real nutrient assessments with other age groupings, replace these with the corresponding age-specific targets and UL rules.
- Uses “whole-combination crossover”: each corresponding side-dish slot, each meal's staple, and fruit module independently inherits from one parent with 50% probability. Each parent pair generates two independent children; parents come from two independent four-way tournaments. `Config.crossover_probability` defaults to 1 and can be adjusted for copying probability.
- By default, staple types change only during initialization and crossover; portions are immediately derived from the formula, without independent weight genes. Use `--staple-mutation 0.05` to enable additional, independent staple-type mutation; weights are automatically recalculated when the type changes. The default is 0 because the assignment does not give this event's probability. Staples do not participate in the 40% side-dish type events or 30% portion events.
- Fruit transfer between meals moves 10 g for each member, totaling `10×member count` g for the household. Crossover repair normalizes each person's three-meal portions and projects them onto the constraint set of 250 g per day and 50–150 g per meal.
- An ingredient-type mutation that cannot be executed does not change the individual or draw another event. Portion mutation draws uniformly from the currently executable operation categories.
- Plan deduplication ignores combination and ingredient ordering and retains weights to two decimal places; “distinct” means different structures or portions and does not guarantee substantial differences between the three menus. Diversity is the number of unique plans under this rule divided by population size.

## Member Input

`data/people.json` is a JSON array, for example:

```json
[
  {"name": "Adult A", "age": 30, "reference_group": "male", "weight_kg": 75, "energy_kcal": 2400},
  {"name": "Adult B", "age": 28, "reference_group": "female", "weight_kg": 60, "energy_kcal": 2000}
]
```

`reference_group` is the nutrient reference group used for targets, taking `male` or `female`; member names must be unique. Energy and body weight must be positive finite numbers.

## Food Database

The default CNF directory requires `food_name.csv`, `food_source.csv`, `cnf_food_group.csv`, `nutrient_name.csv`, `nutrient_amount.csv`, and `candidates.json`. The program dynamically resolves nutrient codes by standard `Tagname` first, then by exact English name, and validates units; numeric nutrient codes are not fixed. Data are read per 100 g of edible food, with CNF `Food_Code` used as the food ID, and the output retains original names, raw/cooked states, and sources. When the source to which the vitamin E UL applies cannot be identified, it remains marked as unassessed.

Custom JSON is also supported, with the top-level structure `{"metadata": {...}, "foods": [...]}`. Each database must have 25–40 foods with different IDs, exactly three staples (cooked rice, bread, and cooked egg noodles entered separately), and at least two fruits. Each entry in `foods` has the structure below.

| Field | Meaning |
| --- | --- |
| `id`, `name` | Unique code, display name |
| `category` | `vegetable`: non-starchy vegetables; `starch`: starchy foods; `protein`: protein foods; `fruit`: fruit; `other`: other |
| `role` | `staple`: staple; `fruit`: fruit; `mixed`: mixing allowed; `standalone`: served separately |
| `mix_group` | Allowed mixing compatibility group, default `savory`; different groups are not mixed, and each group has at least four foods |
| `per100g` | Nutrient values per 100 g of edible food |

Same-category replacement also requires compatible roles; `standalone` items such as milk and yogurt are always served separately. Classification does not determine nutrient calculation: all fields of each food participate in accumulation.

| `per100g` key | Unit/purpose |
| --- | --- |
| `energy_kcal` | kcal, displayed actual energy, not included in the 100-point score |
| `carbohydrate_g`, `protein_g`, `fat_g` | g |
| `vitamin_a_rae_ug` | μg RAE, vitamin A adequacy scoring |
| `vitamin_c_mg` | mg |
| `vitamin_d_ug` | μg |
| `vitamin_e_mg` | mg α-tocopherol, adequacy scoring |
| `folate_dfe_ug` | μg DFE, folate adequacy scoring |
| `calcium_mg`, `iron_mg`, `potassium_mg` | mg |
| `preformed_a_ug` | μg preformed vitamin A, UL only |
| `ul_applicable_e_mg` | Vitamin E in the forms and sources covered by the assignment's UL, mg, UL only |
| `synthetic_folic_acid_ug` | μg synthetic folic acid, UL only |

The last three UL-specific fields may be `null` or omitted, indicating that they could not be assessed; missing other fields causes an error. Only true zeros should be entered as 0. When a person eats any food with an unknown UL field, that field's daily total is `null`; partial data are not presented as complete. It is normally marked `not_assessed`; if the known intake lower bound already exceeds the UL, it is marked `exceeded`, using the known lower bound to calculate excess while retaining `fully_assessed: false`. Vitamin A RAE cannot replace preformed A, DFE cannot replace synthetic folic acid, and total vitamin E cannot replace E from sources covered by the UL.

## Encoding and Algorithm

Genetic operations are split into separate modules by responsibility:

| File | Responsibility and entry points |
| --- | --- |
| `genetic_operators/survivor_selection.py` | `rank_population()` sorts by UL-priority rules; `select_survivors()` retains elites by deep copy |
| `genetic_operators/parent_selection.py` | `select_parent()` selects one parent by tournament; `select_parents()` performs two independent tournaments |
| `genetic_operators/crossover.py` | `crossover()` inherits combinations, staples, and fruit by corresponding slots, and repairs fruit totals |
| `genetic_operators/mutation.py` | `mutate_kind()` for ingredient-type mutation, `mutate_portions()` for portion mutation, `mutate_staple()` for optional staple mutation |
| `genetic_operators/evolution.py` | `evolve()` organizes the generation loop, plan archive, and statistics, directly calling each genetic-operator module |
| `ga_models.py` | Shared foods, members, individuals, configuration, evaluation results, and nutrient constants |
| `device_utils.py` | `resolve_device()` resolves CPU/CUDA/auto devices and checks CUDA availability and GPU indices |
| `meal_ga.py` | Data loading, initialization, validation, PyTorch batch scoring, execution entry point, and result output |

Selection modules receive the population, evaluations, and random number generator; crossover and mutation modules receive the `MealGA` context to access compatibility rules, meal weight bounds, and repair functions. All operators share the instance's `rng`. To extend a strategy, modify its corresponding module. `MealGA.run()` is the shared entry point and calls `evolution.evolve(ga)`; `MealGA` no longer retains forwarding methods for selection, crossover, and mutation, and the evolution module calls the operators directly.

To execute operators separately, call them directly:

```python
from genetic_operators import parent_selection, mutation
from genetic_operators.crossover import crossover

a, b = parent_selection.select_parents(population, evaluations, ga.rng)
child = crossover(ga, a, b)
mutation.mutate_kind(ga, child)
mutation.mutate_portions(ga, child)
ga.validate(child)
```

Existing data-type imports such as `from meal_ga import Config, Individual` remain compatible.

Each generation converts all individuals into an actual-gram tensor shaped `[population size, member count, food count]`, multiplied by a per-100-g nutrient matrix shaped `[food count, nutrient field count]` to obtain nutrient totals for each person in the entire generation at once. Target attainment ratios, scores, deficits, normalized UL excess, and fitness are all calculated in float64 batches. The PyTorch path uses `torch.inference_mode()` without constructing a gradient graph. See the [PyTorch matrix multiplication documentation](https://docs.pytorch.org/docs/stable/generated/torch.matmul) for matrix dimension rules.

Missing UL fields use an independent mask, propagating unknown status only from foods actually eaten; placeholder zeros in the numeric matrix are used only to accumulate known intake lower bounds and do not mean the field has passed its check. Single-individual `evaluate()` and batch `evaluate_population()` share the same calculation logic. Current actual weights are re-encoded each generation without reusing parent nutrient results.

Individuals still use the variable-length combination encoding below, with selection, crossover, mutation, and validity repair performed in Python. Random selection uses the instance's independent `random.Random(seed)`; tensor scoring itself does not use randomness. Runs are reproducible with the same inputs, seed, backend version, and device; floating-point rounding may lead to different evolutionary trajectories on different devices. CPU/GPU performance depends on population and household size; small calculations are not guaranteed to be faster on a GPU.

`Dish` stores food IDs, integer proportion units in tenths, and household combination weight in grams; `Fruit` stores food IDs, proportions, and each member's meal weight; `Meal` stores side-dish combinations, a staple ID, and fruit; `Individual` contains three meals.

Breakfast/lunch/dinner slot counts are `n / n+1 / n`. Household side-dish budgets for the meals are `150/250/200 × Σ(E/2000)` g; dividing by slot count gives B. Initial weight ranges are 0.8B–1.2B, with mutation bounds of 0.5B–1.5B. The mixed ingredient count is sampled from a truncated normal `N(2.5, 1)` and rounded; 1–4 distinct ingredients are used, while standalone items always use one. Combinations are allocated in proportion to each member's `E/2000`, and nutrients are accumulated separately for each person's actual grams.

Ingredient-type mutation draws only one event per non-elite child: 25% same-category replacement, 5% copying a code from another compatible module, 5% deletion, 5% addition, and 60% no change. Proportions use 10-percentage-point units, with a minimum of 10% for side-dish ingredients; two fruits each occupy 20%–80%, and an added second fruit uses 20%. Deletion and addition preserve the module's total weight. Olive oil (CNF code 422) has `max_percent` set to 50 in `dataset/candidates.json`, meaning it can occupy at most 50% of the edible weight of a single side dish and cannot form a dish alone. Initialization, replacement, proportion redistribution after deletion, and portion mutation all limit proportions according to this field; whole-dish crossover inherits existing valid proportions.

Portion mutation is independently triggered with 30% probability, choosing among four executable operations: side-dish proportions, side-dish total weight, fruit proportions, and fruit transfers between meals. Fruit proportion changes transfer 10 or 20 percentage points, each with 50% probability when both are feasible. All children are validated before nutrients are recalculated, without reusing parent caches. Elites are deep-copied and do not undergo crossover or mutation.

## Output

`results.json` contains:

- `config`, `seed`, `members`, and original data metadata.
- `backend`: the actual scoring library, version, computing device, and floating-point precision.
- `history`: best-plan base score, best fitness, current-generation maximum base score, mean score, known-excess fraction, fraction with no known excess but incomplete checks, and diversity for generations 0–50. The best base score is affected by UL ordering and may differ from the maximum base score.
- `plans`: up to three distinct plans, including household combination grams for three meals, each ingredient's grams/proportions, and each person's staple and fruit grams.
- Each person's `portions` lists actual grams by meal and food; `totals` contains actual energy and all nutrient totals; `targets`, `scores`, and `deficits` are targets, scores, and shortfalls; `ul_checks` lists actual amounts, upper limits, normalized excess, and check status.

100 points only means that the nutrient indicators selected by this model reach their targets. The model does not deduct points for energy, carbohydrate, or fat above target and does not calculate sodium; it also does not represent a complete conclusion about dietary quality or safety.

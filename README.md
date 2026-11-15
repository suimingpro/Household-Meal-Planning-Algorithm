# Evolutionary Algorithm for Household Meal Planning

**CISC455/851 group project** · Python prototype and reproducible evaluation

The project builds a one-day breakfast, lunch and dinner plan for three sample adults using 30 foods from the Canadian Nutrient File (CNF). It searches over ingredients, portions and combinations to improve a capped nutrient-adequacy score subject to meal-structure constraints. The repository name in the original ZIP referred to doctor matching, but the supplied prototype implements meal planning.

**Start with [the Sections 4–5 report](docs/Sections_4_and_5.pdf)** ([editable Markdown](docs/Sections_4_and_5.md)). It presents the experimental design, convergence and outcome figures, and a reflection on what the results do and do not establish.

## Quick start

Use Python 3.10 or newer. From the extracted project directory, run:

```bash
python -m pip install -r requirements.txt
python meal_ga.py --backend numpy --seed 42 --output my_plan.json
python -m unittest -v
```

NumPy handles CPU scoring and Matplotlib produces the figures. The included CNF CSVs are used offline. The optional original PyTorch backend can still be used. Install `requirements-torch.txt` with a PyTorch build for your platform, then run `python meal_ga.py --backend torch --device cpu --seed 42`. Use `--device cuda` only with a CUDA-capable PyTorch installation. The prototype's `auto` backend prefers PyTorch when available, while the experiments explicitly default to NumPy CPU.

## Reproduce Sections 4 and 5

```bash
python -m experiments.run_experiments --quick
python -m experiments.run_experiments --output reproduced_results
```

The quick run only checks that the pipeline works. The full command repeats the six conditions across ten fixed seeds, with 100 individuals and 50 evolutionary updates (51 scored batches, including initialization). The checked-in results are under [experiments/results/](experiments/results/). Use `--output` to keep the checked-in study intact. Timing is machine-dependent. Very small floating-point differences across backends may change the ordering of tied plans.

| File or directory | Contents |
| --- | --- |
| `meal_ga.py`, `ga_models.py`, `genetic_operators/` | Original meal representation, constraints and EA operators, with CPU scoring and extra experimental telemetry |
| `data/people.json`, `dataset/`, `cnf_data.py` | Sample household, bundled CNF data, candidate pool and loader |
| `experiments/run_experiments.py` | Fixed-budget random baseline, two operator ablations and a candidate-pool stress test |
| `experiments/results/` | Per-run and per-batch data, nutrient outcomes, paired comparisons, figures, input manifest and a detailed example plan |
| `results.json` | Standalone prototype example generated on NumPy CPU with seed 42 |
| `experiments/reference/` | Original PyTorch/CUDA example from the input archive, retained separately from the new results |
| `test_all.py`, `test_experiments.py` | Prototype tests, scalar scoring checks and experimental protocol checks |
| `docs/Prototype_Design_ZH.md` | Original prototype design notes, with current setup information |

A score of 100 means the modeled adequacy targets were met. It does not establish dietary safety, calorie balance or suitability for a real household. See [dataset provenance](dataset/README.md) and the report for details.

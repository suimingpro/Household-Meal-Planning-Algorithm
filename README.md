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

默认CNF目录需要 `food_name.csv`、`food_source.csv`、`cnf_food_group.csv`、`nutrient_name.csv`、`nutrient_amount.csv` 及 `candidates.json`。程序优先按标准 `Tagname`、其次按精确英文名称动态解析营养素编号，并验证单位；不固定营养素数字编号。按每100g可食部分读取数据，使用CNF的 `Food_Code` 作为食材ID，并在输出中保留原始名称、生熟状态和来源。维生素E的UL适用来源无法识别时仍标记未能评估。

也支持自定义JSON：顶层为 `{"metadata": {...}, "foods": [...]}`。每份数据库必须有25–40个不同ID的食材，恰好3种主食（分别录入熟米饭、面包、熟鸡蛋面），至少2种水果。`foods`中每个条目的结构如下表所述。

| 字段 | 含义 |
| --- | --- |
| `id`, `name` | 唯一编号、展示名称 |
| `category` | `vegetable`非淀粉蔬菜、`starch`淀粉类、`protein`蛋白质食物、`fruit`水果、`other`其他 |
| `role` | `staple`主食、`fruit`水果、`mixed`允许混合、`standalone`单独供应 |
| `mix_group` | 允许混合的兼容组，默认`savory`；不同组不混合，每组至少4种食材 |
| `max_percent` | 在非主食组合中的可食重量比例上限，默认100，按10个百分点设置；橄榄油设为50 |
| `per100g` | 每100g可食部分的营养值 |

同类替换同时要求角色兼容；牛奶、酸奶等 `standalone` 项始终单独供应。分类不决定营养计算：每种食材的所有字段都参与累计。

橄榄油在 `dataset/candidates.json` 中配置 `"max_percent": 50`，不能独立组成100%橄榄油的组合。初始化、替换、删除后的比例再分配和分量突变均遵守上限；交叉继承整个合法组合。此限制按食材在组合中的重量比例计算，不是每日油量上限或组合的总脂肪比例。其余食材默认仍可占100%。

| `per100g`键 | 单位/用途 |
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

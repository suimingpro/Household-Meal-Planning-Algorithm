# 成人家庭三餐食材规划：Python 遗传算法

Python 3.10+，使用PyTorch批量计算营养和适应度。为若干名成人生成一天的主食、非主食组合和水果，按个人营养目标评分，以四元锦标赛、交叉、变异和精英保留优化。仓库原有名称不影响本项目运行。

## 运行

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
python3 meal_ga.py --seed 42
python3 -m unittest -v test_all
```

全部测试集中在 `test_all.py`，覆盖CNF数据加载、遗传算子、三餐约束和PyTorch评分；也可直接运行 `python3 test_all.py -v`。

本次已在项目的 `.venv` 中安装CPU版PyTorch。IDE请将Python解释器设为 `.venv/bin/python`，或先激活上述环境。

默认使用CPU。若已有CUDA版PyTorch及可用GPU，可用 `--device cuda` 或 `--device cuda:0`；`--device auto` 自动选择可用CUDA，否则使用CPU。CPU版PyTorch无法使用CUDA，GPU环境的安装方式见 [PyTorch官方安装说明](https://docs.pytorch.org/get-started/locally/)。本机验证使用CPU，CUDA测试在无可用GPU时跳过。

默认读取 `dataset/` 中的 **CNF 2026真实食材与营养CSV**，由 `dataset/candidates.json` 选取30种食材；成员仍读取 `data/people.json`。运行100个个体、50次进化，生成 `results.json`。包含第0代在内共51条统计，返回历代最多3个不同的优选方案。结果按UL规则优先，未必是单看基础分数最高的方案。

```bash
python3 meal_ga.py --foods dataset --people data/people.json \
  --population 200 --generations 50 --seed 123 --output my_results.json
```

数据来源为 [Health Canada的CNF 2026](https://open.canada.ca/data/en/dataset/1b6139bd-ed7e-4043-bc28-ff00e10f3109)。本地原有的名称、来源、分类表已补齐官方 `nutrient_name.csv` 和 `nutrient_amount.csv`，按编号连接，营养值不再使用演示估计。条目选择、下载地址与完整性规则见 [dataset/README.md](dataset/README.md)。固定250g水果计可食重量。脚本不会自行预测能量需求。

可编辑 `dataset/candidates.json` 改变候选池，或使用 `--candidates my_candidates.json` 指定另一份清单。也可通过 `--foods my_foods.json` 使用下述格式的自定义JSON食材数据。

## 要求中冲突与未定义部分的处理

- 种群规模采用末尾运行流程的100，保留2个精英；支持参数改为前文的200。
- 碳水、蛋白质、脂肪分别严格按 `E×0.55/4`、`0.8×体重`、`E×0.275/9` 计算。E是kcal/天的参考能量，不是280g碳水；280g、70g、55g不覆盖公式。
- 家庭基础分为成员基础分的算术平均，最高100。各项得分在优化时保留小数；输出以通常四舍五入到整数展示各项和总分。总分由未取整项相加后取整，因此展示的分项整数相加可能不同于展示总分。
- 超限扣分采用补充的工程参数：`fitness = 家庭基础分 − 10 × 家庭归一化超限比例总和 / 人数`。比较时先用UL状态和超限程度，再用适应度；扣分无法让明确超限者超过已完整检查且无超限者。
- 排序顺序为：完整检查且无超限、无已知超限但检查不完整、有已知超限。同为超限者先比较各成员各UL的 `max(实际/UL−1, 0)` 之和，再比较适应度。无法检查的字段不会被认定为通过。
- UL使用题目给出的固定模型上限，不按年龄重算；只接受18岁及以上成人。年龄会保留在结果中。若用于其他年龄分层的真实营养评估，应替换为对应年龄的目标及UL规则。
- 采用“整道组合交叉”：每个对应非主食槽位、每餐主食和水果模块各自以50%概率继承一方。每对父代生成两个独立子代，父代来自两次独立的四元锦标赛。`Config.crossover_probability` 默认1，可调为复制概率。
- 默认主食仅初始化及交叉时改变种类，分量从公式即时派生，没有独立重量基因。可用 `--staple-mutation 0.05` 开启额外、独立的主食种类变异；种类改变时重量自动一起重算。默认0，因为题目未给出该事件概率。主食不参加非主食40%种类事件或30%分量事件。
- 水果转餐操作每名成员转移10g，家庭共转移 `10×人数` g。交叉修复逐人将三餐份量归一化并投影到每日250g、每餐50–150g的约束集合。
- 不能执行的种类变异不改变个体，不重新抽另一个事件。分量变异从当前可执行的操作类别中均匀抽取。
- 方案去重忽略组合和食材排列顺序，克数保留两位小数；“不同”表示结构或份量不同，不保证三套菜单差异显著。多样性是该规则下的唯一方案数/种群数。

## 输入成员

`data/people.json` 为JSON数组，例如：

```json
[
  {"name": "成人A", "age": 30, "reference_group": "male", "weight_kg": 75, "energy_kcal": 2400},
  {"name": "成人B", "age": 28, "reference_group": "female", "weight_kg": 60, "energy_kcal": 2000}
]
```

`reference_group` 是目标值采用的营养参考分组，取 `male` 或 `female`；成员名称需唯一。能量与体重必须为正有限数。

## 食材数据库

默认CNF目录需要 `food_name.csv`、`food_source.csv`、`cnf_food_group.csv`、`nutrient_name.csv`、`nutrient_amount.csv` 及 `candidates.json`。程序优先按标准 `Tagname`、其次按精确英文名称动态解析营养素编号，并验证单位；不固定营养素数字编号。按每100g可食部分读取数据，使用CNF的 `Food_Code` 作为食材ID，并在输出中保留原始名称、生熟状态和来源。维生素E的UL适用来源无法识别时仍标记未能评估。

也支持自定义JSON：顶层为 `{"metadata": {...}, "foods": [...]}`。每份数据库必须有25–40个不同ID的食材，恰好3种主食（分别录入熟米饭、面包、熟鸡蛋面），至少2种水果。`foods`中每个条目的结构如下表所述。

| 字段 | 含义 |
| --- | --- |
| `id`, `name` | 唯一编号、展示名称 |
| `category` | `vegetable`非淀粉蔬菜、`starch`淀粉类、`protein`蛋白质食物、`fruit`水果、`other`其他 |
| `role` | `staple`主食、`fruit`水果、`mixed`允许混合、`standalone`单独供应 |
| `mix_group` | 允许混合的兼容组，默认`savory`；不同组不混合，每组至少4种食材 |
| `per100g` | 每100g可食部分的营养值 |

同类替换同时要求角色兼容；牛奶、酸奶等 `standalone` 项始终单独供应。分类不决定营养计算：每种食材的所有字段都参与累计。

| `per100g`键 | 单位/用途 |
| --- | --- |
| `energy_kcal` | kcal，实际能量展示，不参与100分评分 |
| `carbohydrate_g`, `protein_g`, `fat_g` | g |
| `vitamin_a_rae_ug` | μg RAE，维生素A达标评分 |
| `vitamin_c_mg` | mg |
| `vitamin_d_ug` | μg |
| `vitamin_e_mg` | mg α-生育酚，达标评分 |
| `folate_dfe_ug` | μg DFE，叶酸达标评分 |
| `calcium_mg`, `iron_mg`, `potassium_mg` | mg |
| `preformed_a_ug` | μg预成型维生素A，仅用于UL |
| `ul_applicable_e_mg` | 符合题目UL适用形式与来源的维生素E，mg，仅用于UL |
| `synthetic_folic_acid_ug` | μg合成叶酸，仅用于UL |

最后三个UL专用字段允许 `null` 或缺省，表示未能评估；其余字段缺失会报错。真实的零才应填写0。某个人吃到任何UL字段未知的食物时，该项全天总量为 `null`，不会用部分数据假装完整。通常标记 `not_assessed`；如果已知部分的摄入下界已经超UL，则标记 `exceeded`，以已知下界计算超限程度，同时保留 `fully_assessed: false`。A的RAE不能替代预成型A，DFE不能替代合成叶酸；总维生素E也不能替代UL适用来源的E。

## 编码与算法

遗传操作按职责拆为独立模块：

| 文件 | 职责与入口 |
| --- | --- |
| `genetic_operators/survivor_selection.py` | `rank_population()`按UL优先规则排序；`select_survivors()`深拷贝保留精英 |
| `genetic_operators/parent_selection.py` | `select_parent()`锦标赛选一个父代；`select_parents()`执行两次独立锦标赛 |
| `genetic_operators/crossover.py` | `crossover()`按对应槽位继承组合、主食和水果，并修复水果总量 |
| `genetic_operators/mutation.py` | `mutate_kind()`种类突变、`mutate_portions()`分量突变、`mutate_staple()`可选主食突变 |
| `genetic_operators/evolution.py` | `evolve()`组织逐代循环、方案归档和统计，直接调用各遗传操作模块 |
| `ga_models.py` | 共享的食材、成员、个体、配置、评分结果和营养常量 |
| `device_utils.py` | `resolve_device()`解析CPU/CUDA/auto设备，检查CUDA可用性和GPU编号 |
| `meal_ga.py` | 数据加载、初始化、校验、PyTorch批量评分、运行入口和结果输出 |

选择模块接收种群、评分结果及随机数生成器；交叉和突变模块接收 `MealGA` 上下文以访问兼容规则、餐次重量边界及修复函数。所有算子共用实例的 `rng`。扩展相应策略时，在对应模块修改实现即可。`MealGA.run()`是统一入口，调用 `evolution.evolve(ga)`；`MealGA` 不再保存选择、交叉和突变的转发方法，进化模块直接调用对应算子。

如需单独执行算子，可直接调用：

```python
from genetic_operators import parent_selection, mutation
from genetic_operators.crossover import crossover

a, b = parent_selection.select_parents(population, evaluations, ga.rng)
child = crossover(ga, a, b)
mutation.mutate_kind(ga, child)
mutation.mutate_portions(ga, child)
ga.validate(child)
```

原有 `from meal_ga import Config, Individual` 等数据类型导入继续兼容。

每代把全部个体转换成 `[种群大小, 成员数, 食材数]` 的实际克数张量，与 `[食材数, 营养字段数]` 的每100g营养矩阵相乘，一次获得整代每人的营养总量。目标达标比例、分数、缺口、UL归一化超量和适应度均在PyTorch中批量计算，使用 `torch.float64` 及 `torch.inference_mode()`，不建立梯度图。矩阵维度规则见 [PyTorch矩阵乘法文档](https://docs.pytorch.org/docs/stable/generated/torch.matmul)。

缺失UL字段使用独立掩码，只对实际吃到的食材传播未知状态；数值矩阵中的占位0仅用于累计已知摄入下界，不代表该字段已通过检查。`evaluate()` 单个评分和 `evaluate_population()` 批量评分共享同一套计算逻辑。每代都重新编码当前实际克数，不沿用父代营养结果。

个体仍使用下述可变长度组合编码，选择、交叉、变异和合法性修复由Python执行。随机选择使用实例独立的 `random.Random(seed)`；张量评分本身不使用随机数。相同输入、种子、PyTorch版本和设备下可复现；浮点舍入可能导致不同设备的演化轨迹不同。CPU/GPU性能取决于种群和家庭规模，小规模计算不保证GPU更快。

`Dish`保存食材ID、十分比整数单位与全家组合总克数；`Fruit`保存食材ID、比例与每名成员的餐次克数；`Meal`保存非主食组合、主食ID和水果；`Individual`包含三餐。

早餐/午餐/晚餐槽位数为 `n / n+1 / n`。各餐全家非主食预算为 `150/250/200 × Σ(E/2000)` g，除以槽位数得到B。初始化重量范围0.8B–1.2B，变异边界0.5B–1.5B。混合组合数量从截断正态 `N(2.5, 1)` 抽样后取整；1–4种且不重复，单独供应项固定一种。组合按各成员 `E/2000` 的比例分配，每个人对所有实际克数单独累计营养。

种类变异每个非精英子代只抽一个事件：25%同类替换、5%从其他兼容模块复制编号、5%删除、5%添加、60%不变。比例用10个百分点为单位，非主食最小10%；水果两种时各20%–80%，新增第二种水果使用20%。删除与添加保持模块总克数。

分量变异独立以30%触发，在非主食比例、非主食总克数、水果配比、水果转餐四种可执行操作中选择。水果配比转移10或20个百分点，两者可行时各50%。全部子代均校验后重新计算营养，不沿用父代缓存。精英深拷贝，不接受交叉或变异。

## 输出

`results.json`包含：

- `config`、`seed`、`members`、原始数据元信息。
- `backend`：PyTorch版本、实际计算设备及浮点精度。
- `history`：第0–50代的最优方案基础分、最优适应度、当代最高基础分、平均分、已知超限比例、无已知超限但检查不完整比例、多样性。最优基础分受UL排序影响，可能不同于最高基础分。
- `plans`：最多3个不同方案，包括三餐全家组合克数、各食材克数/比例、每人主食及水果克数。
- 每人 `portions` 列出逐餐逐食材实际克数；`totals`包含实际能量和所有营养总量；`targets`、`scores`、`deficits`分别为目标、分数和缺口；`ul_checks`列出实际量、上限、归一化超量及检查状态。

100分只表示本模型选中的营养指标达到目标。模型不对能量、碳水或脂肪超目标扣分，不计算钠；亦不代表完整的膳食质量或安全结论。

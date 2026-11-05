"""成人家庭三餐遗传算法：组织遗传操作并使用PyTorch批量计算营养与适应度。"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

import device_utils
from cnf_data import load_cnf_data
from ga_models import (
    ALL_FIELDS,
    CATEGORIES,
    EPS,
    MEALS,
    NUTRIENTS,
    UL,
    WEIGHTS,
    Config,
    Dish,
    Evaluation,
    Food,
    Fruit,
    Individual,
    Meal,
    Person,
)
from genetic_operators import evolution


def load_foods(
    path: str | Path, candidates_path: str | Path | None = None
) -> tuple[list[Food], dict[str, Any]]:
    """从CNF数据目录或自定义JSON文件加载候选食材。

    Args:
        path: CNF关系表所在目录，或包含foods和可选metadata的JSON文件。
        candidates_path: CNF候选清单路径；省略时使用数据目录下的candidates.json。
            读取自定义食材JSON时不能指定此参数。

    Returns:
        食材对象列表与数据来源、字段映射等元信息组成的二元组。

    Raises:
        OSError: 输入文件不存在或无法读取。
        ValueError: JSON格式、CNF数据校验失败，或候选清单参数使用不当。
        KeyError: 输入缺少必要字段。
    """
    if Path(path).is_dir():
        rows, metadata = load_cnf_data(path, candidates_path)
        return [Food(**row) for row in rows], metadata
    if candidates_path is not None:
        raise ValueError("--candidates is only valid with a CNF dataset directory")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Food(**row) for row in data["foods"]], data.get("metadata", {})


def bounded_fruit_weights(values: list[float]) -> list[float]:
    """将某位成员的三餐水果重量修复为全天250g、每餐50–150g。

    先按比例缩放到250g，再通过二分查找公共偏移量，在满足上下界的
    同时保持总量约束。输入全为0时从三餐均分开始，不修改原列表。

    Args:
        values: 按早、午、晚顺序排列的三个非负有限重量，单位g。

    Returns:
        修复后的三个浮点重量，总量在数值精度范围内等于250g。

    Raises:
        ValueError: 长度不是3，或包含负数、无穷大、NaN。
    """
    if len(values) != 3 or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Three finite nonnegative fruit weights required")
    total = sum(values)
    scaled = [v * 250 / total for v in values] if total else [250 / 3] * 3
    lo, hi = min(scaled) - 150, max(scaled) - 50
    for _ in range(80):
        shift = (lo + hi) / 2
        if sum(min(150, max(50, v - shift)) for v in scaled) > 250:
            lo = shift
        else:
            hi = shift
    return [min(150, max(50, v - (lo + hi) / 2)) for v in scaled]


class MealGA:
    """为成人家庭生成并优化一日三餐的遗传算法运行器。

    负责输入校验、个体初始化、份量分配、PyTorch批量评分、进化流程和
    结果组织；父代选择、幸存者选择、交叉与突变委托给genetic_operators。
    每个实例使用独立随机数生成器，营养计算采用float64且不建立梯度图。

    Attributes:
        config: 种群规模、代数、概率、份量和设备等运行参数。
        rng: 由seed初始化的random.Random实例，供全部遗传操作共用。
        people: 按输入顺序保存的成人成员列表。
        foods: 以食材编号为键的候选食材字典。
        device: 实际使用的PyTorch计算设备。
        nutrient_matrix: 形状为[食材数, 营养字段数]的每100g营养矩阵。
        missing_matrix: 与营养矩阵同形的缺失标记，用于传播UL未知状态。
        target_matrix: 形状为[成员数, 评分营养素数]的个人每日目标矩阵。
    """

    def __init__(
        self,
        foods: list[Food],
        people: list[Person],
        config: Config | None = None,
        seed: int | None = None,
    ):
        """校验成员与候选池，并构建固定的营养、目标及UL张量。

        Args:
            foods: 25–40种编号唯一的食材，包含恰好3种主食、至少2种水果。
                每个混合兼容组至少有4种食材，评分必需营养值不能缺失。
            people: 非空的成人成员列表，成员名称必须唯一。
            config: 算法配置；省略时使用Config默认值。
            seed: 遗传操作的随机种子；None表示不指定固定种子。

        Raises:
            ValueError: 成员、候选池、营养值或设备配置不满足要求。
        """
        self.config = config or Config()
        self.rng = random.Random(seed)
        self.people = people
        if not people or len({p.name for p in people}) != len(people):
            raise ValueError("At least one adult and unique member names required")
        self.foods = {f.id: f for f in foods}
        if not 25 <= len(foods) <= 40 or len(self.foods) != len(foods):
            raise ValueError("Provide 25–40 foods with unique IDs")
        for f in foods:
            if f.category not in CATEGORIES or f.role not in {
                "staple",
                "fruit",
                "mixed",
                "standalone",
            }:
                raise ValueError(f"Invalid food category/role: {f.id}")
            if (f.role == "fruit") != (f.category == "fruit"):
                raise ValueError(f"Fruit role/category mismatch: {f.id}")
            if f.role == "staple" and f.category != "starch":
                raise ValueError(f"Staple must be starch: {f.id}")
            for k in ALL_FIELDS:
                value = f.per100g.get(k)
                if value is None:
                    if k not in {
                        "preformed_a_ug",
                        "ul_applicable_e_mg",
                        "synthetic_folic_acid_ug",
                    }:
                        raise ValueError(
                            f"Missing nutrient {k}: {f.id}; unknown is not zero"
                        )
                elif (
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"Invalid nutrient {k}: {f.id}")
        self.staples = [f.id for f in foods if f.role == "staple"]
        self.fruits = [f.id for f in foods if f.role == "fruit"]
        self.sides = [f.id for f in foods if f.role in {"mixed", "standalone"}]
        if len(self.staples) != 3 or len(self.fruits) < 2 or not self.sides:
            raise ValueError(
                "Exactly 3 staples, at least 2 fruits, and side foods required"
            )
        if any(self.foods[f].per100g["carbohydrate_g"] <= 0 for f in self.staples):
            raise ValueError("Staples require positive carbohydrate values")
        groups: dict[str, int] = {}
        for f in foods:
            if f.role == "mixed":
                groups[f.mix_group] = groups.get(f.mix_group, 0) + 1
        if any(v < 4 for v in groups.values()):
            raise ValueError(
                "Each mixed-food compatibility group needs at least 4 foods"
            )
        self.factor_sum = sum(p.factor for p in people)
        self.device = device_utils.resolve_device(self.config.device)
        self.dtype = torch.float64
        self.food_index = {fid: i for i, fid in enumerate(self.foods)}
        self.field_index = {name: i for i, name in enumerate(ALL_FIELDS)}
        # Missing UL fields have a separate mask. Their zeros in the arithmetic
        # matrix are only used to calculate a known lower bound, never a pass.
        self.nutrient_matrix = torch.tensor(
            [
                [
                    f.per100g.get(k) if f.per100g.get(k) is not None else 0.0
                    for k in ALL_FIELDS
                ]
                for f in self.foods.values()
            ],
            dtype=self.dtype,
            device=self.device,
        )
        self.missing_matrix = torch.tensor(
            [
                [f.per100g.get(k) is None for k in ALL_FIELDS]
                for f in self.foods.values()
            ],
            dtype=self.dtype,
            device=self.device,
        )
        self.nutrient_columns = [self.field_index[k] for k in NUTRIENTS]
        self.ul_columns = [self.field_index[k] for k in UL]
        self.target_matrix = torch.tensor(
            [[p.targets()[k] for k in NUTRIENTS] for p in self.people],
            dtype=self.dtype,
            device=self.device,
        )
        self.score_weights = torch.tensor(
            list(WEIGHTS.values()), dtype=self.dtype, device=self.device
        )
        self.ul_limits = torch.tensor(
            list(UL.values()), dtype=self.dtype, device=self.device
        )

    def slot_count(self, meal: int) -> int:
        """返回指定餐次的非主食槽位数：早/晚餐为n，午餐为n+1。

        Args:
            meal: 餐次索引，0为早餐、1为午餐、2为晚餐。

        Returns:
            该餐全家共用的非主食组合数量。
        """
        return len(self.people) + (meal == 1)

    def baseline(self, meal: int) -> float:
        """计算某餐每个非主食槽位的基准重量B，单位g。

        Args:
            meal: 餐次索引，取0、1或2。

        Returns:
            参考成人该餐预算乘以全家份量系数之和，再除以槽位数。
            人数只通过份量系数和槽位数参与计算，不额外重复乘人数。
        """
        return (
            self.config.side_budgets_g[meal] * self.factor_sum / self.slot_count(meal)
        )

    def staple_weights(self, meal: int, food_id: str) -> list[float]:
        """按个人碳水目标与餐次比例即时计算主食重量。

        Args:
            meal: 餐次索引，取0、1或2。
            food_id: 所选主食的食材编号，其每100g碳水含量必须大于0。

        Returns:
            按成员顺序排列的可食重量列表，单位g。每人重量等于每日
            碳水目标×主食供应比例×餐次比例÷每100g碳水含量×100。
        """
        c = self.foods[food_id].per100g["carbohydrate_g"]
        return [
            p.targets()["carbohydrate_g"]
            * self.config.staple_fraction
            * self.config.meal_shares[meal]
            / c
            * 100
            for p in self.people
        ]

    def compatible(self, ids: list[str]) -> bool:
        """判断食材能否组成非主食组合，不检查比例和重量。

        Args:
            ids: 已存在于候选池中的食材编号列表。

        Returns:
            单独供应项或可混合食材独立出现时为True；多种食材必须均为
            mixed角色且属于同一个mix_group。水果和主食不作为非主食混合。
        """
        fs = [self.foods[f] for f in ids]
        return (len(fs) == 1 and fs[0].role in {"mixed", "standalone"}) or (
            all(f.role == "mixed" for f in fs) and len({f.mix_group for f in fs}) == 1
        )

    def new_dish(self, meal: int) -> Dish:
        """随机创建一个符合角色兼容规则的非主食组合。

        Args:
            meal: 餐次索引，用于确定该槽位的基准重量B。

        Returns:
            包含食材编号、比例单位和全家总重量的Dish。混合食材数由
            截断正态分布N(2.5, 1)取整得到1–4种；单独供应项固定一种。
            比例以10%为单位且总和100%，重量在0.8B–1.2B内均匀抽取。
        """
        first = self.rng.choice(self.sides)
        food = self.foods[first]
        if food.role == "standalone":
            ids = [first]
        else:
            sample = self.rng.gauss(2.5, 1)
            while not 0.5 <= sample < 4.5:
                sample = self.rng.gauss(2.5, 1)
            n = math.floor(sample + 0.5)
            pool = [
                f
                for f in self.sides
                if self.foods[f].role == "mixed"
                and self.foods[f].mix_group == food.mix_group
            ]
            ids = self.rng.sample(pool, n)
        units = [1] * len(ids)
        for _ in range(10 - len(ids)):
            units[self.rng.randrange(len(ids))] += 1
        return Dish(ids, units, self.baseline(meal) * self.rng.uniform(0.8, 1.2))

    def initialize(self) -> Individual:
        """生成并校验一个随机个体，不生成整个种群。

        Returns:
            含三餐的Individual。各餐独立选择主食；水果以相同概率选择
            一种或两种，两种时各占50%，每人每天250g初始均分到三餐。
        """
        meals = []
        for m in range(3):
            fruit_ids = self.rng.sample(
                self.fruits, 1 if self.rng.random() < 0.5 else 2
            )
            fruit = Fruit(
                fruit_ids,
                [10] if len(fruit_ids) == 1 else [5, 5],
                [250 / 3] * len(self.people),
            )
            meals.append(
                Meal(
                    [self.new_dish(m) for _ in range(self.slot_count(m))],
                    self.rng.choice(self.staples),
                    fruit,
                )
            )
        result = Individual(meals)
        self.validate(result)
        return result

    def validate(self, individual: Individual) -> None:
        """检查个体的组合数量、食材角色、比例、重量和水果总量。

        Args:
            individual: 待检查的完整三餐个体；检查过程不修改其内容。

        Raises:
            ValueError: 餐次或槽位数量错误、编号重复或无效、食材不兼容、
                比例不合法、非主食重量超出0.5B–1.5B，或水果不满足每人
                全天250g及每餐50–150g的约束。
        """
        if len(individual.meals) != 3:
            raise ValueError("Exactly three meals required")
        for m, meal in enumerate(individual.meals):
            if (
                len(meal.dishes) != self.slot_count(m)
                or meal.staple_id not in self.staples
            ):
                raise ValueError("Invalid meal slots/staple")
            for module in [*meal.dishes, meal.fruit]:
                fruit = isinstance(module, Fruit)
                minimum, maximum = (2, 2) if fruit else (1, 4)
                if (
                    not 1 <= len(module.food_ids) <= maximum
                    or len(set(module.food_ids)) != len(module.food_ids)
                    or len(module.units) != len(module.food_ids)
                    or any(f not in self.foods for f in module.food_ids)
                    or any(type(u) is not int or u < minimum for u in module.units)
                    or sum(module.units) != 10
                ):
                    raise ValueError("Invalid food IDs or proportion units")
                if fruit:
                    if any(f not in self.fruits for f in module.food_ids):
                        raise ValueError("Non-fruit in fruit module")
                elif not self.compatible(module.food_ids):
                    raise ValueError("Incompatible side ingredients")
            for dish in meal.dishes:
                b = self.baseline(m)
                if (
                    not math.isfinite(dish.weight_g)
                    or not 0.5 * b - EPS <= dish.weight_g <= 1.5 * b + EPS
                ):
                    raise ValueError("Dish weight outside slot bounds")
            weights = meal.fruit.person_weights_g
            if len(weights) != len(self.people) or any(
                not math.isfinite(w) or not 50 - EPS <= w <= 150 + EPS for w in weights
            ):
                raise ValueError("Fruit meal weights outside bounds")
        for i in range(len(self.people)):
            if not math.isclose(
                sum(m.fruit.person_weights_g[i] for m in individual.meals),
                250,
                abs_tol=EPS,
            ):
                raise ValueError("Daily fruit must equal 250g per adult")

    def repair_fruit(self, individual: Individual) -> None:
        """逐人原地修复三餐水果克数，保持食材编号与餐内比例。

        Args:
            individual: 需要修复的个体，通常是跨餐继承水果配置后的子代。

        Notes:
            修复后每位成员全天水果共250g、每餐50–150g。
            此方法仅修复重量，不负责修复其他非法编码或计算营养。
        """
        for p in range(len(self.people)):
            weights = bounded_fruit_weights(
                [m.fruit.person_weights_g[p] for m in individual.meals]
            )
            for meal, w in zip(individual.meals, weights):
                meal.fruit.person_weights_g[p] = w

    def portions(self, individual: Individual, person: int):
        """逐项生成某成员分到的主食、非主食和水果可食克数。

        Args:
            individual: 包含三餐的个体。
            person: 成员在people中的索引。

        Yields:
            (餐次索引, 模块名称, 食材编号, 实际克数)。主食重量按公式
            计算；共用非主食按成员E/2000占全家系数之和的比例分配；
            水果按该成员餐次重量及食材比例分配。同一食材可多次产出。
        """
        fraction = self.people[person].factor / self.factor_sum
        for m, meal in enumerate(individual.meals):
            yield (
                m,
                "staple",
                meal.staple_id,
                self.staple_weights(m, meal.staple_id)[person],
            )
            for j, dish in enumerate(meal.dishes):
                for fid, units in zip(dish.food_ids, dish.units):
                    yield m, f"dish_{j + 1}", fid, dish.weight_g * fraction * units / 10
            for fid, units in zip(meal.fruit.food_ids, meal.fruit.units):
                yield m, "fruit", fid, meal.fruit.person_weights_g[person] * units / 10

    def evaluate(self, individual: Individual) -> Evaluation:
        """通过统一的PyTorch批量评分逻辑评估单个个体。

        Args:
            individual: 待校验并重新计算营养的个体。

        Returns:
            各成员营养明细、家庭基础分、适应度、归一化超限总量和UL状态。
        """
        return self.evaluate_population([individual])[0]

    def population_weights(self, population: list[Individual]) -> torch.Tensor:
        """校验种群并将逐人食材摄入量编码为float64张量。

        Args:
            population: 待编码的个体列表，可为空。

        Returns:
            位于self.device、形状为[种群大小, 成员数, 食材数]的克数张量。
            食材列顺序由food_index确定，跨餐和跨模块的同编号食材累计
            到同一列；空种群保留成员数与食材数两个维度。

        Raises:
            ValueError: 任意个体未通过合法性检查。
        """
        if not population:
            return torch.empty(
                (0, len(self.people), len(self.foods)),
                dtype=self.dtype,
                device=self.device,
            )
        weights = []
        for individual in population:
            self.validate(individual)
            members = []
            for p in range(len(self.people)):
                row = [0.0] * len(self.foods)
                for _, _, fid, grams in self.portions(individual, p):
                    row[self.food_index[fid]] += grams
                members.append(row)
            weights.append(members)
        return torch.tensor(weights, dtype=self.dtype, device=self.device)

    @torch.inference_mode()
    def evaluate_population(self, population: list[Individual]) -> list[Evaluation]:
        """使用PyTorch批量计算整代营养、达标分、缺口与UL超限。

        Args:
            population: 需要重新评分的个体列表，不使用父代营养结果缓存。

        Returns:
            与输入顺序一致的Evaluation列表，空输入返回空列表。家庭基础分
            为成员得分均值，适应度再减去人均归一化超限扣分。UL数据
            不完整时保留未知状态；已知摄入下界已超限时仍判为超限。

        Raises:
            ValueError: 任意个体未通过合法性检查。

        Notes:
            所有食材来源的全部营养均参与累计。计算使用float64且禁用
            自动微分，返回普通Python对象以供选择、记录和JSON序列化。
        """
        if not population:
            return []
        grams = self.population_weights(population)
        # [B, P, F] @ [F, K] -> [B, P, K]. Repeated ingredients have
        # already been combined, so all meal/module sources contribute once.
        known_totals = (grams / 100) @ self.nutrient_matrix
        missing = (grams.gt(0).to(self.dtype) @ self.missing_matrix).gt(0)
        actual = known_totals[..., self.nutrient_columns]
        scores = torch.clamp(actual / self.target_matrix, max=1) * self.score_weights
        deficits = torch.clamp(self.target_matrix - actual, min=0)
        person_scores = scores.sum(dim=-1)
        base_scores = person_scores.mean(dim=-1)
        ul_missing = missing[..., self.ul_columns]
        ul_excess = torch.clamp(
            known_totals[..., self.ul_columns] / self.ul_limits - 1, min=0
        )
        # With incomplete data, only a known excess above the tolerance counts.
        assessed = ~ul_missing | ul_excess.gt(EPS)
        ul_excess = torch.where(assessed, ul_excess, 0.0)
        total_excess = ul_excess.sum(dim=(-2, -1))
        fitness = base_scores - self.config.excess_penalty * total_excess / len(
            self.people
        )
        fully_checked = ~ul_missing.any(dim=(-2, -1))

        # Convert complete batches, avoiding a device synchronization per scalar.
        totals_cpu = known_totals.cpu().tolist()
        missing_cpu = missing.cpu().tolist()
        scores_cpu, deficits_cpu = scores.cpu().tolist(), deficits.cpu().tolist()
        person_scores_cpu = person_scores.cpu().tolist()
        ratios_cpu, assessed_cpu = ul_excess.cpu().tolist(), assessed.cpu().tolist()
        summary_cpu = (
            torch.stack((base_scores, fitness, total_excess), dim=-1).cpu().tolist()
        )
        checked_cpu = fully_checked.cpu().tolist()
        evaluations = []
        for b in range(len(population)):
            results = []
            for p, person in enumerate(self.people):
                totals = {
                    k: None if missing_cpu[b][p][j] else totals_cpu[b][p][j]
                    for j, k in enumerate(ALL_FIELDS)
                }
                checks = {}
                for j, (k, limit) in enumerate(UL.items()):
                    ratio = ratios_cpu[b][p][j] if assessed_cpu[b][p][j] else None
                    checks[k] = {
                        "actual": totals[k],
                        "limit": limit,
                        "excess_ratio": ratio,
                        "known_lower_bound": totals_cpu[b][p][self.field_index[k]],
                        "fully_assessed": totals[k] is not None,
                        "status": "not_assessed"
                        if ratio is None
                        else ("exceeded" if ratio > EPS else "checked_no_excess"),
                    }
                results.append(
                    {
                        "name": person.name,
                        "totals": totals,
                        "targets": person.targets(),
                        "scores": dict(zip(NUTRIENTS, scores_cpu[b][p])),
                        "score": person_scores_cpu[b][p],
                        "deficits": dict(zip(NUTRIENTS, deficits_cpu[b][p])),
                        "ul_checks": checks,
                    }
                )
            base, fit, excess = summary_cpu[b]
            status = (
                "exceeded"
                if excess > EPS
                else ("checked_no_excess" if checked_cpu[b] else "not_fully_assessed")
            )
            evaluations.append(Evaluation(results, base, fit, excess, status))
        return evaluations

    @staticmethod
    def signature(individual: Individual) -> str:
        """构建用于方案去重和统计种群多样性的规范化JSON字符串。

        Args:
            individual: 待生成签名的个体，不会被修改。

        Returns:
            忽略餐内组合顺序及食材排列、将重量保留两位小数的签名。
            餐次顺序、主食类型、配比及各成员水果重量仍参与区分。
        """
        # Ingredient order is irrelevant; weights rounded only for diversity/deduplication.
        meals = []
        for meal in individual.meals:
            dishes = sorted(
                (tuple(sorted(zip(d.food_ids, d.units))), round(d.weight_g, 2))
                for d in meal.dishes
            )
            meals.append(
                (
                    meal.staple_id,
                    dishes,
                    sorted(zip(meal.fruit.food_ids, meal.fruit.units)),
                    [round(w, 2) for w in meal.fruit.person_weights_g],
                )
            )
        return json.dumps(meals, sort_keys=True)

    def run(self) -> tuple[list[tuple[Individual, Evaluation]], list[dict[str, Any]]]:
        """通过独立的evolution模块运行进化流程。

        Returns:
            (优选方案, 逐代统计)二元组。方案为最多3个不同的
            (Individual, Evaluation)，统计包含第0代至配置代数。

        Notes:
            此方法是统一运行入口；选择、交叉、突变及逐代循环全部由
            genetic_operators中的模块执行，使用并推进本实例的rng状态。
        """
        return evolution.evolve(self)

    def report(self, individual: Individual, evaluation: Evaluation) -> dict[str, Any]:
        """将方案与对应评分组织为可供JSON序列化的展示数据。

        Args:
            individual: 要展示的完整三餐方案。
            evaluation: 该个体的最新评分结果，调用方需保证两者对应。

        Returns:
            含三餐食材、全家组合克数、各成员实际克数、营养总量、目标、
            缺口、UL检查与得分的字典。展示用整数分采用四舍五入，同时
            保留未取整的原始分数。本方法不重新评分，也不修改原评分。
        """
        meals = []
        for m, meal in enumerate(individual.meals):
            meals.append(
                {
                    "meal": MEALS[m],
                    "staple": {
                        "id": meal.staple_id,
                        "name": self.foods[meal.staple_id].name,
                        "person_weights_g": self.staple_weights(m, meal.staple_id),
                    },
                    "dishes": [
                        {
                            "total_weight_g": dish.weight_g,
                            "ingredients": [
                                {
                                    "id": fid,
                                    "name": self.foods[fid].name,
                                    "percent": u * 10,
                                    "household_weight_g": dish.weight_g * u / 10,
                                }
                                for fid, u in zip(dish.food_ids, dish.units)
                            ],
                        }
                        for dish in meal.dishes
                    ],
                    "fruit": {
                        "person_weights_g": meal.fruit.person_weights_g,
                        "ingredients": [
                            {"id": fid, "name": self.foods[fid].name, "percent": u * 10}
                            for fid, u in zip(meal.fruit.food_ids, meal.fruit.units)
                        ],
                    },
                }
            )
        people = copy.deepcopy(evaluation.people)
        for i, person in enumerate(people):
            person["score_rounded"] = math.floor(person["score"] + 0.5)
            person["scores_rounded"] = {
                k: math.floor(v + 0.5) for k, v in person["scores"].items()
            }
            person["portions"] = [
                {
                    "meal": MEALS[m],
                    "module": module,
                    "id": fid,
                    "name": self.foods[fid].name,
                    "weight_g": grams,
                }
                for m, module, fid, grams in self.portions(individual, i)
            ]
        return {
            "base_score": evaluation.base_score,
            "fitness": evaluation.fitness,
            "normalized_ul_excess": evaluation.excess,
            "ul_status": evaluation.ul_status,
            "meals": meals,
            "people": people,
        }


def main() -> None:
    """运行命令行入口：加载输入、执行遗传算法并写出JSON结果。

    参数由argparse从命令行读取，支持指定数据集、候选池、成员文件、
    种群规模、代数、种子、主食突变概率、计算设备与输出路径。
    完成后向标准输出打印数据来源、PyTorch设备和各优选方案摘要。
    输入或配置无效时由参数解析器或相应校验报告错误。
    """
    parser = argparse.ArgumentParser(
        description="成人家庭一日三餐遗传算法（默认使用CNF 2026数据集）"
    )
    parser.add_argument(
        "--foods", default="dataset", help="CNF数据目录，或自定义食材JSON"
    )
    parser.add_argument(
        "--candidates", help="CNF候选清单JSON，默认使用数据目录下的candidates.json"
    )
    parser.add_argument("--people", default="data/people.json")
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--staple-mutation", type=float, default=0.0)
    parser.add_argument(
        "--device", default="cpu", help="PyTorch设备：cpu、cuda、cuda:N或auto"
    )
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()
    try:
        foods, metadata = load_foods(args.foods, args.candidates)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    people = [
        Person(**p) for p in json.loads(Path(args.people).read_text(encoding="utf-8"))
    ]
    config = Config(
        population_size=args.population,
        generations=args.generations,
        staple_mutation_probability=args.staple_mutation,
        device=args.device,
    )
    try:
        ga = MealGA(foods, people, config, args.seed)
    except ValueError as error:
        parser.error(str(error))
    plans, history = ga.run()
    output = {
        "data_metadata": metadata,
        "config": asdict(config),
        "seed": args.seed,
        "backend": {
            "framework": "pytorch",
            "version": str(torch.__version__),
            "device": str(ga.device),
            "dtype": str(ga.dtype),
        },
        "members": [asdict(p) for p in people],
        "history": history,
        "plans": [ga.report(ind, ev) for ind, ev in plans],
    }
    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if metadata.get("demo_only"):
        print("注意：当前为未核对的演示食材数据，结果仅用于算法测试。")
    else:
        print(
            f"食材数据：{metadata.get('source', args.foods)}，候选食材{len(foods)}种。"
        )
    print(f"评分后端：PyTorch {torch.__version__}，设备={ga.device}，精度={ga.dtype}。")
    for i, (_, ev) in enumerate(plans, 1):
        print(
            f"方案{i}: 家庭平均分={ev.base_score:.2f}, 适应度={ev.fitness:.2f}, UL={ev.ul_status}"
        )
    print(f"已保存 {len(history)} 代统计及最多3个方案：{args.output}")


if __name__ == "__main__":
    main()

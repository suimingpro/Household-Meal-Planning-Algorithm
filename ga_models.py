"""三餐规划算法共用的数据模型、营养常量与运行配置。

本模块使用标准库dataclass表示食材、成人成员、三餐编码和评分结果，
供数据加载、遗传操作及PyTorch评分模块共用，不依赖计算设备。

Attributes:
    MEALS: 早餐、午餐、晚餐的固定顺序，对应索引0、1、2。
    CATEGORIES: 项目采用的五类食材标识，不限制实际参与累计的营养字段。
    NUTRIENTS: 参与100分达标评分的11项营养指标及其固定顺序。
        字段后缀g、mg、ug分别表示克、毫克、微克。
    WEIGHTS: 各评分营养指标的权重，总和为100分。
    UL: 题目指定的每日可耐受最高摄入量参数，单位由字段后缀确定。
        A与叶酸的UL字段分别使用预成型A和合成叶酸，不替代充足性字段。
    ALL_FIELDS: 能量、评分指标及UL指标去重后的有序字段集合。
    EPS: 比较重量边界和超限程度时使用的浮点容差。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

MEALS = ("breakfast", "lunch", "dinner")
CATEGORIES = {"vegetable", "starch", "protein", "fruit", "other"}
NUTRIENTS = ("carbohydrate_g", "protein_g", "fat_g", "vitamin_a_rae_ug",
             "vitamin_c_mg", "vitamin_d_ug", "vitamin_e_mg", "folate_dfe_ug",
             "calcium_mg", "iron_mg", "potassium_mg")
WEIGHTS = dict(zip(NUTRIENTS, (20, 20, 20, 4, 4, 4, 4, 4, 20 / 3, 20 / 3, 20 / 3)))
# These are the assignment's model parameters, not age-specific clinical advice.
UL = {"preformed_a_ug": 3000, "vitamin_c_mg": 2000, "vitamin_d_ug": 100,
      "ul_applicable_e_mg": 1000, "synthetic_folic_acid_ug": 1000,
      "calcium_mg": 2500, "iron_mg": 45}
ALL_FIELDS = tuple(dict.fromkeys(("energy_kcal",) + NUTRIENTS + tuple(UL)))
EPS = 1e-7


@dataclass(frozen=True)
class Food:
    """一种候选食材及其每100g可食部分的营养数据。

    Attributes:
        id: 唯一食材编号；使用CNF数据时对应Food_Code。
        name: 展示名称，应能区分生熟及加工状态。
        category: vegetable、starch、protein、fruit或other五种项目分类之一。
        role: staple主食、fruit水果、mixed允许混合或standalone单独供应。
        per100g: 营养字段到每100g含量的映射，能量单位为kcal，其余单位
            由字段后缀确定。允许缺失的UL字段以None表示未知，不能当成0。
        mix_group: 混合兼容组，默认savory；不同兼容组的食材不混合。
        max_percent: 在非主食组合中的重量比例上限，默认100，按10个百分点
            设置。低于100时必须为mixed角色，不能单独供应。

    Notes:
        食材角色、营养完整性及兼容组校验由加载器和MealGA负责。
        frozen只禁止重新绑定属性，不会冻结per100g字典内部的数据。
    """

    id: str
    name: str
    category: str
    role: str  # staple, fruit, mixed, standalone
    per100g: dict[str, float | None]
    mix_group: str = "savory"
    max_percent: int = 100

    def __post_init__(self):
        """检查比例上限是否为10–100之间的10倍整数及角色是否可混合。"""
        if type(self.max_percent) is not int or not 10 <= self.max_percent <= 100 or self.max_percent % 10:
            raise ValueError("max_percent must be an integer multiple of 10 in [10, 100]")
        if self.max_percent < 100 and self.role != "mixed":
            raise ValueError("Only mixed side ingredients can have max_percent below 100")


@dataclass(frozen=True)
class Person:
    """成人成员的参考分组、体重和预先给定的每日能量需求。

    Attributes:
        name: 成员名称，家庭内的唯一性由MealGA检查。
        age: 年龄，必须为18岁及以上的整数。
        reference_group: male或female，用于选择模型中的营养参考目标。
        weight_kg: 体重，单位kg，必须为正有限数。
        energy_kcal: 输入的参考能量需求，单位kcal/天，必须为正有限数。

    Notes:
        本模型不预测能量需求，目标与UL使用项目指定的参数，未按年龄
        进一步细分。年龄用于成人身份校验和结果记录。
    """

    name: str
    age: int
    reference_group: str  # male / female, used for reference targets
    weight_kg: float
    energy_kcal: float

    def __post_init__(self):
        """构造后检查参考分组、成人年龄、体重及能量需求。

        Raises:
            ValueError: 分组不是male/female、年龄不符合成人要求，或体重、
                能量需求不是正有限数。
        """
        if self.reference_group not in {"male", "female"}:
            raise ValueError("reference_group must be male or female")
        if not isinstance(self.age, int) or self.age < 18:
            raise ValueError("This model accepts adults aged 18+ only")
        if any(not math.isfinite(v) or v <= 0 for v in (self.weight_kg, self.energy_kcal)):
            raise ValueError("Weight and reference energy must be positive and finite")

    @property
    def factor(self) -> float:
        """返回以2000kcal参考成人为基准的无量纲份量系数E/2000。

        该系数用于非主食预算和全家共用组合的个人份量分配。
        """
        return self.energy_kcal / 2000

    def targets(self) -> dict[str, float]:
        """计算此成员每日用于达标评分的营养目标。

        Returns:
            按NUTRIENTS顺序构造的目标字典。碳水为E×0.55÷4 g，蛋白质
            为0.8×体重kg g，脂肪为E×0.275÷9 g；维生素和矿物质使用
            项目规定的固定值或male/female分组值。

        Notes:
            A以μg RAE计，叶酸以μg DFE计；这些充足性目标不是UL。
        """
        male = self.reference_group == "male"
        return dict(zip(NUTRIENTS, (
            self.energy_kcal * .55 / 4, .8 * self.weight_kg,
            self.energy_kcal * .275 / 9, 900 if male else 700,
            90 if male else 75, 15, 15, 400, 1000,
            8 if male else 18, 3400 if male else 2600)))


@dataclass(frozen=True)
class Config:
    """遗传算法与三餐份量的不可重新赋值配置。

    Attributes:
        population_size: 种群规模，默认100，至少4以支持四元锦标赛。
        generations: 初始第0代之后的进化次数，默认50。
        elite_count: 每代直接深拷贝保留的精英数，默认2，须小于种群规模。
        staple_fraction: 主食供应每日碳水目标的比例，默认0.8。
        meal_shares: 主食碳水在早、午、晚餐的分配比例，默认0.25/0.4/0.35。
        side_budgets_g: 参考成人各餐非主食预算，默认150/250/200g。
        portion_mutation_probability: 每个非精英子代分量突变的独立概率，默认0.3。
        crossover_probability: 每个子代执行交叉而非随机复制父代的概率，默认1。
        staple_mutation_probability: 可选主食种类突变的独立概率，默认0。
        excess_penalty: 人均归一化UL超限比例的适应度扣分系数，默认10。
        device: PyTorch设备名称，默认cpu，也可用cuda、cuda:N或auto。

    Notes:
        种类突变的25%/5%/5%/5%互斥概率由mutation模块定义。
        设备是否可用由device_utils解析，不在本数据类中检测CUDA。
    """

    population_size: int = 100
    generations: int = 50
    elite_count: int = 2
    staple_fraction: float = .8
    meal_shares: tuple[float, ...] = (.25, .4, .35)
    side_budgets_g: tuple[float, ...] = (150, 250, 200)
    portion_mutation_probability: float = .3
    crossover_probability: float = 1.0
    staple_mutation_probability: float = 0.0
    excess_penalty: float = 10.0
    device: str = "cpu"

    def __post_init__(self):
        """构造后校验种群、代数、精英数、餐次比例及概率等参数。

        Raises:
            ValueError: 种群不足4、精英数或代数越界、餐次参数不是三个
                正有限数、分配比例之和不为1、概率不在[0, 1]，或扣分
                系数不是非负有限数。
        """
        if self.population_size < 4 or not 0 <= self.elite_count < self.population_size:
            raise ValueError("Population >= 4 and 0 <= elite_count < population required")
        if self.generations < 0:
            raise ValueError("generations must be nonnegative")
        if (len(self.meal_shares) != 3 or len(self.side_budgets_g) != 3
                or any(not math.isfinite(x) or x <= 0 for x in self.meal_shares + self.side_budgets_g)
                or not math.isclose(sum(self.meal_shares), 1)):
            raise ValueError("Three positive meal shares summing to 1 and three budgets required")
        if any(not 0 <= p <= 1 for p in (self.staple_fraction,
                self.portion_mutation_probability, self.crossover_probability,
                self.staple_mutation_probability)):
            raise ValueError("Probabilities and staple fraction must be in [0, 1]")
        if not math.isfinite(self.excess_penalty) or self.excess_penalty < 0:
            raise ValueError("excess_penalty must be finite and nonnegative")


@dataclass
class Dish:
    """一个全家共用的非主食组合，可包含混合食材或单独供应项。

    Attributes:
        food_ids: 1–4个互不重复的食材编号，单独供应项只能有一个编号。
        units: 与编号一一对应的整数比例单位，每单位为10%，每项至少1，
            总和为10；单食材时为[10]。
        weight_g: 此组合的全家可食总重量，单位g，不是每位成员的重量。

    Notes:
        个人分量按E/2000系数占全家系数之和的比例分配。
        本类不自动校验编码，合法性由MealGA.validate检查。
    """

    food_ids: list[str]
    units: list[int]  # tenths, always sum to 10
    weight_g: float


@dataclass
class Fruit:
    """某一餐全家共用的水果种类、配比及逐人成品可食重量。

    Attributes:
        food_ids: 1或2个互不重复的水果编号。
        units: 对应食材的10%比例单位，总和为10；一种水果为[10]，
            两种水果时各占2–8单位，即20%–80%。
        person_weights_g: 按成员顺序保存的该餐水果总重量，单位g。
            每位成员每餐50–150g，跨三餐合计250g。

    Notes:
        同餐所有成员使用相同水果配比，但可有不同餐次总重量。
        重量修复和跨餐总量校验由运行器执行。
    """

    food_ids: list[str]
    units: list[int]
    person_weights_g: list[float]


@dataclass
class Meal:
    """一餐的非主食组合、主食编号和水果配置。

    Attributes:
        dishes: 全家共用的非主食组合；早餐和晚餐各n个，午餐n+1个。
        staple_id: 所选主食编号，个人主食重量按营养目标即时派生。
        fruit: 本餐水果配置，包含食材配比及每名成员的水果克数。

    Notes:
        餐次由其在Individual.meals中的位置决定，不额外保存餐次字段。
        主食重量不作为独立基因保存。
    """

    dishes: list[Dish]
    staple_id: str
    fruit: Fruit


@dataclass
class Individual:
    """遗传算法中的一个完整家庭一日三餐方案。

    Attributes:
        meals: 长度为3的餐次列表，依次为早餐、午餐和晚餐。

    Notes:
        嵌套列表和组合均可变；交叉和精英保留时须深拷贝，避免子代
        修改父代或多个个体共享同一份编码。营养评分另存于Evaluation。
    """

    meals: list[Meal]


@dataclass
class Evaluation:
    """一个个体的逐人营养明细、家庭适应度及UL检查结果。

    Attributes:
        people: 按成员顺序保存的营养总量、目标、得分、缺口和UL检查字典。
        base_score: 成员基础得分的算术平均，最高100分，保留浮点精度。
        fitness: 基础分减去人均归一化超限扣分后的适应度，可为负数。
        excess: 全家各成员各项UL的归一化超限比例之和；信息不全但
            已知摄入下界超限时使用该下界计算。
        ul_status: checked_no_excess表示完整检查且无超限；
            not_fully_assessed表示无已知超限但检查不完整；
            exceeded表示存在已知超限。

    Notes:
        不按fitness单独排序，父代及幸存者选择共用key中的UL优先规则。
    """

    people: list[dict[str, Any]]
    base_score: float
    fitness: float
    excess: float
    ul_status: str

    @property
    def key(self) -> tuple[float, ...]:
        """返回可直接按升序比较的(UL状态等级, 超限程度, 负适应度)。

        完整检查且无超限优先于检查不完整，后者优先于已知超限。
        同等级先取归一化超限总量较小者，再取适应度较高者。

        Returns:
            三元比较键，数值越小代表方案越优。

        Raises:
            KeyError: ul_status不是模型支持的三个状态之一。
        """
        # Explicitly checked safe > incomplete UL assessment > known exceedance.
        tier = {"checked_no_excess": 0, "not_fully_assessed": 1, "exceeded": 2}[self.ul_status]
        return tier, self.excess, -self.fitness

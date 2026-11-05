"""加载CNF 2026关系表，并转换为三餐规划算法使用的候选食材数据。

依据候选清单的Food_Code关联食材名称、来源、分类及营养含量，校验
营养素代码、单位和记录完整性。模块只读取本地数据，不下载文件，
不执行遗传算法，也不计算个人营养得分。

Attributes:
    CNF_URL: 官方数据集地址，用于记录数据来源。
    NUTRIENT_MAP: 程序营养字段到CNF营养素代码及预期单位的映射。
        A的RAE与预成型A、叶酸DFE与合成叶酸分别映射，不互相替代。
    OPTIONAL_FIELDS: 允许缺失的预成型A与合成叶酸UL专用字段。
        缺失保留为None，不替换为0。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

CNF_URL = "https://open.canada.ca/data/en/dataset/1b6139bd-ed7e-4043-bc28-ff00e10f3109"
# field: (CNF nutrient code, required unit). Do not confuse RAE with retinol,
# DFE with folic acid, or microgram vitamin D with its IU counterpart.
NUTRIENT_MAP = {
    "energy_kcal": ("208", "kilocalorie"),
    "carbohydrate_g": ("205", "Gram"),
    "protein_g": ("203", "Gram"),
    "fat_g": ("204", "Gram"),
    "vitamin_a_rae_ug": ("320", "Microgram"),
    "vitamin_c_mg": ("401", "Milligram"),
    "vitamin_d_ug": ("328", "Microgram"),
    "vitamin_e_mg": ("323", "Milligram"),
    "folate_dfe_ug": ("435", "Microgram"),
    "calcium_mg": ("301", "Milligram"),
    "iron_mg": ("303", "Milligram"),
    "potassium_mg": ("306", "Milligram"),
    "preformed_a_ug": ("319", "Microgram"),
    "synthetic_folic_acid_ug": ("431", "Microgram"),
}
OPTIONAL_FIELDS = {"preformed_a_ug", "synthetic_folic_acid_ug"}


def csv_rows(path: Path, required: set[str]):
    """按行读取CSV，在开始迭代时检查必需列是否齐全。

    Args:
        path: 待读取的CSV文件路径，支持UTF-8 BOM及CRLF换行。
        required: 必须出现在表头中的列名集合；空集合表示不要求特定列。

    Yields:
        csv.DictReader生成的原始行字典，以列名为键，不转换营养数值。

    Raises:
        OSError: 文件不存在、无读取权限或发生其他文件读取错误。
        ValueError: 表头缺少required指定的列。

    Notes:
        此函数是生成器，读取及校验在迭代期间执行；使用流式读取避免
        将整张营养含量表同时加载到内存，结束迭代后关闭文件。
    """
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path}: missing CSV columns {', '.join(sorted(missing))}"
            )
        yield from reader


def indexed_rows(path: Path, key: str, required: set[str]) -> dict[str, dict[str, str]]:
    """读取CSV，并以指定列建立唯一记录索引。

    Args:
        path: 待读取的CSV文件路径。
        key: 用作索引键的列名，例如Food_Code或Nutrient_Code。
        required: 除索引列以外还必须存在的列名集合。

    Returns:
        索引值到原始行字典的映射。索引键移除首尾空白，行内容保留原值。

    Raises:
        OSError: 文件无法读取。
        ValueError: 缺少必要列，或索引键为空、重复。

    Notes:
        此函数将整张表加载到内存，适用于食材名称、来源、分类及营养素
        名称等索引表；大型营养含量表由load_cnf_data流式筛选。
    """
    result = {}
    for row in csv_rows(path, {key} | required):
        code = row[key].strip()
        if not code or code in result:
            raise ValueError(f"{path}: empty or duplicate {key}: {code}")
        result[code] = row
    return result


def load_cnf_data(
    directory: str | Path, candidates_path: str | Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按候选清单关联CNF关系表，提取每100g可食部分的营养与来源信息。

    Args:
        directory: 包含food_name.csv、food_source.csv、cnf_food_group.csv、
            nutrient_name.csv及nutrient_amount.csv的CNF数据目录。
        candidates_path: 候选清单JSON路径；省略时读取directory下的
            candidates.json。清单的foods列表需包含25–40个唯一food_code，
            各条目提供name、category、role，以及可选的description_en、
            mix_group。description_en存在且非空时须与原始食材描述一致。

    Returns:
        (foods, metadata)二元组：foods按候选清单顺序保存食材字典，包含
        id、name、category、role、mix_group及per100g，可用于构造Food对象；
        metadata记录官方来源、候选路径、原始食材和营养记录、字段映射
        及所有输入文件的SHA-256校验值。此函数不直接创建Food实例。

    Raises:
        OSError: 已存在的输入文件无法读取。
        ValueError: 文件缺失、JSON无效、候选数量不符、编号或营养记录重复、
            食材描述不匹配、单位不符、必需营养字段缺失，或数值为负、
            非有限值、无法转换为浮点数。
        KeyError: 候选清单或候选条目缺少必要字段。

    Notes:
        营养记录空白或缺省表示未知，数值0保留为真实零。评分必需字段
        缺失时拒绝加载；预成型A与合成叶酸允许为None。现有关系表无法
        完整识别维生素E的UL适用形式与来源，因此ul_applicable_e_mg固定
        保留为None，不以总α-生育酚替代。食材分类和角色取自项目候选
        清单，后续由MealGA检查其取值与组合兼容性。
    """
    directory = Path(directory)
    candidates_path = (
        Path(candidates_path) if candidates_path else directory / "candidates.json"
    )
    files = [
        directory / name
        for name in (
            "food_name.csv",
            "food_source.csv",
            "cnf_food_group.csv",
            "nutrient_name.csv",
            "nutrient_amount.csv",
        )
    ]
    missing = [str(p) for p in [*files, candidates_path] if not p.is_file()]
    if missing:
        raise ValueError(
            "CNF数据文件缺失："
            + ", ".join(missing)
            + "。营养计算需要nutrient_name.csv和nutrient_amount.csv；"
            "下载地址见dataset/README.md，不会回退到演示值。"
        )
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))["foods"]
    if not 25 <= len(candidates) <= 40:
        raise ValueError("CNF candidate list must contain 25–40 foods")
    codes = [str(c["food_code"]) for c in candidates]
    if len(set(codes)) != len(codes):
        raise ValueError("Duplicate Food_Code in CNF candidate list")
    food_names = indexed_rows(
        files[0],
        "Food_Code",
        {"Food_Description_EN", "Food_Source_Code", "CNF_Food_Group_Code"},
    )
    sources = indexed_rows(files[1], "Food_Source_Code", {"Food_Source_Description_EN"})
    groups = indexed_rows(
        files[2], "CNF_Food_Group_Code", {"CNF_Food_Group_Description_EN"}
    )
    nutrient_names = indexed_rows(
        files[3],
        "Nutrient_Code",
        {"Nutrient_Unit", "Nutrient_Symbol", "Nutrient_Name_EN"},
    )
    for field, (code, unit) in NUTRIENT_MAP.items():
        if (
            code not in nutrient_names
            or nutrient_names[code]["Nutrient_Unit"].strip() != unit
        ):
            raise ValueError(
                f"CNF nutrient code/unit mismatch: {field}, expected {code} ({unit})"
            )
    for candidate, code in zip(candidates, codes):
        if code not in food_names:
            raise ValueError(f"Unknown CNF Food_Code: {code}")
        expected = candidate.get("description_en")
        if expected and expected != food_names[code]["Food_Description_EN"]:
            raise ValueError(
                f"CNF description mismatch for {code}: expected {expected!r}"
            )

    selected_values: dict[str, dict[str, float | None]] = {code: {} for code in codes}
    selected_records: dict[str, dict[str, Any]] = {code: {} for code in codes}
    needed_codes = {code for code, _ in NUTRIENT_MAP.values()}
    for row in csv_rows(files[4], {"Food_Code", "Nutrient_Code", "Nutrient_Amount"}):
        food_code, nutrient_code = (
            row["Food_Code"].strip(),
            row["Nutrient_Code"].strip(),
        )
        if food_code not in selected_values or nutrient_code not in needed_codes:
            continue
        if nutrient_code in selected_values[food_code]:
            raise ValueError(
                f"Duplicate CNF nutrient record: {food_code}/{nutrient_code}"
            )
        text = row["Nutrient_Amount"].strip()
        # Blank means missing. A recorded numeric zero remains a real zero.
        value = float(text) if text else None
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(
                f"Invalid CNF nutrient amount: {food_code}/{nutrient_code}: {text}"
            )
        selected_values[food_code][nutrient_code] = value
        selected_records[food_code][nutrient_code] = {
            "amount_per100g": value,
            "source_code": row.get("Nutrient_Source_Code"),
            "last_updated": row.get("Nutrient_Last_Updated_Date"),
        }

    foods, provenance = [], {}
    for candidate, code in zip(candidates, codes):
        values = {
            field: selected_values[code].get(nutrient_code)
            for field, (nutrient_code, _) in NUTRIENT_MAP.items()
        }
        absent = [
            field
            for field, value in values.items()
            if value is None and field not in OPTIONAL_FIELDS
        ]
        if absent:
            raise ValueError(
                f"CNF {code} ({candidate['name']}): missing required nutrients: "
                + ", ".join(absent)
            )
        # Total/added alpha-tocopherol alone does not identify the UL's applicable
        # chemical forms and sources. Never equate it with a checked-safe amount.
        values["ul_applicable_e_mg"] = None
        foods.append(
            {
                "id": code,
                "name": candidate["name"],
                "category": candidate["category"],
                "role": candidate["role"],
                "mix_group": candidate.get("mix_group", "savory"),
                "per100g": values,
            }
        )
        row = food_names[code]
        source_code, group_code = row["Food_Source_Code"], row["CNF_Food_Group_Code"]
        if source_code not in sources or group_code not in groups:
            raise ValueError(f"CNF {code}: missing food source/group record")
        provenance[code] = {
            "description_en": row["Food_Description_EN"],
            "food_source_code": source_code,
            "food_source": sources[source_code]["Food_Source_Description_EN"],
            "cnf_food_group_code": group_code,
            "cnf_food_group": groups[group_code]["CNF_Food_Group_Description_EN"],
            "last_updated": row.get("Food_Last_Updated_Date"),
            "nutrient_records": selected_records[code],
        }
    return foods, {
        "demo_only": False,
        "source": "Canadian Nutrient File 2026",
        "source_url": CNF_URL,
        "weight_basis": "per 100g edible portion; preparation state is given in each CNF description",
        "candidate_file": str(candidates_path),
        "food_records": provenance,
        "nutrient_mapping": {
            field: {"code": code, "unit": unit}
            for field, (code, unit) in NUTRIENT_MAP.items()
        },
        "ul_note": "Vitamin E UL applicable forms/sources are not identifiable from these tables; not assessed.",
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*files, candidates_path]
        },
    }

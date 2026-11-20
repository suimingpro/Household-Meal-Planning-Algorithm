"""Load the CNF 2026 relational tables and convert them into candidate food data for the three-meal planning algorithm.

Join food names, sources, groups, and nutrient amounts using Food_Code from the
candidate list, and validate nutrient codes, units, and record completeness.
This module only reads local data; it does not download files, run the genetic
algorithm, or calculate individual nutrient scores.

Attributes:
    CNF_URL: Official dataset URL, used to record the data source.
    NUTRIENT_SPECS: Standard tags, English names, and expected units for the
        program's nutrient fields. Codes are resolved dynamically from
        nutrient_name.csv. Vitamin A RAE and preformed A, and folate DFE and
        synthetic folic acid, are looked up separately and do not replace one another.
    OPTIONAL_FIELDS: UL-specific preformed A and synthetic folic acid fields
        that may be missing. Missing values remain None rather than becoming 0.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

CNF_URL = "https://open.canada.ca/data/en/dataset/1b6139bd-ed7e-4043-bc28-ff00e10f3109"
# Field: (standard Tagname, exact English name used when the tag does not match, expected unit).
# Only the nutrient meanings required by the algorithm are declared here, not fixed dataset Nutrient_Code values.
NUTRIENT_SPECS = {
    "energy_kcal": ("ENERC_KCAL", "Energy (kilocalories)", "kilocalorie"),
    "carbohydrate_g": ("CHOCDF", "Carbohydrate, total (by difference)", "Gram"),
    "protein_g": ("PROCNT", "Protein", "Gram"),
    "fat_g": ("FAT", "Fat (total lipids)", "Gram"),
    "vitamin_a_rae_ug": ("VITA_RAE", "Retinol activity equivalents", "Microgram"),
    "vitamin_c_mg": ("VITC", "Vitamin C", "Milligram"),
    "vitamin_d_ug": ("VITD", "Vitamin D (D2 + D3)", "Microgram"),
    "vitamin_e_mg": ("TOCPHA", "Alpha-tocopherol", "Milligram"),
    "folate_dfe_ug": ("FOLDFE", "Dietary folate equivalents", "Microgram"),
    "calcium_mg": ("CA", "Calcium", "Milligram"),
    "iron_mg": ("FE", "Iron", "Milligram"),
    "potassium_mg": ("K", "Potassium", "Milligram"),
    "preformed_a_ug": ("RETOL", "Retinol", "Microgram"),
    "synthetic_folic_acid_ug": ("FOLAC", "Folic acid", "Microgram"),
}
OPTIONAL_FIELDS = {"preformed_a_ug", "synthetic_folic_acid_ug"}


def csv_rows(path: Path, required: set[str]):
    """Read a CSV row by row, checking required columns when iteration begins.

    Args:
        path: Path of the CSV file to read; supports UTF-8 BOM and CRLF line endings.
        required: Set of column names that must appear in the header; an empty
            set means no specific columns are required.

    Yields:
        Raw row dictionaries produced by csv.DictReader, keyed by column name,
        without converting nutrient values.

    Raises:
        OSError: The file does not exist, read access is denied, or another file
            reading error occurs.
        ValueError: The header lacks columns specified by required.

    Notes:
        This function is a generator; reading and validation occur during
        iteration. Streaming avoids loading the entire nutrient amount table
        into memory at once, and the file is closed after iteration ends.
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
    """Read a CSV and build a unique-record index using the specified column.

    Args:
        path: Path of the CSV file to read.
        key: Column name used as the index key, such as Food_Code or Nutrient_Code.
        required: Set of column names required in addition to the index column.

    Returns:
        A mapping from index values to raw row dictionaries. Index keys have
        leading and trailing whitespace removed; row contents retain their original values.

    Raises:
        OSError: The file cannot be read.
        ValueError: Required columns are missing, or index keys are empty or duplicated.

    Notes:
        This function loads the entire table into memory and is suitable for
        index tables such as food names, sources, groups, and nutrient names.
        The large nutrient amount table is filtered by streaming in load_cnf_data.
    """
    result = {}
    for row in csv_rows(path, {key} | required):
        code = row[key].strip()
        if not code or code in result:
            raise ValueError(f"{path}: empty or duplicate {key}: {code}")
        result[code] = row
    return result


def _normalize_label(value: str | None) -> str:
    """Ignore case and excess whitespace in tags or names; normalize None to an empty string."""
    return " ".join((value or "").split()).casefold()


def resolve_nutrient_map(
    nutrient_names: dict[str, dict[str, str]],
) -> dict[str, tuple[str, str]]:
    """Dynamically look up nutrient codes by standard tags or exact English names in the data table.

    Args:
        nutrient_names: Raw nutrient records keyed by Nutrient_Code. Each record
            must have Nutrient_Unit and provide Tagname or Nutrient_Name_EN for matching.

    Returns:
        A mapping from program fields to (dataset nutrient code, dataset unit).

    Raises:
        ValueError: Neither tag nor name matches, multiple codes match the same
            item, units differ, or multiple program indicators resolve to the same code.

    Notes:
        Exact Tagname matching takes priority; when no tag matches, fall back
        to the English name. Matching ignores case and excess whitespace and
        does not use substring or fuzzy matching, to avoid confusing RAE with
        Retinol, DFE with Folic acid, or vitamin D in micrograms with IU.
        Units are checked independently after matching.
    """
    resolved = {}
    used_codes = {}
    for field, (tag, name, expected_unit) in NUTRIENT_SPECS.items():
        matches = [
            code for code, row in nutrient_names.items()
            if _normalize_label(row.get("Tagname")) == _normalize_label(tag)
        ]
        if not matches:
            matches = [
                code for code, row in nutrient_names.items()
                if _normalize_label(row.get("Nutrient_Name_EN")) == _normalize_label(name)
            ]
        if not matches:
            raise ValueError(f"CNF nutrient not found: {field}, tag={tag}, name={name}")
        if len(matches) != 1:
            raise ValueError(f"Ambiguous CNF nutrient: {field}, matching codes={matches}")
        code = matches[0]
        unit = nutrient_names[code].get("Nutrient_Unit", "").strip()
        if _normalize_label(unit) != _normalize_label(expected_unit):
            raise ValueError(
                f"CNF nutrient code/unit mismatch: {field}, code={code}, "
                f"expected {expected_unit}, got {unit}"
            )
        if code in used_codes:
            raise ValueError(f"CNF nutrient code reused: {code}, {used_codes[code]} and {field}")
        used_codes[code] = field
        resolved[field] = (code, unit)
    return resolved


def load_cnf_data(
    directory: str | Path, candidates_path: str | Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join CNF relational tables using the candidate list to extract nutrients per 100 g of edible food and source information.

    Args:
        directory: CNF data directory containing food_name.csv, food_source.csv,
            cnf_food_group.csv, nutrient_name.csv, and nutrient_amount.csv.
        candidates_path: Path to the candidate-list JSON; when omitted, read
            candidates.json in directory. The foods list must contain 25–40
            unique food_code values. Each entry provides name, category, role,
            and optional description_en, mix_group, and max_percent.
            max_percent is the upper limit on the side-dish weight proportion,
            defaulting to 100. If description_en is present and nonempty, it
            must match the original food description.

    Returns:
        A (foods, metadata) pair: foods contains food dictionaries in candidate-list
        order, with id, name, category, role, mix_group, and per100g, suitable for
        constructing Food objects. metadata records the official source,
        candidate path, original food and nutrient records, field mappings,
        and SHA-256 checksums of all input files. This function does not directly
        create Food instances.

    Raises:
        OSError: An existing input file cannot be read.
        ValueError: Missing files, invalid JSON, incorrect candidate count,
            duplicate codes or nutrient records, mismatched food descriptions
            or units, missing required nutrient fields, or values that are
            negative, non-finite, or cannot be converted to floats.
        KeyError: Required fields are missing from the candidate list or an entry.

    Notes:
        Nutrient codes are resolved from the nutrient name table first by
        standard tag, then by exact English name; codes are not fixed.
        Blank or omitted nutrient records mean unknown; numeric 0 remains a
        true zero. Loading is rejected when required scoring fields are missing;
        preformed A and synthetic folic acid may be None. The existing relational
        tables cannot fully identify the forms and sources of vitamin E to which
        the UL applies, so ul_applicable_e_mg always remains None and is not
        replaced by total alpha-tocopherol. Food categories and roles come from
        the project's candidate list; MealGA subsequently checks their values
        and combination compatibility.
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
        {"Nutrient_Unit"},
    )
    nutrient_map = resolve_nutrient_map(nutrient_names)
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
    needed_codes = {code for code, _ in nutrient_map.values()}
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
            for field, (nutrient_code, _) in nutrient_map.items()
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
                "max_percent": candidate.get("max_percent", 100),
                "per100g": values,
            }
        )
        row = food_names[code]
        source_code, group_code = row["Food_Source_Code"], row["CNF_Food_Group_Code"]
        if source_code not in sources or group_code not in groups:
            raise ValueError(f"CNF {code}: missing food source/group record")
        provenance[code] = {
            "max_percent": candidate.get("max_percent", 100),
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
            field: {"code": code, "unit": unit,
                    "tag": nutrient_names[code].get("Tagname"),
                    "name": nutrient_names[code].get("Nutrient_Name_EN")}
            for field, (code, unit) in nutrient_map.items()
        },
        "ul_note": "Vitamin E UL applicable forms/sources are not identifiable from these tables; not assessed.",
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*files, candidates_path]
        },
    }

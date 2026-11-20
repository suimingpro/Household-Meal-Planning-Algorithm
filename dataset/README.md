# CNF 2026 Dataset

The program loads food and nutrient data from this directory by default, joining relational tables through `Food_Code` and `Nutrient_Code`. The candidate pool explicitly selects 30 records through `candidates.json` and can be changed to contain 25-40 records.

## Data Sources

[Health Canada: Canadian Nutrient File 2026](https://open.canada.ca/data/en/dataset/1b6139bd-ed7e-4043-bc28-ff00e10f3109). The original CSV files retain their official content and are subject to the Open Government Licence - Canada listed on the dataset page.

Original local files:

- `food_name.csv`: food names and codes.
- `food_source.csv`: food source descriptions.
- `cnf_food_group.csv`: original CNF food groups.

Additional official files:

- [nutrient_name.csv: nutrient codes, names, and units](https://open.canada.ca/data/dataset/1b6139bd-ed7e-4043-bc28-ff00e10f3109/resource/e0aca283-16b5-4eba-a54a-4994490a3262/download/nutrient_name.csv)
- [nutrient_amount.csv: nutrient amounts per 100 g of edible food](https://open.canada.ca/data/dataset/1b6139bd-ed7e-4043-bc28-ff00e10f3109/resource/0ff718fc-1133-4154-80c5-3d619e6c63be/download/nutrient_amount.csv)

These files were downloaded from the official sources on 2026-11-05. The program reads local files at runtime, requires no network access, and does not automatically fall back to demonstration data.

## Candidate List

Each entry contains `food_code`, a Chinese display `name`, the project's `category` and `role`, the original `description_en`, and an optional `mix_group`. Nutrient values are not duplicated in this list; they are read from the original tables by food code. `description_en` verifies that the code matches the intended food and preparation state.

The three default staples are:

| CNF code | Food |
| --- | --- |
| 4523 | Cooked long-grain white rice |
| 4066 | Commercial white bread |
| 4458 | Cooked enriched egg noodles |

The project's five food categories are kept separate from official CNF groups. For example, CNF places potatoes and non-starchy vegetables in the same broad group, while the project's candidate list explicitly assigns potatoes to `starch`. Milk and yogurt are assigned the `standalone` role.

Weights must match the edible state described by the food record. For example, boiled spinach uses its weight after boiling and draining, while raw cucumber uses edible raw weight. The algorithm does not simulate additional cooking losses.

## Fields and Data Integrity

Nutrient requirements are declared in `cnf_data.NUTRIENT_SPECS`, which stores standard tags, exact English names, and expected units rather than fixed numeric codes. During loading, `resolve_nutrient_map()` first matches `Tagname` in `nutrient_name.csv`, falling back to `Nutrient_Name_EN` when no tag matches. Matching ignores case and extra whitespace and does not use fuzzy matching. Missing nutrients, multiple matching codes, or incorrect units cause explicit errors.

For example, protein is identified by `PROCNT` / `Protein`; vitamin A adequacy uses `VITA_RAE`, while its UL uses `RETOL`; folate adequacy uses `FOLDFE`, while its UL uses `FOLAC`; vitamin D uses `VITD` in micrograms rather than `VITD_IU`. The current CNF 2026 data still resolves to codes such as 203, 320, and 319. If codes change consistently in both nutrient tables, the loader does not need to be modified.

Missing required scoring fields cause explicit errors. Recorded zeros are distinguished from missing values. The UL-specific preformed vitamin A and synthetic folic acid fields may be missing, in which case assessment is incomplete. These tables cannot fully identify the forms and sources of vitamin E to which the UL applies, so `ul_applicable_e_mg` remains unknown. Total or added alpha-tocopherol is not substituted for it.

The output's `data_metadata` retains each selected food's CNF description, official group, source, update date, nutrient record source codes, dynamically resolved nutrient codes/units/tags/names, and SHA-256 hashes of the input files for verification and reproducibility.

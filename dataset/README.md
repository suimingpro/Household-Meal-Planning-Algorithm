# CNF 2026 数据集

程序默认从本目录加载食材与营养数据，按 `Food_Code` 和 `Nutrient_Code` 连接关系表。食材候选池由 `candidates.json` 显式选取30个条目，可修改为25–40个条目。

## 数据来源

[Health Canada：Canadian Nutrient File 2026](https://open.canada.ca/data/en/dataset/1b6139bd-ed7e-4043-bc28-ff00e10f3109)。原始CSV保留官方内容，遵循数据集页面列出的 Open Government Licence – Canada。

原有文件：

- `food_name.csv`：食材名称及编号。
- `food_source.csv`：食材来源说明。
- `cnf_food_group.csv`：CNF原始食材类别。

本次补齐的官方文件：

- [nutrient_name.csv：营养素代码、名称与单位](https://open.canada.ca/data/dataset/1b6139bd-ed7e-4043-bc28-ff00e10f3109/resource/e0aca283-16b5-4eba-a54a-4994490a3262/download/nutrient_name.csv)
- [nutrient_amount.csv：食材每100g可食部分的营养含量](https://open.canada.ca/data/dataset/1b6139bd-ed7e-4043-bc28-ff00e10f3109/resource/0ff718fc-1133-4154-80c5-3d619e6c63be/download/nutrient_amount.csv)

这些数据在2026-11-05从官方地址下载。程序运行时只读取本地文件，不需要联网，也不会自动回退到演示数据。

## 候选清单

每个条目包含 `food_code`、中文 `name`、项目 `category`、`role`、原始 `description_en`，以及可选的 `mix_group`。营养值不在候选清单中重复保存；代码按食材编号从原表读取。`description_en` 用于检查编号与预期食材是否一致，防止生熟状态或条目选错。

默认三种主食为：

| CNF编号 | 食材 |
| --- | --- |
| 4523 | 熟长粒白米饭 |
| 4066 | 市售白面包 |
| 4458 | 强化熟鸡蛋面 |

项目的五种食材分类与CNF官方分组分别保留，不将二者直接等同。例如，CNF将土豆与非淀粉蔬菜放在同一个大组，项目清单则显式将土豆设为 `starch`。牛奶、酸奶设为 `standalone`。

所有重量均匹配条目所述的可食状态：例如水煮菠菜用水煮沥干后的重量，生黄瓜用可食生重。算法不会模拟额外烹调损失。

## 字段与完整性

完整映射见 `cnf_data.NUTRIENT_MAP`。A达标使用320（RAE），A上限使用319（Retinol）；叶酸达标使用435（DFE），上限使用431（Folic acid）；D使用328（μg），不误用324（IU）。

必需的评分字段缺失时明确报错，数值0与缺失值分开处理。UL专用的预成型A和合成叶酸允许缺失并标记未能评估。当前文件不能完整识别维生素E的UL适用形式与来源，因此 `ul_applicable_e_mg` 保留未知；不会将总α-生育酚或添加α-生育酚直接替代它。

结果的 `data_metadata` 保存每个入选条目的CNF名称、官方分类、来源、更新日期、营养记录来源代码、营养素映射及输入文件SHA-256，便于核对与复现。

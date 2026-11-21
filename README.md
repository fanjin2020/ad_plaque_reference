# AD Plaque Reference Audit

Reproducible analysis and figure generation for estimating how biological reference choice affects selective protein enrichment in donor-paired human Alzheimer’s disease plaque proteomics.

本仓库配套论文 *Biological Reference Choice Shapes Estimates of Selective Protein Enrichment in Human Alzheimer Plaques*。流程从已发表的蛋白质定量工作簿开始，重建同一供体的斑块与邻近组织配对，计算不同生物学参照下的相对效应，并复算主稿和补充材料所用的数值与图件。

## 运行入口 / Workflows

| 入口 | 分析内容 | 默认结果目录 |
| --- | --- | --- |
| `run.py` | 斑块主分析、覆盖审计、丰度匹配、基于实测协方差的模拟，以及 v9 基础图表 | `results/` |
| `run_v10.py` | 外部 tau 缠结应用、SSR 留出仿真、贝叶斯参照平均、GLS 权重和 CoDA 数值核对 | `results_v10/` |
| `paper4.v10.publication_followup` | 冻结面板的 MCI 比较、MCI–AD 交互、背景蛋白三态判定和稳健权重比较 | 通过 `--out` 指定 |
| `paper4.v10.manuscript_figures` | 用前三步的结果生成 v10 主图、目录图和 4 张修订或新增补充图 | 通过 `--out` 指定 |

斑块研究按供体计算配对的组织强度差，再计算靶蛋白平均变化减去参照蛋白平均变化。覆盖表保留缺少完整配对的比较。各研究组单独估计。tau 缠结来自独立发表的资料，用于检验流程在另一类病变中的应用。[v10 分析定义与结果字段](docs/v10_experiments.md)

## 环境 / Requirements

- 使用 Python **3.11、3.12 或 3.13**。两个根目录入口读取 [requirements.txt](requirements.txt)。现用环境已满足锁定版本时直接运行，否则在仓库下创建 `.paper4-env/` 并安装依赖。
- 在新环境中首次安装需要能访问 Python 软件包源。联网模式的分析也需要能访问数据清单中的出版社和 PDB。已备齐数据时可指定 `--input-dir`。
- `--workers` 接受 1–16 个模拟工作进程，默认最多 4 个。桌面机可先用 2 个，内存紧张时用 1 个。脚本用 CPU 计算，批量绘图可在无显示器的 Linux 服务器运行。
- 输出所在目录须可写。临时工作目录与输出位于同一文件系统，分析和校验通过后才生成最终输出目录。两个根目录入口遇到同名输出时会自动加时间戳。

## 从头运行 / Run from source

在仓库根目录操作，依次完成以下四步即可复算 v10 主稿的分析和图件。前两个步骤会下载并核查公开输入。若安装和开始阶段通过，终端随后会显示分析阶段，最后打印实际结果路径。

### Linux 或 macOS

~~~bash
python3 run.py --workers 2
python3 run_v10.py --workers 2

paper4_python=.paper4-env/bin/python
[ -x "$paper4_python" ] || paper4_python=python3

"$paper4_python" -m paper4.v10.publication_followup --out results_followup

"$paper4_python" -m paper4.v10.manuscript_figures \
  --base results/Supporting_Data.zip \
  --followup results_followup \
  --v10 results_v10 \
  --out results_manuscript_v10
~~~

### Windows PowerShell

~~~powershell
python run.py --workers 2
python run_v10.py --workers 2

$paper4Python = './.paper4-env/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $paper4Python)) { $paper4Python = 'python' }

& $paper4Python -m paper4.v10.publication_followup --out results_followup

& $paper4Python -m paper4.v10.manuscript_figures --base results/Supporting_Data.zip --followup results_followup --v10 results_v10 --out results_manuscript_v10
~~~

如果前两步因目录已存在而使用带时间戳的路径，请在后续命令中代入终端打印的新路径。`publication_followup` 要求 `--out` 为新目录。重新比较图件时也建议另选输出目录。

## 复用本地输入 / Local inputs

按下表把下载的公开文件放在仓库之外的同一目录。使用代码清单中的本地文件名。脚本核对字节数和 SHA-256，确保使用论文对应的文件版本。完整七份输入合计约 **32.4 MB**，每个阶段只读取需要的文件。分阶段使用默认联网模式时，公共数据可能下载多次。

| 本地文件名 | 用途 | 使用步骤 |
| --- | --- | --- |
| `leitner_2026_S5.xlsx` | 人脑 AD、MCI 和对照组织定量工作簿 | 基础分析、v10 实验、MCI 与判定迁移 |
| `marta_2025_S10.xlsx` | EOAD、LOAD 和唐氏综合征斑块定量 | 基础分析、v10 实验、MCI 与判定迁移 |
| `drummond2022_plaque_supp.xlsx` | 早发 AD 与唐氏综合征斑块定量 | 基础分析、v10 实验、MCI 与判定迁移 |
| `leitner_2024_caa_S2.xlsx` | 脑淀粉样血管病对比 | 基础分析 |
| `xiong_2019_table1.xlsx` | 已发表的组级斑块结果 | 基础分析 |
| `6WM2.cif.gz` | 人源 V-ATPase 冷冻电镜结构 | 基础分析和图件 |
| `thierry_2026_tangle_R1.xlsx` | tau 缠结与邻近神经毡的配对定量 | v10 实验 |

各输入的原文、精确下载地址与校验值记录在 [基础输入清单](paper4/data_manifest.json) 和 [v10 方案](paper4/v10/protocol.json) 的 `external_source` 中。

以下以 Linux 为例，前两步使用同一个本地目录。Windows 把 `python3` 换成 `python`，并使用上节的 `$paper4Python` 调用子模块。

~~~bash
inputs="$HOME/paper4_inputs"
python3 run.py --input-dir "$inputs" --workers 2
python3 run_v10.py --input-dir "$inputs" --workers 2
"$paper4_python" -m paper4.v10.publication_followup --input-dir "$inputs" --out results_followup
"$paper4_python" -m paper4.v10.manuscript_figures --base results/Supporting_Data.zip --followup results_followup --v10 results_v10 --structure "$inputs/6WM2.cif.gz" --out results_manuscript_v10
~~~

上面的 `paper4_python` 使用“从头运行”中的解释器选择方式。绘图时省略 `--structure` 会自动从 PDB 下载并校验坐标文件。已有本地坐标时指定该参数即可复用。

## 看进度与校验 / Monitor and verify

服务器可用后台方式启动单个阶段。下面以 v10 实验为例，命令从仓库根目录执行。

~~~bash
nohup python3 -u run_v10.py --input-dir "$HOME/paper4_inputs" --out "$HOME/paper4_v10_results" --workers 2 > "$HOME/paper4_v10.log" 2>&1 &
tail -f "$HOME/paper4_v10.log"
~~~

日志出现 `PASS` 后，用实际输出目录核对文件。上例的命令是 `python3 run_v10.py --check "$HOME/paper4_v10_results"`。若运行仍在计算，应先看日志阶段，完成目录出现后再执行检查。

~~~bash
python3 run.py --check results
python3 run_v10.py --check results_v10
python3 -c "from pathlib import Path; from paper4.source_inputs import verify_output; print(verify_output(Path('results_followup')))"
~~~

Windows 用 `python` 运行同样的检查。各分析阶段的 `verification.json` 记录校验状态和逐文件 SHA-256。主分析和 v10 实验还记录输入版本、运行环境、预计结果的数值核对及实际耗时。完整复算的耗时随硬件和并行进程数变化。本次在 8 GB Windows 机器使用 Python 3.13、2 个工作进程和校验过的本地输入，基础流程约 16 分 27 秒，v10 五项实验约 8 分 32 秒。首次依赖安装另计。

本次 83 张基础补充表与最终稿归档的数值在浮点容差内一致，7 张新增分析 CSV 与归档逐字节一致。Windows 与 Linux 的 v10 汇总结果一致。近奇异协方差矩阵的条件数会随线性代数实现变化，本次差异涉及的条件数均远高于 `10¹⁰`，两次运行都将相应的无约束逆协方差估计判为不可用。

## 结果对照 / Outputs

| 结果 | 先看什么 |
| --- | --- |
| `results/Figures.pdf` | v9 基础图册，包含 8 张主图、17 张补充图和目录图 |
| `results/Supporting_Data.zip` | S1–S83、Data S1、Data S2，以及包含每个表名和行数的索引 |
| `results/tables/`、`results/paired_measurements.csv.gz` | 3 张正文表、3 张补充导读表和从原工作簿重建的逐供体定量 |
| `results_v10/Summary.md` | 五项 v10 实验的主要结果、各组配对数和模拟分母 |
| `results_v10/tables/`、`results_v10/ssr_repeats.csv.gz` | 逐供体效应、场景级性能、贝叶斯与 GLS 结果、CoDA 核查和逐次模拟结果 |
| `results_v10/figures/` | 三张 v10 实验图，对应最终稿补充图 S18–S20 的分析 |
| `results_followup/` | 7 张新增 CSV 和结果校验记录。S84–S88 的 CSV 对应关系见下表 |
| `results_manuscript_v10/figures/` | 最终稿的 8 张主图和 TOC |
| `results_manuscript_v10/supporting/figures/` | S16、S18、S19、S20。其余补充图使用 `results/figures/supplement/` 中的对应图件 |

| 最终稿表号 | `results_followup/` 文件 |
| --- | --- |
| S84 | `phenotype_estimates.csv`，冻结面板在 MCI 和 AD 的比较 |
| S85 | `phenotype_stage_interaction.csv`，两组更换参照所产生的效应差 |
| S86 | `background_decisions.csv`，点估计换向和同时区间支持的方向 |
| S87 | `panel_transitions.csv`，支持正向、未决和支持反向的迁移 |
| S88 | `robust_weights.csv`，等权、中位数和截尾均值的敏感性比较 |

绘图脚本为每张图输出 PDF、PNG 和 TIFF。图 1A 和 S16 的虚拟切片用于展示取样与参照选择，图 1B 和 S17 使用公开 PDB 6WM2 坐标。模拟错误率图和人脑组织的配对实测图分别使用各自的结果表。[基础图件来源说明](docs/逐图数据来源与用途.md)

## 常见问题 / Troubleshooting

| 提示或现象 | 处理方式 |
| --- | --- |
| `Use Python 3.11, 3.12 or 3.13` | 检查 `python --version` 或 `python3 --version`，再用支持的解释器运行。 |
| `Could not install the locked Python packages` | 检查命令输出中的 pip 错误、软件包源连通性和该 Python 版本对应的二进制包。Linux 上如果错误来自 `venv` 或 `ensurepip`，补齐相应 Python 安装中的虚拟环境组件。 |
| `Public download failed` | 按数据清单从可访问的机器获取同版本文件，再用 `--input-dir`。绘图的 PDB 坐标用 `--structure`。 |
| `Input size or SHA-256 mismatch` | 检查文件是否下载完整、文件名是否与清单一致、文件是否曾被电子表格软件重新保存。重新取得清单对应的原始附件。 |
| `FileNotFoundError` 指向 `verification.json` | `--check` 要接收已完成目录本身。用终端最后打印的路径，必要时查看带时间戳的同名目录。 |
| `Output file manifest differs` | 根据错误列出的文件名，确认是否有输出被改动、删减或额外添加。要编辑表格时，把副本保存到校验目录之外。 |

进程报错并需要完整追踪时，可在 Linux 命令前加 `PAPER4_DEBUG=1`，或在 PowerShell 中先运行 `$env:PAPER4_DEBUG = '1'`。源码按 [公开输入清单](paper4/data_manifest.json)、[冻结参照协议](paper4/reference_extensions/data/protocol.json) 和 [v10 实验方案](paper4/v10/protocol.json) 记录数据与分析版本。引用复用数据时请同时引用清单收录的原始研究。


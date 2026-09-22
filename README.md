# pdbenergy — 用 PyTorch 学习蛋白质构象能量的神经网络代理模型

给定一个 PDB 结构文件，直接输出它的**构象能量**（kcal/mol），不需要跑分子动力学、
不需要力场参数化。物理力场（AMBER14 + GBn2 隐式溶剂，通过 OpenMM）只用来**生成训练标签**，
神经网络学会复现它。

> 想系统学习这个项目用到的全部知识，请阅读 **[`TeachFlow.md`](TeachFlow.md)**。
> 那是一份从 PDB 文件格式、力场物理、构象采样，一直讲到图神经网络、PyTorch 训练细节
> 和评估方法学的完整教程。
>
> 想快速判断"这是什么、做到什么程度、值不值得用"，请看 **[`DESCRIPTION.md`](DESCRIPTION.md)**。

---

## 它到底在预测什么

| 概念 | 说明 |
|---|---|
| 输入 | 一个蛋白质构象（PDB 坐标 + 元素类型），**含氢**的全原子结构 |
| 输出 | 该构象的相对构象能 `ΔE = E - min(E)`，单位 kcal/mol |
| 标签来源 | OpenMM 的 `amber14-all.xml` + `implicit/gbn2.xml` 力场能量 |
| 为什么是"相对" | 绝对能量主要由原子组成和长程溶剂化决定；跨分子比较没有意义，同一分子的不同构象比较才有意义 |

## 安装

```powershell
git clone https://github.com/Yore-ASH/pdbenergy.git && cd pdbenergy
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[physics]"
# CPU 版 PyTorch（Windows 上 PyPI 的 torch 就是 CPU 版；Linux 上需要显式指定）
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`pip install -e .` 会注册一个 `pdbenergy` 命令，所以下面所有
`python -m pdbenergy.cli <子命令>` 都可以写成 `pdbenergy <子命令>`。

> `data/interim/` 里已经包含 14 个蛋白、730 个构象的**已标注数据集**（约 3.8 MB），
> 所以你可以**跳过最耗时的物理打标签步骤**，直接 `dataset` → `train`。

## 三种调用方式

### 方式一：命令行（日常使用）

| 子命令 | 作用 | 典型耗时 |
|---|---|---|
| `download` | 从 RCSB 下载 PDB 结构 | 秒级 |
| `inventory` | 体检：残基数、链数、模型数、是否可用 | 秒级 |
| `ensemble` | 生成构象系综并用力场打标签 | **最慢，约 2–3 分钟/蛋白** |
| `dataset` | 按蛋白质切分 + 目标标准化 | 秒级 |
| `train` | 训练（`--model schnet` 或 `mlp`） | Schnet 约 100 秒/轮 |
| `evaluate` | 指标 + 全部图表 | 约 1 分钟 |
| `predict` | PDB 文件 → 能量（`--verify` 附真值） | 秒级（含加氢） |
| `ablate` | 数据泄漏消融 | 分钟级 |
| `all` | 一步到底 | 视数据量而定 |

```powershell
# 完整流程
pdbenergy download
pdbenergy inventory
pdbenergy ensemble --threads 8
pdbenergy dataset
pdbenergy train --model schnet
pdbenergy evaluate --run-dir outputs/schnet_protein

# 直接用自己的结构预测
pdbenergy predict my_structure.pdb --verify
pdbenergy predict nmr_ensemble.pdb --max-models 38     # 给整个 NMR 系综打分并排序
```

常用开关：`--preset quick|default|thorough`、`--config configs/quick.json`、
`--ids 1CRN 1L2Y`（只处理指定条目）、`--make` 见 `pdbenergy <子命令> --help`。

### 方式二：现成脚本

```powershell
python examples\predict_pdb.py data\raw\1L2Y.pdb --models 3 --verify   # 打分 + 排序 + 对照真值
python examples\train_from_python.py --epochs 5                        # 用 Python API 训练
.\scripts\run_pipeline.ps1 -SkipEnsemble                               # 全流程 + 分阶段日志
python scripts\measure_leakage.py                                      # 泄漏直接测量
python scripts\write_results.py                                        # 由产物重生成 TeachFlow 第 11 章
```

### 方式三：作为库调用

```python
from pdbenergy.config import Config
from pdbenergy.dataset import build_bundle, load_ensembles
from pdbenergy.train import train_model
from pdbenergy.predict import EnergyPredictor

cfg = Config()
cfg.model.hidden_dim = 128          # 所有超参数都是 dataclass 字段
cfg.features.cutoff = 4.5
cfg.train.epochs = 100

ensembles = load_ensembles("data/interim")
bundle = build_bundle(ensembles, cfg)        # 按蛋白切分，无泄漏
result = train_model(bundle, out_dir="outputs/my_run")

predictor = EnergyPredictor("outputs/my_run/checkpoint.pt")
for pred in predictor.predict_file("my_structure.pdb"):
    print(pred.predicted_relative_energy, "kcal/mol")
```

更多例子见 [`examples/`](examples/)。

## 迭代训练：加数据，看精度真的提升

单次训练回答"这能不能跑"；迭代训练回答"数据变多了，模型有没有变好，怎么看得见"。

```powershell
# 把新的 PDB 放进 data\raw\，然后跑一轮（会自动打标签 + 重训 + 评估 + 记录）
python -m pdbenergy.iterate --rounds 1 --build-new --threads 8

# 继续加、继续跑；会自动从上一轮的权重热启动
python -m pdbenergy.iterate --rounds 3 --build-new --threads 8

# 只看累计的轮次表格
python -m pdbenergy.iterate --report
```

它替你处理了两件**很容易骗到自己**的事：

1. **冻结切分** —— 第 1 轮把 train/val/test 的蛋白固定进 `split.json`，
   之后新增的蛋白**只进 train**。否则测试集每轮都在变，
   "test 指标变好了"可能只是因为某个难蛋白被换出去了。
2. **只用验证集选模型** —— `best/` 里放的是验证 MAE 最好的那一轮；
   测试指标只报告、不参与选择。

产物在 `outputs/iterative/`：`rounds.jsonl`（机器可读）、`rounds.md`（表格）、
`round_001/…`（每轮检查点、历史、优化器状态）、`best/`。

### 三种"继续训练"的区别

| 命令 | 优化器状态 | epoch 计数 | 用在哪 |
|---|---|---|---|
| `train` | 全新 | 从 1 开始 | 从头训练 |
| `train --init-from <ckpt>` | 全新 | 从 1 开始 | **数据变多了，在旧模型基础上提升** |
| `train --resume <run_dir>` | **恢复** | **接着数** | 长时间训练被中断，接着跑 |
| `python -m pdbenergy.iterate` | 自动 | 自动 | 多轮"加数据 → 重训 → 比较"的完整循环 |

`--init-from` 默认**沿用检查点里的目标归一化**（mean/std）：换个尺度会让输出层一开始
就标定错，前几轮全花在把尺度掰回来。新蛋白分布确实不同时加 `--recompute-normalisation`。

⚠️ **改了网络结构就不能热启动了**（`hidden_dim` / `n_interactions` / `cutoff` / `n_rbf`
都会改变张量形状）。代码会直接报错并指出是哪个字段变了，不会悄悄只加载一半。
数据量上去之后想换大模型，就换个 `--out-dir` 从头训——**数据够了，大模型才开始值钱**。

### 扩大数据集时要盯的三件事

| 限制 | 默认值 | 超过会怎样 | 怎么办 |
|---|---|---|---|
| `prepare.max_residues` | 120 | 大蛋白被 `inventory` 标为不可用、`ensemble` 跳过 | 调大它；注意大蛋白构建时间超线性增长 |
| 图缓存内存 | 开启 | 实测 **0.5 MB/帧** 全部读进内存：2 万帧约 10 GB，10 万帧放不下 | 超约 2 万帧时用 `--no-cache-graphs`，改成在线特征化 + `--num-workers` |
| 并行度 | `--workers 1` | `workers × threads` 超过物理核数会互相抢核，反而更慢 | 4 核 8 线程的机器用 `--workers 4 --threads 2` |

时间上，实测单蛋白打标签 **2–4 分钟**（8 线程），所以 100 个蛋白是几小时量级——
**分批做、中断了接着跑**：已完成的蛋白写成 `data/interim/<ID>.npz`，重跑会自动跳过。

数据规模上来之后，比调容量更值得先试的是 **RBF 分辨率**
（`features.n_rbf` 从 32 提到 64~96）——详见 `TeachFlow.md` §12.3 的诊断。

## 项目结构

```
pdbenergy/
  pdbio.py       PDB 固定列格式的解析与写出（零依赖）
  prepare.py     从 RCSB 下载、扫描与体检
  labels.py      OpenMM 力场引擎：加氢、参数化、单点能、能量分解、最小化、MD
  ensemble.py    构象系综生成：二面角旋转 + NMR 模型 + 朗之万 MD + 冲突筛选
  dataset.py     数据装配、按蛋白切分、目标标准化
  features.py    结构 → 图（距离边特征、共价键标记、基线描述符）
  graphcache.py  图缓存（把每轮重复的建图开销消掉）
  models.py      SchNet 消息传递网络 + MLP 基线
  train.py       训练循环、早停、检查点、指标
  evaluate.py    指标、分组分解、排序能力、全部图表
  leakage.py     不依赖模型的数据泄漏直接测量
  predict.py     单文件推理：PDB → 能量
  cli.py         命令行入口
examples/        可运行示例（命令行 / 脚本 / 库三种用法）
configs/         默认与快速配置（`--config configs/quick.json`）
scripts/         run_pipeline.ps1（一键全流程）、write_results.py（从产物生成本文档结果章节）
benchmarks/      物理与模型开销的实测脚本，以及实测数据表
tests/           单元测试（PDB 解析、对称性、置换不变性、力场自洽性）
outputs/         检查点、指标、图
```

## 测试

```powershell
# 全部测试（物理测试需要 OpenMM，约 4 分钟）
python -m unittest discover -s tests -v

# 只跑不需要 OpenMM 的部分（秒级）
python -m unittest tests.test_pdbio tests.test_ml -v
```

CI（[`.github/workflows/ci.yml`](.github/workflows/ci.yml)）分两个 job：
`tests` 在 Python 3.11 / 3.13 上跑解析与机器学习测试（无 OpenMM、无网络），
`physics` 安装 OpenMM 并下载 `1CRN` 后跑力场自洽性测试。
两个 job 都使用 CPU 版 PyTorch，避免把 2.5 GB 的 CUDA 轮子拖进 CI。

测试覆盖了几个容易被忽略的性质：

* `TER` 记录**只在链结束处**出现（每残基一个 `TER` 会破坏模板匹配）；
* 原子顺序置换、整体旋转平移**不改变**模型输出；
* 二面角旋转**精确保持**所有键长与键角；
* 能量分解的四项之**和等于**总能量；
* 按蛋白质切分是**互斥且完整**的，且对同一种子可复现。

## 主要结果（诚实版）

数据集：14 个蛋白、730 个构象（力场标签为 AMBER14+GBn2），按**蛋白质**切分为
6 训练 / 4 验证 / 4 测试，测试集蛋白在训练中从未出现。

| 模型 | 参数量 | 测试 MAE | 测试 Spearman | 组内平均 ρ | 常数基线 MAE |
|---|---|---|---|---|---|
| SchNet（图神经网络） | 42,562 | 121.06 | **0.431** | 0.147 | 107.29 |
| MLP（手工描述符基线） | 9,985 | 105.97 | — | — | 107.29 |

单位 kcal/mol。**结论：流水线完整可用，但模型还不适合当打分函数。**

* ✅ **能跑通**：PDB 下载 → 力场打标签 → 构象采样 → 按蛋白切分 → 训练 → 评估 → 推理；
* ✅ **带真实排序信号**：全局 Spearman 0.43；在真实的 `1L2Y.pdb` 上，
  模型把三个 NMR 模型排成了与真值一致的能量顺序；
* ❌ **绝对精度不够**：预测区间只有 84–186 kcal/mol，而真实跨度是 0–709，
  **只覆盖了 14%**，是典型的"回归到均值"；MAE 没有超过常数基线；
* ❌ **高能区最差**：`md450K` 帧的 MAE 是 `torsion` 帧的 5.7 倍。

原因与改进方向见 [`TeachFlow.md`](TeachFlow.md) 第 11 章的自动诊断与第 12.3 节
（诊断指向：RBF 特征分辨率 0.161 Å 粗于键长热涨落 0.05–0.1 Å；以及数据量不足）。

### 数据泄漏消融（不需要训练模型）

| 切分方式 | 测试帧最近邻同蛋白比例 | 最近邻距离中位数 | 近似重复比例 |
|---|---|---|---|
| 按蛋白质切分（正确） | **0.0%** | 4.775 | 0.0% |
| 按帧随机切分（常见错误） | **97.3%** | 0.044 | **52.7%** |

按帧随机切分时，**一半以上的"测试集"是训练集的近似副本**。
用 MAE 差距去论证泄漏是间接的（取决于模型容量），这张表是直接证据。

```powershell
python scripts\measure_leakage.py      # 复现上表
python scripts\write_results.py        # 由产物重新生成 TeachFlow 第 11 章
```

## 复现性

所有随机源（Python / NumPy / Torch / 采样器）都由 `--seed` 控制；
检查点里同时保存了模型权重、配置、归一化统计量和切分清单，
所以单独一个 `checkpoint.pt` 就足以自解释地做推理。

# pdbenergy — 用 PyTorch 学习蛋白质构象能量的神经网络代理模型

给定一个 PDB 结构文件，直接输出它的**构象能量**（kcal/mol），不需要跑分子动力学、
不需要力场参数化。物理力场（AMBER14 + GBn2 隐式溶剂，通过 OpenMM）只用来**生成训练标签**，
神经网络学会复现它。

> 想系统学习这个项目用到的全部知识，请阅读 **[`TeachFlow.md`](TeachFlow.md)**。
> 那是一份从 PDB 文件格式、力场物理、构象采样，一直讲到图神经网络、PyTorch 训练细节
> 和评估方法学的完整教程。

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
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# CPU 版 PyTorch（本项目在 CPU 上训练，几百万参数、上千样本完全够用）
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 快速开始

```powershell
# 1) 下载 PDB 结构（内置 16 个小蛋白，X-ray + NMR）
python -m pdbenergy.cli download

# 2) 先看看数据里到底有什么
python -m pdbenergy.cli inventory

# 3) 生成构象系综并用物理力场打标签（最耗时的一步，约 30 分钟/16 个蛋白）
python -m pdbenergy.cli ensemble --threads 8

# 4) 切分数据集（按蛋白质切分，避免泄漏）
python -m pdbenergy.cli dataset

# 5) 训练图神经网络 + 手工描述符基线
python -m pdbenergy.cli train --model schnet
python -m pdbenergy.cli train --model mlp

# 6) 评估：指标 + 图
python -m pdbenergy.cli evaluate --run-dir outputs/schnet_protein

# 7) 预测你自己的 PDB 文件
python -m pdbenergy.cli predict data/raw/1CRN.pdb --verify
```

或者一步到底：

```powershell
python -m pdbenergy.cli all
```

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
  predict.py     单文件推理：PDB → 能量
  cli.py         命令行入口
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

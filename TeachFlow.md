# TeachFlow — 从 PDB 文件到构象能量

> 这份文档是这个项目所用到的**全部知识**的自学教程。
> 目标读者：会写一点 Python、听说过神经网络、但没有做过计算化学或分子机器学习的人。
> 读完你应该能：看懂本项目每一行代码的物理含义、自己造一个类似的分子数据集、
> 训练并**正确地评估**一个分子性质预测模型。
>
> 阅读方式：第 1–5 章是"物理与数据"（能量从哪来、数据怎么造），
> 第 6–8 章是"机器学习"（结构怎么变成图、网络怎么工作、怎么训练），
> 第 9 章是"评估方法学"（最容易骗自己的地方，**强烈建议不要跳过**），
> 第 10–12 章是代码导读、实测结果与局限。

---

## 目录

| 章 | 内容 |
|---|---|
| 1 | [问题定义：什么叫"构象能量"](#1-问题定义什么叫构象能量) |
| 2 | [蛋白质结构基础](#2-蛋白质结构基础) |
| 3 | [PDB 文件格式与它的所有坑](#3-pdb-文件格式与它的所有坑) |
| 4 | [分子力学力场：标签的物理来源](#4-分子力学力场标签的物理来源) |
| 5 | [构象采样：数据集是怎么造出来的](#5-构象采样数据集是怎么造出来的) |
| 6 | [把分子变成图：机器学习的表示](#6-把分子变成图机器学习的表示) |
| 7 | [图神经网络与 SchNet](#7-图神经网络与-schnet) |
| 8 | [PyTorch 训练实践](#8-pytorch-训练实践) |
| 9 | [评估方法学：数据泄漏与指标陷阱](#9-评估方法学数据泄漏与指标陷阱) |
| 10 | [代码导读](#10-代码导读) |
| 11 | [实测结果](#11-实测结果) |
| 12 | [局限、失败模式与后续方向](#12-局限失败模式与后续方向) |
| 13 | [术语表与延伸阅读](#13-术语表与延伸阅读) |

---

## 1. 问题定义：什么叫"构象能量"

### 1.1 一个分子有很多种形状

蛋白质是一条由氨基酸串成的链。这条链上的单键可以旋转，所以同一个蛋白质可以有
天文数字量级的**不同三维形状**——每一个叫做一个**构象（conformation）**。

"构象"和"构型（configuration）"要区分开：构象变化不需要打断任何化学键，
只是绕着单键转；构型变化要断键。我们这里讨论的全是构象。

### 1.2 势能面（PES）

把分子的所有原子坐标记作 $\mathbf{R} \in \mathbb{R}^{3N}$，那么它的势能是坐标的函数：

$$E = E(\mathbf{R})$$

这个函数叫做**势能面（Potential Energy Surface, PES）**。它是 $3N$ 维空间里的一个曲面，
有山峰（高能、不稳定）、有山谷（局部极小、亚稳态）、有全局最低点（通常对应天然结构）。

在**玻恩–奥本海默近似**下（见 §4.1），电子"瞬间"跟上原子核的运动，
所以对每一个原子核构型都有一个确定的电子能量，$E(\mathbf{R})$ 这个函数是良定义的。

本项目做的就是：**用神经网络拟合 $E(\mathbf{R})$ 这个函数。**

### 1.3 为什么需要神经网络代理模型

$E(\mathbf{R})$ 可以用量子化学精确算，但代价极高（一个几十残基的蛋白，DFT 单点能
基本不可能）。实践中用**分子力学力场**（§4）近似，代价低得多，但依然是：

* 一次单点能评估要遍历所有原子对；
* 要探索构象空间就要跑分子动力学（MD），每一步都要算一次力（即能量的梯度）；
* 隐式溶剂模型让代价从 $O(N)$ 涨到 $O(N^2)$ 甚至更贵。

这个项目在 CPU 上的实测（1CRN，642 原子，AMBER14+GBn2）：

| 操作 | 无截断 | 1.0 nm 截断 |
|---|---|---|
| 单点能 | 65.6 ms | 22.0 ms |
| MD 每一步 | 80.5 ms | 22.2 ms |
| L-BFGS 最小化 50 步 | 23.8 s | 6.1 s |

而训练好的神经网络做一次预测是 **毫秒量级**，且可以做**批量并行**、可做**自动微分**
（从而免费得到力，用于几何优化或 MD，这是后续可以扩展的方向）。
这就是所谓的 **MLIP / ML potential / 机器学习势函数** 思路，
代表性工作有 ANI-1、SchNet、NequIP、MACE 等（见 §13）。

### 1.4 相对能量才是可学的、有意义的量

一个蛋白的**绝对**势能大约是 $-1500$ kcal/mol 这种量级，其中绝大部分来自
原子组成（多少个 C、N、O、H）和长程静电/溶剂化。换一个蛋白，这个数完全不可比。

真正有物理含义、也真正影响构象分布的是**相对能量**：

$$\Delta E = E(\mathbf{R}) - \min_{\mathbf{R}' \in \text{该蛋白}} E(\mathbf{R}')$$

本项目默认就预测这个 $\Delta E$（配置项 `train.target = "relative"`）。
含义是："这个构象比这个蛋白最舒服的构象差多少 kcal/mol"。
这正是打分函数（scoring function）关心的问题：
给定同一个蛋白的一堆候选构象，**能不能把好的排在前面**。

> 代码位置：`pdbenergy/dataset.py::DataBundle.raw_target`

### 1.5 本项目端到端流程

```
 RCSB PDB 文件 (data/raw/*.pdb)
        │  ① 下载 + 体检                    prepare.py
        ▼
 干净的全原子结构（加氢、补原子）           labels.py::protonate
        │  ② 力场参数化 + 能量最小化
        ▼
 OpenMM System + 参考能量 E_min             labels.py::prepare
        │  ③ 构象采样：二面角旋转 / NMR / MD
        ▼
 (构象坐标, ΔE) 样本对                      ensemble.py
        │  ④ 按蛋白切分 + 目标标准化
        ▼
 train / val / test                          dataset.py
        │  ⑤ 结构 → 图（RBF 距离 + 共价键标记）
        ▼
 图数据                                      features.py
        │  ⑥ SchNet 消息传递 + 求和池化
        ▼
 预测 ΔE                                     models.py / train.py
        │  ⑦ 指标、分组分解、排序能力、图表
        ▼
 评估报告                                    evaluate.py
```

---

## 2. 蛋白质结构基础

要读懂 PDB 文件和处理蛋白质，这几个概念是必须的。

### 2.1 氨基酸与肽链

蛋白质由 20 种**标准氨基酸**组成。每个氨基酸有：

* 一个**主链（backbone）**：`N`–`Cα`–`C`–`O`（还有酰胺氢 `H`、`Hα`）；
* 一个**侧链（side chain）**：从 `Cα` 上长出去的 `Cβ`、`Cγ`……不同氨基酸不同。

氨基酸之间通过**肽键**（前一个残基的 `C` 与后一个残基的 `N`）连成链。
注意三字母码和单字母码的对应，代码里在 `pdbenergy/pdbio.py::AA3_TO_1`。

### 2.2 二面角：构象空间的自然坐标

四个原子 A–B–C–D 绕 B–C 键的**二面角（dihedral / torsion angle）**
定义为：把 A 和 D 投影到垂直于 B–C 的平面上，两个投影之间的夹角。

蛋白质里最重要的几个：

| 角度 | 四个原子 | 含义 |
|---|---|---|
| $\phi$ (phi) | C(i-1)–N(i)–Cα(i)–C(i) | 主链骨架旋转 |
| $\psi$ (psi) | N(i)–Cα(i)–C(i)–N(i+1) | 主链骨架旋转 |
| $\omega$ (omega) | Cα(i)–C(i)–N(i+1)–Cα(i+1) | 肽键，几乎恒为 180°（反式） |
| $\chi_1, \chi_2, \dots$ | 侧链上的单键 | 侧链旋转 |

**为什么这很重要**：二面角旋转是**唯一**能在不破坏键长、键角的前提下改变分子形状的方式。
本项目生成构象就是绕这些键旋转（§5.2）。

### 2.3 肽键平面性与刚性

肽键 C–N 有部分双键性质（共振结构），导致：

* 六个主链原子 `Cα(i)–C(i)–O–N(i+1)–H–Cα(i+1)` 近似共面；
* 绕肽键的旋转被强烈限制（$\omega \approx 180°$）。

所以在 `pdbenergy/ensemble.py::find_rotatable_bonds` 里，我们**明确排除**了
`C`–`N` 肽键作为可旋转键。这是蛋白质里最重要的一处刚性。

### 2.4 拉氏图（Ramachandran plot）

把每个残基的 $(\phi, \psi)$ 画成散点图，会发现只有少数区域有分布：
右手 α 螺旋（约 $-57°, -47°$）、β 折叠（约 $-120°, +130°$）等。
这说明构象空间虽然维数高，但**可行区域是低维的、高度集中的**——
这也是为什么能量预测任务是可学的。

### 2.5 让蛋白质稳定的相互作用

* **疏水效应**：非极性侧链倾向埋在里面（本项目用隐式溶剂近似，见 §4.5）；
* **氢键**：`N–H···O=C`，主链之间形成 α 螺旋和 β 折叠；
* **盐桥**：带正电侧链（Lys、Arg）与带负电侧链（Asp、Glu）之间的静电吸引；
* **二硫键**：两个 Cys 之间的共价 S–S 键（注意：PDB 里常写作 `CYX`，见 §3.5）；
* **范德华**：短程吸引 + 极短程强烈排斥（这个排斥项是数值稳定性的最大麻烦，见 §5.5）。

### 2.6 二级结构

α 螺旋、β 折叠、转角、无规卷曲。`pdbio.py` 里不做二级结构判定——
能量函数不需要知道二级结构，它只知道原子和距离。**这正是机器学习势函数的魅力：
它不需要人为的特征工程，几何本身就编码了一切。**

---

## 3. PDB 文件格式与它的所有坑

### 3.1 固定列格式

PDB 格式是 1971 年设计的**固定列（fixed-column）**文本格式。每一行的每一列都有约定含义，
不是空格分隔的！例如 ATOM 行：

```
ATOM      1  N   THR A   1      17.047  14.099   3.625  1.00 13.79           N
12345678901234567890123456789012345678901234567890123456789012345678901234567890
```

| 列 | 字段 | 说明 |
|---|---|---|
| 1–6 | 记录类型 | `ATOM  ` / `HETATM` / `TER   ` / `MODEL ` / `ENDMDL` / `END   ` |
| 7–11 | 原子序号 | |
| 13–16 | 原子名 | 这是**最阴险**的一列，见 §3.6 |
| 17 | altLoc | 备选位置标记 |
| 18–20 | 残基名 | 三字母码 |
| 22 | 链 ID | |
| 23–26 | 残基序号 | |
| 27 | 插入码 | |
| 31–38 / 39–46 / 47–54 | x / y / z | 单位 Å，宽度只有 8 列，所以坐标绝对值不能超过 9999.999 |
| 55–60 | 占据率 occupancy | |
| 61–66 | 温度因子 B-factor | |
| 77–78 | 元素符号 | 很多文件是空的！ |

> 代码：`pdbenergy/pdbio.py::parse_atom_line` 用 1-based 的 `_f(line, start, end)` 切片，
> 就是为了让"列号"和文档一一对应。

### 3.2 为什么不用 Biopython

不是为了造轮子。原因是：**你必须亲眼看到这些列**。
一旦用高层库，当某天遇到 altLoc 或元素列为空时，你会完全不知道数据长什么样。
本项目自带解析器（零依赖），单元测试也覆盖了这些坑。
`requirements.txt` 里把 Biopython 列为可选项，交叉验证用。

### 3.3 多模型（NMR 系综）

X 射线晶体学给出一个结构；**NMR 解出来的是一个系综**（通常 10–40 个模型），
因为 NMR 的约束不足以确定唯一结构。文件里写作：

```
MODEL        1
... ATOM 行 ...
ENDMDL
MODEL        2
...
```

**这是白送的真实构象多样性**，本项目把它们全部用上（`use_nmr_models`）。
本项目的 16 个条目里有 6 个 NMR 条目，其中 1L2Y 有 38 个模型、1FME 有 34 个、3GB1 有 32 个。

> 代码：`pdbenergy/pdbio.py::parse_pdb_string` 返回 `list[Structure]`，每个模型一个。

### 3.4 altLoc：一个原子有两个位置

侧链无序时，晶体学家会给出两套坐标，用 altLoc 区分（`A`、`B`）。
力场无法表示"一个原子在两个位置"，必须二选一。

选择策略（`pdbio.py::resolve_altlocs`）：空白 altLoc 优先 → 占据率高者优先 → `A` 优先。

### 3.5 HETATM、水、离子、配体、非标准残基

* `HETATM` 行不一定都是"非蛋白质"，`MSE`（硒代甲硫氨酸）就常写作 HETATM；
* 水分子（`HOH`/`WAT`）、离子（`NA`/`CL`）、配体、去污剂都要去掉——
  因为我们用的是**蛋白质力场**，它没有这些分子的参数模板；
* 有些残基是标准残基的**变体**：`HID`/`HIE`/`HIP`（组氨酸三种质子化态）、
  `CYX`（成二硫键的半胱氨酸）、`ASH`/`GLH`（质子化的酸）、`LYN`（中性赖氨酸）。
  代码在 `AA3_VARIANTS` 里做了映射，并把它们视为可用。

> 代码：`pdbenergy/pdbio.py::protein_only`

### 3.6 元素列的坑

`CA` 这个原子名有两种含义：

* 在氨基酸残基里是 **α 碳（C）**；
* 在钙离子里是 **钙（Ca）**。

规则是：单字母元素右对齐（` CA `），双字母元素左对齐（`CA  `）。
但大量文件的元素列是空的，必须靠原子名推断。本项目
（`pdbio.py::infer_element`）的做法是：

1. 去掉原子名里的数字（`1HB ` → `HB`）；
2. 如果残基名**不是**标准氨基酸，才允许匹配双字母元素（`SE`、`ZN`、`CA` 等）；
3. 否则取首字母。

```python
if res not in STANDARD_AA:
    two = stripped[:2].upper()
    if two in TWO_LETTER_ELEMENTS:
        return two
return stripped[0].upper()
```

### 3.7 缺失原子与缺失残基

晶体结构常常缺失柔性环（disordered loop）或末端残基的原子。
**加氢之前必须先把重原子补齐**，否则模板匹配会失败。

本项目用 **PDBFixer**（OpenMM 作者写的修复工具）：

```python
fixer.findNonstandardResidues(); fixer.replaceNonstandardResidues()
fixer.findMissingAtoms();        fixer.addMissingAtoms()
fixer.addMissingHydrogens(pH=7.0)
```

注意我们**默认不补缺失的环**（`fixer.missingResidues = {}`），
因为补环等于**凭空编造坐标**，会污染数据集。这是一个可以配置的取舍。

### 3.8 为什么几乎必须自己加氢

PDB 里通常是**没有氢**的（X 射线看不到氢，只有中子衍射或超高分辨率才看得到）。
但：

* 力场是**全原子**的，没有氢就没有参数；
* **氢键**是蛋白质稳定的关键，没有氢就丢了核心物理。

所以预处理必须加氢。加氢要选**质子化态**（pH 依赖），本项目默认 pH 7.0。
这意味着 **pH 是一个超参数**：换 pH 会改变 His/Asp/Glu/Lys/Cys 的质子化态，
从而改变能量和标签。这一点必须记录在配置里（`prepare.ph`）。

### 3.9 写回 PDB：TER 记录的陷阱

本项目在 `pdbio.py::write_pdb` 里自己写 PDB 文件。
第一版有一个真实的 bug：在**每个残基之间**都写了 `TER`。

`TER` 是"链结束"标记，不是"残基结束"。每残基一个 `TER` 会让每个残基都变成
独立的一条链，于是每个残基都被当成 N 端/C 端残基，模板匹配直接崩：

```
ValueError: No template found for residue 0 (THR).
            The set of atoms is similar to CTYR, but is missing 5 C atoms.
```

正确做法是在**链 ID 变化时**才写 `TER`。这个 bug 说明一件事：
**数据格式错误会伪装成物理错误**。看到奇怪的模板匹配报错，先去检查文件格式。

### 3.10 PDB 之外的格式

现代标准是 **mmCIF / PDBx**（`.cif`），它解决了 PDB 格式的列宽限制和 10 万原子上限问题，
RCSB 现在默认只推荐 cif。本项目用 PDB 格式是因为它简单、肉眼可读、
且对本项目的小蛋白完全够用。生产环境建议改用 `gemmi` 或 Biopython 读 cif。

---

## 4. 分子力学力场：标签的物理来源

这一章解释"标签是从哪来的"。**如果这部分错了，后面所有机器学习都是垃圾**——
垃圾进，垃圾出（garbage in, garbage out）。

### 4.1 近似链条

1. **玻恩–奥本海默近似**：电子运动远快于原子核，可以分离。
   → 于是存在一个只依赖原子核坐标的能量函数 $E(\mathbf{R})$。
2. **经典近似**：不显式处理电子，把原子当成带电荷的球，用经验函数描述相互作用。
3. **可加性近似**：总能量拆成若干项之和（键、角、二面角、范德华、静电）。

### 4.2 AMBER 型的能量表达式

本项目用的 AMBER14SB 力场能量为：

$$
E = \underbrace{\sum_{\text{bonds}} k_b (r - r_0)^2}_{\text{键伸缩}}
  + \underbrace{\sum_{\text{angles}} k_\theta (\theta - \theta_0)^2}_{\text{角弯曲}}
  + \underbrace{\sum_{\text{dihedrals}} \frac{V_n}{2}\left[1 + \cos(n\phi - \gamma)\right]}_{\text{二面角}}
  + \underbrace{\sum_{\text{impropers}} k_\xi (\xi - \xi_0)^2}_{\text{异常二面角}}
  + \underbrace{\sum_{\text{CMAP}} V_{\text{cmap}}(\phi,\psi)}_{\text{主链校正图}}
  + \underbrace{\sum_{i<j} 4\epsilon_{ij}\left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{6}\right]}_{\text{范德华（Lennard-Jones 12-6）}}
  + \underbrace{\sum_{i<j} \frac{q_i q_j}{4\pi\epsilon_0 \epsilon_r r_{ij}}}_{\text{库仑静电}}
$$

逐项理解：

* **键伸缩 / 角弯曲**：谐振子近似（胡克定律）。真实化学键是非谐的，但在常温下
  键长偏离平衡值很小，所以谐振近似很好。这一项也是**最硬**的（力常数很大），
  意味着它一旦被违反（比如人为加坐标噪声），能量会爆炸——见 §5.1。
* **二面角**：周期函数 $1+\cos(n\phi-\gamma)$，$n$ 是周期数（如 $sp^3$–$sp^3$ 键 $n=3$）。
  这一项决定了旋转异构体（rotamer）的偏好，是**构象能的主要来源**。
* **异常二面角（improper）**：维持手性和平面性（例如苯环、肽键平面）。
* **CMAP**：AMBER 特有，用一个二维 $(\phi,\psi)$ 网格校正主链能量，
  用来修正力场对二级结构的偏好。
* **Lennard-Jones**：$r^{-12}$ 项是泡利排斥（防止原子重叠），$r^{-6}$ 项是色散吸引。
  注意 $r^{-12}$：两个原子靠得太近时能量会到 $10^9$ kcal/mol 量级，
  这是本项目中必须处理的实际问题（§5.5）。
* **库仑**：$q_i q_j / r$。蛋白质里带电基团很多，这一项是**长程**的，衰减很慢。

### 4.3 单位：一个必须谨慎的地方

| 量 | 本项目内部 | 说明 |
|---|---|---|
| 长度 | Å（埃，$10^{-10}$ m） | PDB 用 Å |
| 长度（OpenMM 内部） | nm（纳米） | **1 Å = 0.1 nm**，所有进出 OpenMM 的坐标都要转换 |
| 能量 | kcal/mol | 力场文献的通用单位 |
| 能量（OpenMM 输出） | kJ/mol | **1 kcal = 4.184 kJ** |

代码里 `pdbenergy/labels.py::KJ_PER_KCAL = 4.184`，并且用
`unit.kilocalories_per_mole` 让 OpenMM 自己做单位换算，
但读取坐标时仍然显式乘除 10。单位错误是这类项目最常见的 silent bug——
数值看起来"差不多能跑"，但物理完全错了。

### 4.4 什么是"力场参数化"

`forcefield.createSystem(topology)` 做的事：给拓扑里每个原子按
（残基模板 + 原子名）查表，赋予：

* 质量、电荷 $q_i$；
* LJ 参数 $\sigma_i, \epsilon_i$（不同原子间用几何平均组合：$\sigma_{ij}=\sqrt{\sigma_i\sigma_j}$，
  $\epsilon_{ij}=\sqrt{\epsilon_i\epsilon_j}$）；
* 每一根键的 $k_b, r_0$，每一个角的 $k_\theta, \theta_0$，每一个二面角的 $V_n, n, \gamma$。

**模板匹配失败**是最常见的错误（"No template found for residue X"），
原因通常是：残基名不认识（非标准残基）、原子缺了、电荷无法中和。
处理办法见 §3.7。

### 4.5 隐式溶剂：为什么需要，怎么近似

真实实验在水里。显式加几千个水分子会**成倍增加原子数**（一个 46 残基蛋白约 640 个原子，
至少要加 3000+ 个水），代价无法接受。

**隐式溶剂（implicit solvent）**把水当成连续介质，用**广义玻恩（Generalized Born, GB）**
模型近似溶剂化自由能：

$$
G_{\text{solv}} \approx
-\frac{1}{2}\left(1 - \frac{1}{\epsilon_{\text{solvent}}}\right)
\sum_{i,j} \frac{q_i q_j}{\sqrt{r_{ij}^2 + \alpha_i \alpha_j \exp(-r_{ij}^2/(4\alpha_i\alpha_j))}}
+ G_{\text{SA}}
$$

关键概念：

* $\alpha_i$ 是**玻恩半径**（有效半径），描述原子 $i$ 被溶剂"看到"的程度，
  埋在蛋白质内部的原子的玻恩半径比暴露在外的大；
* 这个双求和让能量变成 $O(N^2)$，比显式溶剂仍便宜得多；
* $G_{\text{SA}}$ 是**非极性溶剂化**项，正比于溶剂可及表面积（SASA）。

本项目用 **GBn2**（`implicit/gbn2.xml`），是 GB 的一个较新改进版本，
对蛋白质的平衡性质更准。

**为什么本项目能用隐式溶剂**：它让"一个孤立蛋白构象的能量"成为良定义的量，
不需要水盒子。这正是"PDB 文件 → 能量"这个任务能成立的前提。

### 4.6 一个 OpenMM 8.x 的 API 坑

网上大量老教程（包括 OpenMM 7.x 时代）这么写：

```python
system = forcefield.createSystem(topology, implicitSolvent=app.OBC2)   # ❌
```

在 OpenMM 8.6 上这会报：

```
ValueError: The argument 'implicitSolvent' was specified to createSystem() but was never used.
```

因为在 8.x 里，隐式溶剂是**由你加载哪个 XML 文件决定的**：
`implicit/gbn2.xml` 内部有一段 `<Script>`，它会自己构造 `CustomGBForce`，
并读取 `soluteDielectric` / `solventDielectric` / `sasaMethod` 这些关键字参数。
所以正确写法是：

```python
forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
system = forcefield.createSystem(topology,
                                 nonbondedMethod=app.CutoffNonPeriodic,
                                 nonbondedCutoff=1.0*unit.nanometer,
                                 constraints=app.HBonds,
                                 soluteDielectric=1.0,
                                 solventDielectric=78.5)          # ✅
```

> 代码：`pdbenergy/labels.py::OpenMMEnergyEngine._create_system`

### 4.7 非键相互作用的截断

LJ 和库仑都是对所有原子对求和。**截断（cutoff）**只算距离小于 $r_c$ 的对：

* 好处：代价从 $O(N^2)$ 降到约 $O(N)$，本项目实测**快 3.6 倍**（65.6 ms → 22.0 ms）；
* 代价：长程库仑被截断在物理上不严格；
* 惯例：隐式溶剂 MD 里常用 **1.0–1.2 nm**，本项目默认 `1.0 nm`。

配置项是 `label.nonbonded_method`（`CutoffNonPeriodic` 或 `NoCutoff`）和 `label.cutoff_nm`。
`preset="thorough"` 会切回 `NoCutoff`（精确但慢）。

### 4.8 氢键约束与时间步

MD 里最快的运动是 X–H 键的伸缩（周期约 10 fs）。若显式积分它，
时间步必须小于 ~1 fs 才稳定。用 **SHAKE/LINCS 类约束**把 X–H 键长固定，
最快的自由度被移除，时间步可以放大到 **2 fs**，速度翻倍。

配置项：`label.constraints = "HBonds"`。

### 4.9 能量分解

把总能量按项拆开，既方便调试也是很好的物理直觉训练。
OpenMM 的 `Force` 可以打上 **force group** 标签，然后
`getState(getEnergy=True, groups={i})` 只返回该组的能量。

本项目把力分成四组（`labels.py::ENERGY_TERMS`）：

| 组 | 力 | 含义 |
|---|---|---|
| 0 | `HarmonicBondForce` | 键伸缩 |
| 1 | `HarmonicAngleForce` | 角弯曲 |
| 2 | `PeriodicTorsionForce`, `RBTorsionForce`, 其他 | 二面角 / 异常二面角 / CMAP |
| 3 | `NonbondedForce`, `CustomNonbondedForce` | LJ + 库仑 + GB 溶剂化 |

**验证代码正确性的一个好办法**：四组能量之和必须等于 `getState()` 得到的总能量。
本项目在 `smoke_test` 里验证了这一点（`sum of terms == total`，误差 0）。

---

## 5. 构象采样：数据集是怎么造出来的

有了一台"能量计算器"（力场），还需要**一堆构象**才能构成数据集。
这一章讲怎么造，以及造错了会怎样。

### 5.1 反面教材：为什么不能直接加高斯噪声

最直观的想法：给坐标加随机噪声不就有新构象了吗？

```python
coords + np.random.normal(0, sigma, coords.shape)     # ❌
```

**这是错的**，而且错得很具体。一个 C–H 键只有 **1.09 Å** 长。
独立地给两个端点各加 $\sigma = 0.04$ Å 的噪声，键长变化约 $0.04\sqrt{2} \approx 0.057$ Å。
键伸缩能量是 $\frac{1}{2} k (r-r_0)^2$，AMBER 的 C–H 力常数约 $340$ kcal/mol/Å²，
所以**每一根键**平均贡献约 $\frac{1}{2}\times340\times0.0032 \approx 0.55$ kcal/mol。
一个 642 原子的蛋白有 652 根键，于是：

$$\Delta E_{\text{噪声}} \approx 650 \times 0.55 \approx \textbf{360 kcal/mol}$$

本项目实测（1CRN，`cartesian_jitter=0.04`）：

| 指标 | 加了 0.04 Å 噪声 | 不加噪声 |
|---|---|---|
| ΔE 中位数 | **797 kcal/mol** | 2.2 kcal/mol |
| ΔE 25 分位 | 750 kcal/mol | 2.2 kcal/mol |
| 键项能量范围 | 22 … **10883** kcal/mol | 22 … 27 kcal/mol |

也就是说，噪声造出来的"构象"里 **99% 的能量是纯粹的键长拉伸噪声**，
它和"构象"没有任何关系。更糟的是：

1. 模型看到的是**几何**，但噪声能量主要由**键长**决定——模型学不到有用东西；
2. 能量尺度被抬高三个量级，平方误差损失会被这些点完全主导；
3. 而且一旦用 `constraints=HBonds`，X–H 键被约束住，**最小化也修不回来**。

> 本项目的处理：`EnsembleConfig.cartesian_jitter` 默认 **0.0**，
> 并且这个字段只在你想复现上述实验时才该打开。
> 这是一个"用实测数据否掉一个直觉方案"的典型案例，也是本教程想传达的态度。

### 5.2 正确做法：绕可旋转单键旋转二面角

**二面角旋转**（torsional rotation）是唯一能在**精确保持所有键长、键角**的前提下
改变分子形状的操作。

数学上，要绕轴 $\mathbf{u}$（过点 $\mathbf{p}_0$）旋转角度 $\theta$，
用 **罗德里格斯公式（Rodrigues' rotation formula）**：

$$\mathbf{R} = \mathbf{I} + \sin\theta\,[\mathbf{u}]_\times + (1-\cos\theta)\,[\mathbf{u}]_\times^2$$

其中 $[\mathbf{u}]_\times$ 是 $\mathbf{u}$ 的反对称矩阵。

关键技巧：对一根键 `a–b`，把**轴选为过 `a` 且方向为 $\mathbf{r}_b - \mathbf{r}_a$**，
然后只旋转 `b` 一侧的**整个子树**：

```python
moved = _subtree_atoms(adj, a, b)       # b 一侧的所有原子（含 H）
origin = positions[a]
axis = positions[b] - positions[a]
out[moved] = origin + (positions[moved] - origin) @ R.T
```

为什么这样能**精确**保持键长键角？因为旋转轴**正好穿过 `a` 和 `b`**：

* `a`、`b` 在轴上 → 不动；
* 子树内部所有原子**一起**做刚体旋转 → 内部所有距离不变；
* `b` 上挂的原子到 `b` 的距离不变（都在绕轴旋转）→ $a$–$b$–$X$ 键角不变。

**唯一变化的就是那个二面角。** 代码在 `pdbenergy/ensemble.py::rotate_torsion`。

### 5.3 哪些键可以转

`find_rotatable_bonds` 的判定条件：

1. 两端都是**重原子**（H 的旋转只改变 H 位置，且 H 通常跟随重原子）；
2. **不在环上**（环上的键转不动，会撕开环）——用限制深度的 BFS 判环；
3. **不是末端键**（转一个末端甲基没有意义）；
4. **不是肽键 C–N**（§2.3 的刚性）；
5. 键两端各自至少还有 2 个重原子（否则转了等于没转）。

实测：1CRN（46 残基、642 原子）有 652 根键，其中 **150 根可旋转**。

### 5.4 探索幅度分层

本项目把可旋转键按**旋转幅度**分层（`torsion_levels`），
从 ±55° 到 ±165°，每层生成 `n_torsion_per_level` 个样本。
这样能量分布从"轻微扰动"一直覆盖到"大幅度构象变化"，
而不是全部堆在一处。

每次采样随机选 1–3 根键同时旋转（`moves_min`/`moves_max`），
这样既有单键变化也有协同变化。

### 5.5 关键步骤：冲突筛选

二面角旋转虽然保持键长键角，但可能让一个侧链**直接穿进蛋白质内部**，
造成两个原子几乎重叠。这时 LJ 的 $r^{-12}$ 项会给出天文数字：

$$r = 1.0\,\text{Å},\ \sigma = 3.4\,\text{Å} \Rightarrow
4\epsilon\left(\frac{\sigma}{r}\right)^{12} \approx 0.4 \times 3.4^{12} \approx 1.5\times10^{5}\ \text{kcal/mol}$$

实测（未筛选时）1CRN 的 ΔE 最大达到 **$5.2\times10^{9}$** kcal/mol，中位数也有 797。

**这种几何不是"高能构象"，而是数值伪影**——真实分子不可能让两个原子重叠。
所以本项目**逐帧筛选**：

```python
value = energy_of(pos)
if not np.isfinite(value) or value - e_native > max_relative_energy:
    rejected += 1
    return False
```

* `max_relative_energy = 250` kcal/mol 用于二面角/NMR 帧；
* MD 快照用另一个**宽松得多**的上限 `md_max_relative_energy = 1500`，
  原因见 §5.7（温度带来的能量是**真物理**，不是伪影）；
* 如果某层被拒绝太多，会最多重试 `n_torsion_per_level * max_attempt_factor` 次。

实测筛选效果（1CRN / 1BDD）：

| | 1CRN（46 残基） | 1BDD（60 残基，侧链更埋） |
|---|---|---|
| 二面角帧保留率 | 71%–86% | 29%–46% |
| 被拒绝的冲突帧 | 14 | 85 |
| 最终 ΔE 范围 | 0 – 530 kcal/mol | 0 – 709 kcal/mol |

> `1BDD` 保留率低是**合理的物理**：大蛋白侧链更拥挤，能转的空间更小，
> 一次随机旋转更容易撞上别的原子。这也是为什么筛选必须**逐帧做**，
> 而不能简单地"每层固定生成 N 个"。

### 5.6 能量最小化：L-BFGS 与局部极小

**能量最小化**就是求 $\min_{\mathbf{R}} E(\mathbf{R})$。常用 **L-BFGS**
（有限内存的拟牛顿法）：用历史梯度差近似 Hessian 的逆，每步代价 $O(N)$。

1. **建立参考能量 $E_{\min}$**（默认开启）：把 PDBFixer 加氢后的结构最小化。
   这一步很必要，因为"从 PDB 加氢得到的结构"离局部极小还差几百 kcal/mol
   （实测 1CRN：加氢后结构 vs 最小化后，差约 $1500$ kcal/mol）。
   $E_{\min}$ 是相对能量的**零点**。
2. **产生局部极小样本**（默认关闭，`minimise_fraction = 0.0`）：
   对一部分二面角扰动帧做部分最小化，得到靠近极小值的构象。

**为什么默认关掉第 2 项**——这是一个**用实测数据做出来的决定**：

* 二面角旋转**本身就经常给出低能构象**。暴露在表面的侧链转一下几乎不耗能，
  实测 1CRN 的二面角帧 ΔE **中位数只有 2.2 kcal/mol**，低能区其实已被覆盖；
* 每次部分最小化是**秒级**开销（1D3Z 这种 1231 原子的体系更贵），
  它在流程里占据了**绝大部分运行时间**，却没有带来新的能量区间；
* 关掉之后 1CRN 的帧数从 79 降到 54，ΔE 范围几乎不变（0–580 → 0–530）。

结论：**先量一下某个步骤到底贡献了什么，再决定要不要为它付钱。**
想要显式的局部极小样本，把 `minimise_fraction` 设成 0.15 即可
（文档与代码都保留了这条路径）。

### 5.7 朗之万分子动力学：采样热可及构象

**分子动力学（MD）**数值积分牛顿方程。本项目用 **朗之万（Langevin）动力学**：

$$m_i \ddot{\mathbf{r}}_i = -\nabla_i E - \gamma m_i \dot{\mathbf{r}}_i + \mathbf{R}_i(t)$$

* 摩擦项 $-\gamma m \dot{\mathbf{r}}$ 与随机力 $\mathbf{R}(t)$ 通过**涨落–耗散定理**配套，
  共同充当**恒温器**；
* 采样的是**正则系综（NVT）**，即在温度 $T$ 下的玻尔兹曼分布
  $P(\mathbf{R}) \propto e^{-E(\mathbf{R})/k_BT}$。

**关键物理点：为什么 MD 快照的能量比极小值高几百 kcal/mol？**

这是**能量均分定理**，不是 bug。对 $N_{dof}$ 个自由度，
在平衡时势能的时间平均值比极小值高约：

$$\langle \Delta E \rangle \approx \frac{1}{2} N_{dof} k_B T$$

对 1CRN：$N_{dof} = 3\times642 - 6 \approx 1920$，$k_BT_{300\text{K}} \approx 0.596$ kcal/mol，
于是 $\langle\Delta E\rangle \approx 570$ kcal/mol。

实测的 MD 帧 ΔE 正好落在 **300–580 kcal/mol**，与理论值吻合。
这验证了实现是对的，也说明**筛选阈值必须对 MD 放宽**——
如果用 250 的阈值去筛 MD 帧，正常的热运动帧会全部被误杀
（本项目第一版就是这样，MD 帧保留率 0/5）。

**MD 参数**：每个温度 0.6 ps、时间步 2 fs、每 0.2 ps 存一帧，温度 300/450 K。
温度越高能量分布越宽，为模型提供高能区的样本。

**一个必须注意的实现细节**：MD 必须**从最小化后的结构出发**。
如果从"刚加完氢的 PDB 结构"出发（它在极小值上方 ~1500 kcal/mol），
1 ps 根本不足以弛豫掉这些应变，结果所有帧都会被判为冲突而丢弃。

> 代码：`labels.py::run_md(start_positions_angstrom=...)`、`ensemble.py`

### 5.8 NMR 模型：白送的实验构象

多模型 NMR 条目的每个模型都是**实验测定**的构象。
本项目把每个模型**完整地重新准备一遍**（丢掉原有的氢、用 PDBFixer 重新加氢），
再做 60 步最小化，然后作为样本收入。

**为什么不能"只把重原子坐标搬过来"**——这是一个花了很久才找到的坑，值得单独讲。

最初的实现是：从已准备好的体系里取氢的坐标，把 NMR 模型的重原子坐标**覆盖**上去，
然后做短最小化。结果**一个 NMR 帧都没能通过筛选**（0/38）。原因很具体：

* 氢是**刚性共价连接**在重原子上的，C–H 键只有 1.09 Å；
* NMR 模型与参考构象的重原子位移实测约 **1.0–1.8 Å**；
* 于是每一个 C–H 键都被拉长/压短了将近 1 Å。
  按 $E=\frac12 k(\Delta r)^2$、$k\approx340$ kcal/mol/Å² 估算，
  **每一根键就要几百 kcal/mol**，一百多根键直接到 $10^4$–$10^7$ kcal/mol；
* 短最小化有时能把氢拉回去（ΔE ≈ 50），有时不能（ΔE ≈ 4000），极不稳定。

**根本问题**：这不是"高能构象"，而是**把两个不该混用的坐标拼在了一起**。
正确做法是让每个模型拥有**自己的一套氢坐标**：

```python
heavy = protein_only(model, keep_hydrogens=False)
topology, positions_nm = engine.protonate(heavy, name, workdir=...)   # 重新加氢
if not _same_atom_layout(topology, prepared):   # 布局必须逐原子一致
    continue
```

修好之后，实测每个模型的 ΔE 是 **15–91 kcal/mol**（最小化 60 步后），
全部落在筛选阈值内。1FME / 1L2Y / 3GB1 各贡献 3 个实验构象帧。

> 代码：`ensemble.py` 的 NMR 分支、`ensemble.py::_same_atom_layout`

**另一个实现细节**：NMR 循环要限制的是**尝试次数**而不是成功次数。
如果某个模型弛豫后仍然超限，按"成功数"计数会让循环走遍全部 30–40 个沉积模型，
每个都付一次最小化代价——1FME 有 34 个模型、3GB1 有 32 个。
**运行时间不应该取决于筛选碰巧好不好过。**

### 5.9 数据集的来源与各自的意义

默认配置下每个蛋白产生的帧：

| 来源 tag | 生成方式 | 覆盖的物理区域 | 1CRN 实测帧数 |
|---|---|---|---|
| `nmr` | NMR 模型 + 短最小化 | 实验观测构象 | 0（1CRN 只有一个模型） |
| `torsion` | 二面角旋转 | 低到中高能、无键长应变 | 48（4 层 × 12） |
| `torsion_min` | 二面角旋转 + 部分最小化 | 局部极小、低能 | 0（默认关闭，见 §5.6） |
| `md300K` / `md450K` | 朗之万 MD | 热可及、含均分能量偏置 | 3 + 3 |

合计 **约 53 帧/蛋白**（NMR 条目会多几帧），14 个蛋白约 **750 个样本**。

**为什么要混合**：单一来源的数据分布太窄。

* 只用**最小化**的结构，能量全挤在 0 附近，模型学不到能量差；
* 只用 **MD**，能量全在 300–600 kcal/mol 的大偏置区（均分定理，§5.7），
  低能区的分辨率很低；
* 只用**二面角旋转**，则缺少真实的、热可及的构象分布。

三者叠加，才能同时覆盖"低能盆地"（二面角帧，中位数仅约 2 kcal/mol）
和"高能形变"（大幅二面角旋转 + 高温 MD，一直到 500–700 kcal/mol）。

**另一个重要的观察**：不同来源的 ΔE 分布差别极大，
所以**必须按来源分组报告误差**（§9.5）——
一个总体 MAE 数字会把这些差异完全掩盖掉。

---

## 6. 把分子变成图：机器学习的表示

有了 (构象, 能量) 数据，接下来要决定**怎么把构象喂给神经网络**。这一步的选择
往往比网络结构更重要。

### 6.1 必须尊重的三个对称性

物理告诉我们 $E(\mathbf{R})$ 有哪些对称性？如果模型不内置这些对称性，
它就得从数据里"学"它们，浪费大量样本和参数。

**① 平移不变性**：整体平移分子，能量不变。$E(\mathbf{R} + \mathbf{t}) = E(\mathbf{R})$。

**② 旋转不变性**：整体旋转分子，能量不变。$E(Q\mathbf{R}) = E(\mathbf{R})$，$Q$ 是任意旋转矩阵。

**③ 置换不变性**：把原子列表重新编号，能量不变。$E(\pi(\mathbf{R})) = E(\mathbf{R})$。

**怎么保证前两个**：**永远不要把原始坐标当成特征**。
把每个特征都构造成**只依赖原子间距离**（距离在旋转平移下不变）。
本项目只给网络距离，坐标仅用于算距离。

> 反面例子：直接把 $(x,y,z)$ 当每个原子的特征输入 MLP。模型被迫从数据里学习
> "旋转后的坐标应该给同样的输出"，这需要指数级多的数据。
> 这是新手最常犯的错误。

**怎么保证第三个**：所有聚合操作都用**求和**（或求平均/求最大值），
因为求和与顺序无关（scatter-add，见 §7.2）。

### 6.2 图表示

一个构象被表示成一张图 $G=(V,E)$：

* **节点 $V$**：每个原子。特征 = 原子序数 $Z$（C=6、N=7、O=8、S=16、H=1）。
* **边 $E$**：距离小于 **cutoff**（本项目 4.5 Å）的原子对。特征 = 距离（+ 共价键标记）。

**为什么要 cutoff**：能量虽然名义上是所有原子对的和（库仑是长程的），
但局部环境（键、角、二面角、范德华接触）贡献了绝大部分方差。
用有限 cutoff 把图变稀疏，边数从全连接 $O(N^2)$ 降到约 $O(N\bar{k})$，
$\bar{k}$ 是平均邻居数。

实测（1CRN，642 原子）：全连接会有 $642\times641/2 \approx 205{,}881$ 条边，
用 4.0 Å 的 cutoff 后只剩 **约一万条**，**稀疏了近 20 倍**。

**cutoff 也是模型表达能力的上限**：$k$ 层消息传递后，
一个原子能"看到"约 $k \times r_c$ 的范围（本项目 $4\times4.5 = 18$ Å）。
长程静电的尾部是学不到的——这是这类模型的根本局限，见 §12。

### 6.3 邻居列表的高效计算

对 $N$ 个原子算所有距离是 $O(N^2)$。用完全平方公式展开可以交给 BLAS：

$$d_{ij}^2 = \|\mathbf{r}_i\|^2 + \|\mathbf{r}_j\|^2 - 2\,\mathbf{r}_i\!\cdot\!\mathbf{r}_j$$

```python
sq  = np.einsum("ij,ij->i", coords, coords)
d2  = sq[:, None] + sq[None, :] - 2.0 * (coords @ coords.T)
np.maximum(d2, 0.0, out=d2)          # 数值误差可能给出极小的负数
dist = np.sqrt(d2)
np.fill_diagonal(dist, np.inf)       # 排除自身
```

> 代码：`pdbenergy/features.py::pairwise_distances`

对 $N \lesssim 1500$ 这比建 cell list / KD-tree 更快、更简单。
更大的体系才需要空间划分。

**只算一次**：邻居列表和共价键判断都需要距离矩阵，
本项目把它算一次然后传给两个函数。初版代码算了两遍，白白慢一倍。

**每原子最大邻居数**：为内存安全设了 `max_neighbors = 32`。
选最近 40 个用 `np.argpartition`（$O(N^2)$），而不是 `np.argsort`（$O(N^2\log N)$）。
这个常数优化在每帧都要跑的情况下很值钱。

### 6.4 径向基函数（RBF）：把距离变成好用的特征

**为什么不直接把距离当特征？** 因为距离是无界标量，
而化学上有意义的结构（1.09 Å 的 C–H 键、1.45 Å 的 C–C、
2.8 Å 的氢键、3.8 Å 的疏水接触）全都挤在 1–4 这个小区间里。
直接喂原始距离，网络要学一个非常"陡"的函数，条件数很差。

**高斯径向基**把标量距离展开成平滑向量：

$$e_k(r) = \exp\!\left(-\frac{(r-\mu_k)^2}{2\Delta^2}\right),\qquad
\mu_k = r_{\min} + k\Delta,\quad \Delta = \frac{r_{\max}-r_{\min}}{K-1}$$

$K$ 个中心均匀铺在 $[0, 5]$ Å 上（本项目 `n_rbf = 32`，$\Delta \approx 0.161$ Å）。
每个距离只点亮附近 3–4 个基函数——**这是稀疏表示**，
也是我们把 RBF 展开放进网络而不是存进数据集的原因（§8.2）。

> 代码：`pdbenergy/models.py::GaussianRBFExpansion`（PyTorch 版）
> 和 `pdbenergy/features.py::gaussian_rbf`（NumPy 版，给基线用）

### 6.5 截断包络函数：让能量连续

如果只取 $r < r_c$ 的边，那么当一对原子跨过 cutoff 时，
它们之间的消息会**突然消失**，预测能量出现**跳变**。对势能面来说这是灾难性的：

* 能量对坐标不连续 → 力（梯度）出现尖峰 → 无法做几何优化或 MD；
* 模型会对 cutoff 附近的位置过度敏感。

解决办法是乘一个**平滑窗函数**，在 $r_c$ 处平滑归零：

$$f_c(r) = \begin{cases}
\frac{1}{2}\left[\cos\!\left(\frac{\pi r}{r_c}\right)+1\right], & r < r_c\\
0, & r \ge r_c
\end{cases}$$

它在 $r=0$ 处为 1，在 $r=r_c$ 处为 0 且一阶导数为 0。

> 代码：`models.py::GaussianRBFExpansion.forward` 里的 `envelope`

### 6.6 共价键标记

力场能量里**最大的一项**是键伸缩和角弯曲。
"哪两个原子成键"是纯化学图信息，比距离更容易获得——
与其让网络从 1.09 Å 这个距离自己推断"C 和 H 成键"，
不如直接给它一个二值标记。

判定用**共价半径**：当 $d_{ij} < 1.3\,(r^{\text{cov}}_i + r^{\text{cov}}_j)$ 时认为成键。
系数 1.3 用来容忍不同键级和轻微形变。

> 代码：`features.py::bond_flags_from_distances`、`COVALENT_RADII`

这让每条边有 2 个特征：`[距离, 是否成键]`（`use_bond_feature = True`）。

### 6.7 手工描述符基线：为了知道图模型到底值多少

本项目同时实现了**手工特征 + MLP** 基线（`features.py::global_descriptors`），
24 维**全局**、旋转平移不变的量：

| 特征 | 物理含义 |
|---|---|
| 回转半径 $R_g$ | 分子紧凑程度 |
| 最大原子间距 | 伸展程度 |
| 回转张量特征值（3 个） | 形状：棒状 / 盘状 / 球状 |
| 各向异性 | 形状偏离球形的程度 |
| 元素计数（H/C/N/O/S/其他，除以 N） | 组成 |
| 距离 < 5/8/10/12 Å 的原子对数 | 接触密度 |
| $\sum 1/r_{ij}$（÷10⁴） | 单极静电能量的代理 |
| $\sum 4\epsilon[(\sigma/r)^{12}-(\sigma/r)^6]$ | LJ 能量的代理 |
| 2.5–3.6 Å 的原子对数 | 氢键数量的代理 |

这些特征保留了物理直觉（它们确实是能量的重要组成部分），
但丢掉了"哪个原子挨着哪个原子"这个最关键的信息。

**基线的价值**：如果 GNN 不比它好，说明网络结构没起作用；
如果好很多，说明局部结构信息是能量的主要来源。实测见 §11.3。

---

## 7. 图神经网络与 SchNet

### 7.1 消息传递的一般框架

几乎所有现代分子 GNN（SchNet、MPNN、DimeNet、NequIP、MACE）都可以写成
**消息传递**：

$$\mathbf{h}_i^{(l+1)} = \text{UPDATE}\Big(\mathbf{h}_i^{(l)},\
\bigoplus_{j \in \mathcal{N}(i)} \text{MESSAGE}\big(\mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)}, \mathbf{e}_{ij}\big)\Big)$$

* $\mathbf{h}_i^{(l)}$：第 $l$ 层原子 $i$ 的**特征向量**；
* $\mathcal{N}(i)$：$i$ 的邻居（cutoff 内的原子）；
* $\mathbf{e}_{ij}$：边的几何特征（RBF 展开后的距离 + 键标记）；
* $\bigoplus$：**置换不变的聚合**（求和/平均/最大值）；
* UPDATE / MESSAGE：可学习的函数（这里是 MLP）。

直觉：每个原子把邻居信息"收集"过来，更新自己的表示。
叠 $k$ 层，信息就传播了 $k$ 跳。

### 7.2 scatter-add：置换不变性从哪来

核心操作是**分段求和（scatter-add）**：把每条边的消息按"接收原子编号"累加。

```python
def scatter_sum(src, index, dim_size):
    out = torch.zeros((dim_size, *src.shape[1:]), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)     # 把 src[e] 累加到 out[index[e]]
    return out
```

`index_add_` 就是 scatter-add 的原子操作，**不需要任何图神经网络库**
（不用 PyTorch Geometric）。求和与加数顺序无关，
所以原子怎么编号都不影响结果——这就是置换不变性的来源。

> 代码：`pdbenergy/models.py::scatter_sum`

### 7.3 SchNet 的连续滤波卷积

SchNet（Schütt et al., 2017）的关键创新是把"邻接矩阵权重"
换成**依赖于距离的连续滤波**：

$$\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} +
\sigma\!\left(\mathbf{W}\sum_{j\in\mathcal{N}(i)}
\mathbf{h}_j^{(l)} \odot \mathbf{W}_{\text{filter}}(\mathbf{e}_{ij})\right)$$

逐项看：

* $\mathbf{W}_{\text{filter}}(\mathbf{e}_{ij})$：一个小 MLP，
  把边的 RBF 特征映射成 `hidden_dim` 维的**滤波器**。这就是"连续滤波"——
  同一个权重矩阵在 1.5 Å 和 4 Å 处表现不同，因为滤波器向量不同。
* $\odot$：逐元素相乘（调制邻居特征）。
* 求和：置换不变的聚合。
* $\mathbf{W}+\sigma$（SiLU）：线性变换 + 非线性。
* $+\ \mathbf{h}_i^{(l)}$：**残差连接**，让深层网络可训练。

> 代码：`pdbenergy/models.py::InteractionBlock` 与 `RBFFilter`

### 7.4 原子嵌入

原子序数 $Z$ 是**离散类别**。用 `nn.Embedding(119, hidden_dim)`
把它映射成可学习的连续向量，比 one-hot 更省参数、更易优化。
`Z=0` 保留给未知元素。

### 7.5 求和池化：可扩展性（extensive 性质）

$$\mathbf{E}_{\text{pred}} = \sum_{i=1}^{N} \varepsilon_\theta(\mathbf{h}_i^{(L)})$$

为什么用求和而不是平均/最大？因为**能量是广延量（extensive）**：
两个互不作用的分子放在一起，总能量等于各自能量之和。
求和自带这个性质；平均完全不满足。

> 代码：`models.py::SchNetRegressor.forward`

### 7.6 全局残差头与组成特征

纯求和无法表示**集体效应**，也无法利用"原子组成"这类全局信息。
所以池化之后再接一个小 MLP：

```python
pooled      = scatter_sum(per_atom, batch, n_graphs)         # (B,)
composition = composition_features(z, batch, n_graphs)       # (B, 6)
correction  = global_head(cat([pooled, composition]))        # (B, 1)
return pooled + correction
```

`composition` 是每种元素（H/C/N/O/S/其他）在该分子中的**归一化个数**。
这对**绝对**能量预测特别重要——绝对能量最大的决定因素就是原子组成。

> 代码：`models.py::composition_features`、`SchNetRegressor.global_head`

### 7.7 感受野

$k$ 层消息传递后，原子 $i$ 的特征包含距离它 $k\cdot r_c$ 以内所有原子的信息。
本项目 $k=3$、$r_c=4.0$ Å → 感受野约 12 Å。

**代价**：想覆盖更大范围，要么加层（参数和计算量上去，还有 oversmoothing 问题），
要么用更大的 cutoff（图变密，$O(N^2)$ 风险）。
这是"局部 vs 长程"的基本矛盾。

### 7.8 与其他架构的关系

| 模型 | 边特征 | 特点 |
|---|---|---|
| **SchNet**（本项目） | 距离 | 简单、快、易实现，只用到 2 体信息 |
| DimeNet / DimeNet++ | 距离 + **键角** | 显式建模三体相互作用，更准但更贵 |
| NequIP / Allegro | 距离 + 球谐函数 | 等变网络，理论上更优雅 |
| MACE | 距离 + 高阶多体消息 | 目前精度/效率的领先者之一 |
| ANI-1 / ANI-2x | 距离（改型） | 分子 MLIP 的经典数据集+模型 |

SchNet 是**理解消息传递的最好起点**：结构清晰，
在当前数据集规模（约 750 样本）上参数少、不容易过拟合。

---

## 8. PyTorch 训练实践

### 8.1 数据管线：Dataset + collate

**问题**：一张图有 642 个原子、18407 条边；一个 batch 里各图原子数不同。
常见错误做法是 padding 成矩形张量——浪费大量计算在"假原子"上。

**正确做法**：把 batch 里的图**拼成一张大图**，
再用一个 `batch` 向量记录"第几个原子属于第几个样本"：

```python
edge_index += offset              # 每张图的边索引加上偏移
batch_idx   = [b, b, ..., b]      # 每个原子属于哪个样本
offset     += n_atoms
```

于是模型只处理真实的原子对。求和池化时
`scatter_sum(per_atom, batch, n_graphs)` 自动把原子聚合回各自的分子。

> 代码：`pdbenergy/features.py::collate_graphs`

### 8.2 图缓存：一个必须做的工程决策

**实测**：单帧 featurisation 约 **150 ms**（642 原子）。
数据集约 750 帧 → 每轮 epoch 光造图就要 **110 秒**。
80 个 epoch 就是 **2.5 小时**，全部花在重复计算完全相同的东西上。

**解决方案**：图只依赖 (坐标, 特征配置)，与模型、epoch、目标都无关
→ 算一次，存磁盘，之后从内存取。

配套两个优化：

1. **边特征只存 `[距离, 键标记]`**（2 个 float），
   而不是 48 维 RBF 稠密向量；RBF 展开挪进模型（§6.4）。
   缓存因此缩小**约 25 倍**。
2. **缓存文件名包含特征配置的哈希**，改了 cutoff 或 RBF 参数会自动失效重建，
   不会悄悄用错通道数的旧缓存。

> 代码：`pdbenergy/graphcache.py`

### 8.3 目标标准化

预测目标 ΔE 范围 0 – 857 kcal/mol。直接回归原始值尺度大、学习率难选。

**标准化**：用**训练集**的均值和标准差变换目标

$$y' = \frac{y - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

注意事项：

* $\mu, \sigma$ **只能用训练集算**！用全数据集算是数据泄漏（§9.1）；
* 训练和推理必须用**同一对** $\mu,\sigma$，所以要存进检查点；
* 报告指标时**必须反标准化回 kcal/mol**，否则数字没有物理含义。

### 8.4 损失函数：为什么用 Huber 而不是 MSE

**MSE** 对离群点极其敏感：$\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum (y_i - \hat y_i)^2$

本项目数据里有些构象 ΔE 是 800 kcal/mol，典型值是 3 kcal/mol。
一个 800 的残差贡献的梯度是残差 3 的 **267 倍**，
少数点会把整个拟合拖偏。

**Huber 损失**在误差小时二次、大时线性：

$$\mathcal{L}_\delta(e) = \begin{cases}
\frac{1}{2}e^2, & |e| \le \delta\\
\delta\left(|e| - \frac{1}{2}\delta\right), & |e| > \delta
\end{cases}$$

在最优解附近保留 MSE 的平滑快速收敛，在尾部只线性增长，
**限制任何单个样本的影响**。配置：`train.loss = "huber"`、`huber_delta = 1.0`。

### 8.5 优化器：AdamW

**Adam** 维护梯度的一阶矩（动量）和二阶矩（自适应步长）：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

**AdamW** 修了 Adam 里 L2 正则被自适应步长顺带缩放的问题，
把权重衰减解耦：

$$\theta_t = \theta_{t-1} - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\theta_{t-1}\right)$$

配置：`learning_rate = 5e-4`、`weight_decay = 1e-5`。

### 8.6 学习率调度

固定学习率很难兼顾"早期快速下降"和"后期精细收敛"。
默认用 `ReduceLROnPlateau`：验证损失连续若干轮不改善就减半。

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=max(5, patience // 4))
...
scheduler.step(val_loss)
```

也支持 `cosine`。**注意**：`ReduceLROnPlateau` 要传验证损失，
余弦退火不传参数——写错了不会报错，但调度会失效。

### 8.7 早停与"恢复最佳权重"

**早停**：验证损失连续 `patience` 轮不改善就停。

**最容易犯的错误**：停下来之后直接用**最后一轮**的权重。
最后一轮恰恰是"不再改善的那一轮"，比最佳轮次差。

正确做法是**在内存里保存最佳轮次的权重快照**，最后恢复：

```python
if improved:
    best_val = val_loss
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
...
model.load_state_dict(best_state)
```

> 代码：`pdbenergy/train.py::train_model`

### 8.8 梯度裁剪

消息传递网络偶尔会产生很大的梯度（尤其遇到高能畸变结构）。
**梯度裁剪**把梯度总范数限制在阈值内：

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
```

`clip_grad_norm_`（裁剪总范数）保持梯度方向不变，通常比
`clip_grad_value_`（逐元素裁剪）更好。

### 8.9 可复现性

```python
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
```

诚实的说明：

* CPU 训练在这里是**确定性的**；
* GPU 上要完全可复现还需 `torch.use_deterministic_algorithms(True)`，
  且部分 cuDNN 卷积核本质非确定；
* DataLoader 的 shuffle 也要用带种子的 `torch.Generator`；
* 环境差异（BLAS 版本、线程数）也会带来末位差异。

### 8.10 检查点里应该放什么

| 内容 | 为什么 |
|---|---|
| `model_state_dict` | 权重 |
| `model_config` | 才能重建相同的网络结构 |
| `feature_config` | 才能重建相同维度的输入（cutoff、n_rbf、键标记） |
| `norm_mean` / `norm_std` | 反标准化预测值，**不存就没法推理** |
| `protein_splits` | 保证评估切分与训练完全一致 |
| `target` / `split_mode` | 说明这个模型到底在预测什么 |
| `history` | 学习曲线，便于复现分析 |

> 代码：`train.py::save_checkpoint`、`train.py::load_checkpoint`

### 8.11 一个真实的形状 bug（值得记住）

第一版训练直接报：

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (146068x49 and 65x128)
```

原因：特征生成用了 `features.n_rbf = 48`，模型却用了 `model.n_gaussians = 64`。
**同一种东西存在两个地方**，就会不一致。

修法不是把 64 改成 48，而是**消除重复来源**：
删除 `ModelConfig.n_gaussians`，让模型从特征配置推导
（`models.py::edge_dims_for`）。教训是通用的：
**同一个物理量只允许有一个定义处**。

### 8.12 输出层初始化：一个被低估的细节

这是本项目里**性价比最高的一个修复**。

网络最后是"每原子能量求和"（§7.5）。如果读出头（readout）用默认随机初始化，
那么：

$$E_{\text{pred}} = \sum_{i=1}^{700} \varepsilon_\theta(\mathbf{h}_i)$$

**700 个均值 0、方差 $\sigma^2$ 的随机数之和，标准差是 $\sqrt{700}\,\sigma \approx 26\sigma$**，
一点也不小。于是未训练的模型一上来就预测出偏离均值几百 kcal/mol 的值。

实测（本项目，标准化后的目标）：

| | 第 1 轮验证 MAE | 备注 |
|---|---|---|
| 默认初始化 | **702 kcal/mol** | 比"永远输出均值"还差 6 倍 |
| 把读出头最后一层**置零** | **142 kcal/mol** | 正好等于常数预测器 |

修法只有两行：

```python
for head in (self.atom_readout, self.global_head):
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
```

这样初始预测**恰好等于标准化目标的均值（0）**，也就是常数预测器——
对标准化后的回归目标来说，这是唯一正确的起点。
优化器只需要"加上结构"，不需要先花几十轮把随机的尺度压回去。

**一般化的教训**：当你把很多个小量**求和**成一个输出时，
初始化必须保证"求和结果也是小量"。任何"聚合 N 个东西"的输出层都该这样处理。

### 8.13 继续训练：热启动、断点续训，以及一个必须冻结的东西

当你有更多 PDB 文件时，"再训一遍"有三种语义完全不同的做法：

| 做法 | 权重 | 优化器状态 | epoch 计数 | 适用场景 |
|---|---|---|---|---|
| 从头训练 | 随机 | 全新 | 从 1 开始 | 换了结构、换了目标 |
| **热启动** | **从检查点** | 全新 | 从 1 开始 | **数据变多，在旧模型上继续提升** |
| **断点续训** | 从检查点 | **恢复** | **接着数** | 长训练被中断 |

对应 `train --init-from <ckpt>` 与 `train --resume <run_dir>`；
`python -m pdbenergy.iterate` 把多轮循环包了起来。

#### 为什么"断点续训"必须单独存优化器状态

AdamW 每个参数维护**两个动量**（一阶矩 $m$、二阶矩 $v$，见 §8.5）。
只恢复权重、不恢复动量，优化器等于"失忆重启"：自适应步长要重新估计，
接下来一段会有明显的性能回退。

所以本项目把检查点拆成两个文件：

* `checkpoint.pt` —— 只放**推理**需要的（权重 + 模型/特征配置 + 归一化统计量），保持干净；
* `train_state.pt` —— 放**继续训练**需要的（优化器状态、epoch 计数、完整历史），
  体积约 3 倍，但只有续训时才用。

`tcfg.epochs` 的语义也随之明确：**永远是"这次调用再跑多少轮"**，
不是"总共跑多少轮"。否则续训会静默重复已经跑过的 epoch。

#### 归一化的尺度陷阱（和 §8.12 是同一类问题）

检查点里存着它训练时用的目标 mean/std。如果续训时用**新数据重新算**，
输出层就处在"按旧单位标定、却在新单位下被评价"的状态——
前几轮全花在把尺度掰回来，白费。

所以 `--init-from` / `--resume` **默认沿用检查点里的归一化**，并打印新旧对比；
当新训练集的目标均值偏离旧均值超过 0.25 个旧标准差时给出警告，
提示你可能该用 `--recompute-normalisation`。

#### 改结构就不能热启动——而且必须**报错**，不能静默

`hidden_dim`、`n_interactions`、`cutoff`、`n_rbf` 里任何一个变了，权重形状就对不上。
最坏的结果不是报错，而是**只加载了一部分张量、剩下的保持随机**——
那会得到一个看起来能跑、实际半随机初始化的模型。

所以本项目在 `load_state_dict` **之前**先逐字段比对模型配置和特征配置，
列出到底哪个字段变了，然后拒绝执行：

```
cannot continue from 'outputs/old/checkpoint.pt':
  - feature config differs (n_rbf: 32 -> 64) -> input edge layout changed
```

**这类"配置兼容性检查"是任何支持续训的训练框架都该有的东西。**

#### 迭代训练里最容易被忽略的一点：**必须冻结切分**

`split_proteins` 是**从蛋白列表推导**切分的。所以每加一批新蛋白，
切分就会重新洗牌——train/val/test 的成员都变了。

后果是：**跨轮的测试指标不可比**。这一轮 "test MAE 从 120 降到 95"，
可能完全不是因为模型变好，而只是某个难蛋白恰好被洗出了测试集。
这和 §9.1 是同一类错误：**让评估集移动，等于把评估作废。**

正确做法（本项目 `iterate.py` 的做法）：

1. 第 1 轮算出切分，**写进 `split.json` 冻结**；
2. 之后每一轮都读这个文件；
3. **新出现的蛋白只追加到 train**，绝不进 val/test。

这样测试集在每一轮里都是**同一批蛋白**，指标才真的在度量"模型有没有变好"。
另外，选最优检查点只看 **val**，test 只报告——见 §9.6。

> 代码：`pdbenergy/iterate.py::freeze_split`、`apply_split`；
> 兼容性检查在 `pdbenergy/train.py::_check_continuation_compatible`

---

## 9. 评估方法学：数据泄漏与指标陷阱

> 这一章是本教程最重要的部分。**一个做错的评估比没有评估更糟**，
> 因为它会给你虚假的信心。

### 9.1 按帧切分 = 数据泄漏（最严重的问题）

**错误做法**：把所有 (构象, 能量) 样本随机打乱，8:1:1 切成训练/验证/测试。

**为什么错**：同一个蛋白的相邻 MD 快照相隔 0.2–1 ps，
结构差异只有零点几埃，能量几乎一样。随机切分会把
"某个构象的近亲副本"放进训练集，把"它自己"放进测试集。
模型只要**记住**这个近邻就行，测试误差看起来很漂亮，但**它什么都没学会**。

**正确做法**：**按蛋白质切分**。测试集里的蛋白在训练中**从未出现过**。
这才是真实使用场景："给我一个没见过的新蛋白结构，预测它的构象能量。"

本项目 `dataset.py::split_proteins` 默认按蛋白切分
（`train.split_mode = "protein"`）；另一个选项 `"frame"` 故意保留，
用来**量化泄漏有多大**（§9.2）。

### 9.2 用消融实验量化泄漏

`ablate` 命令用**完全相同的配置**只改切分方式训练两次：

```powershell
python -m pdbenergy.cli ablate
```

两者测试 MAE 的比值就是**泄漏带来的虚假提升倍数**。
实测结果见 §11.2。这个实验把"数据泄漏"从抽象概念变成了一个**数字**。

### 9.3 回归指标：含义与陷阱

| 指标 | 公式 | 含义 | 陷阱 |
|---|---|---|---|
| **MAE** | $\frac1n\sum\lvert y-\hat y\rvert$ | 平均绝对误差，kcal/mol | 最直观，但对大误差不敏感 |
| **RMSE** | $\sqrt{\frac1n\sum(y-\hat y)^2}$ | 对大误差敏感 | 少数离群点就能拉高；与 MAE 差距大说明有离群点 |
| **R²** | $1 - \frac{\sum(y-\hat y)^2}{\sum(y-\bar y)^2}$ | 解释了多少方差 | **会被"区分蛋白"这个简单任务撑高**（见下） |
| **Pearson $r$** | 线性相关 | 线性关系强度 | 对系统偏差不敏感（$\hat y = y + 100$ 的 $r$ 仍是 1） |
| **Spearman $\rho$** | 秩相关 | 排序一致性 | 最贴近"打分/排序"的实际需求 |
| **bias** | $\frac1n\sum(\hat y - y)$ | 系统偏差 | 诊断整体高估/低估 |

**R² 的关键陷阱**：不同蛋白的 ΔE 分布范围差别很大。
模型只要学会"认出这是哪个蛋白、输出它的平均能量"，
就能获得很高的全局 R²——但它**完全没有学会构象之间的能量差别**，
而后者才是我们真正想要的能力。

### 9.4 组内排序相关性：真正该看的指标

针对上面的陷阱，本项目增加了**按蛋白分组**的 Spearman 相关：

$$\bar\rho = \frac{1}{|\mathcal{P}|}\sum_{p \in \mathcal{P}} \rho_{\text{Spearman}}\big(\{y_i\}_{i\in p},\ \{\hat y_i\}_{i\in p}\big)$$

含义是：**在同一个蛋白内部，模型能不能把好构象排在坏构象前面**。
直接对应打分函数的实际用途。

同时报告 `fraction_above_0.5`（有多少蛋白的相关性超过 0.5），
避免平均值被个别容易的蛋白拉高。

> 代码：`pdbenergy/evaluate.py::rank_discrimination`

### 9.5 分组分解：平均值会掩盖失败

1. **每蛋白 MAE**：是不是某个蛋白特别差？是不是小蛋白容易、大蛋白难？
2. **每来源 MAE**：`torsion` / `torsion_min` / `md*K` / `nmr` 的误差。
   模型很可能在低能的 `torsion_min` 上很准、在高能的 `md450K` 上很差。
   **不分解就永远不知道模型在哪一区失效。**
3. **能量范围检查**：预测值的最大/最小值 vs 真实值。
   如果预测范围被**压缩**在均值附近（regression to the mean），
   说明模型欠拟合了能量范围，即使 R² 看起来还行。

### 9.6 必须有的基线

没有基线的指标没有意义。本项目提供两个：

1. **全局均值基线**：永远预测训练集平均 ΔE。任何模型都必须显著优于它。
2. **手工描述符 + MLP**：§6.7 的 24 维物理特征。
   如果 GNN 不能显著超过它，说明图结构的价值在这个数据集上还没体现出来。

### 9.7 必看的图

| 图 | 看什么 |
|---|---|
| **Parity（预测 vs 真实）** | 点云是否贴着 $y=x$？是否被压缩？有没有系统弯曲？ |
| **残差分布** | 是否以 0 为中心？尾部有多重？ |
| **残差 vs 真实值** | **异方差性**：是不是能量越高误差越大？ |
| **学习曲线** | 训练/验证是否分离（过拟合）？是否还在下降（欠训练）？ |
| **能量分布** | **训练前第一件该看的事**：数据覆盖了哪个区间？ |
| **每蛋白/每来源柱状图** | 哪里在失败 |

> 代码：`pdbenergy/evaluate.py` 的 `plot_*` 函数，输出到 `outputs/<run>/eval/`

---

## 10. 代码导读

### 10.1 模块职责

| 模块 | 行数级别 | 职责 | 关键函数 |
|---|---|---|---|
| `pdbio.py` | ~450 | PDB 固定列格式解析/写出、元素推断、altLoc、清洗 | `parse_pdb_string`, `write_pdb`, `protein_only`, `infer_element` |
| `prepare.py` | ~250 | 从 RCSB 下载、扫描体检 | `download_pdb`, `inventory` |
| `labels.py` | ~430 | OpenMM 力场引擎 | `protonate`, `prepare`, `energy_kcal`, `energy_terms_kcal`, `minimise`, `run_md` |
| `ensemble.py` | ~670 | 构象系综生成 | `find_rotatable_bonds`, `rotate_torsion`, `build_ensemble`, `build_all` |
| `dataset.py` | ~330 | 数据装配、按蛋白切分、标准化 | `load_ensembles`, `split_proteins`, `build_bundle` |
| `features.py` | ~390 | 结构→图、描述符基线 | `pairwise_distances`, `frame_to_graph`, `global_descriptors`, `ConformerDataset`, `collate_graphs` |
| `graphcache.py` | ~120 | 图缓存 | `load_or_build`, `feature_signature` |
| `models.py` | ~290 | SchNet + MLP | `GaussianRBFExpansion`, `InteractionBlock`, `SchNetRegressor`, `scatter_sum` |
| `train.py` | ~480 | 训练循环、指标、检查点、热启动/续训 | `train_model`, `regression_metrics`, `save_checkpoint`, `load_checkpoint`, `_check_continuation_compatible` |
| `evaluate.py` | ~430 | 评估与绘图 | `evaluate_run`, `rank_discrimination`, `plot_*` |
| `predict.py` | ~230 | PDB → 能量推理 | `EnergyPredictor.predict_file` |
| `leakage.py` | ~150 | **不依赖模型**地直接测量训练/测试重叠 | `nearest_neighbour_leakage`, `compare_split_modes` |
| `iterate.py` | ~430 | 迭代训练：冻结切分 + 多轮热启动 + 轮次登记 | `freeze_split`, `apply_split`, `run_rounds`, `IterationRegistry` |
| `cli.py` | ~500 | 命令行 | `main` |

### 10.2 完整命令流程

```powershell
# 0) 环境
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# 1) 数据
python -m pdbenergy.cli download          # 16 个内置 PDB 条目
python -m pdbenergy.cli inventory         # 体检：残基数、链数、模型数、是否可用

# 2) 物理标签（最慢的一步）
python -m pdbenergy.cli ensemble --threads 8
#   也可跳过部分： --limit 2（只用前两层二面角幅度） / --overwrite

# 3) 数据集与训练
python -m pdbenergy.cli dataset
python -m pdbenergy.cli train --model schnet
python -m pdbenergy.cli train --model mlp

# 4) 评估与消融
python -m pdbenergy.cli evaluate --run-dir outputs/schnet_protein
python -m pdbenergy.cli ablate --epochs 60

# 5) 推理
python -m pdbenergy.cli predict data/raw/1L2Y.pdb --max-models 5 --verify
```

### 10.3 配置项与理由

**`label`（物理引擎）**

| 项 | 默认 | 理由 |
|---|---|---|
| `forcefield_files` | `amber14-all.xml`, `implicit/gbn2.xml` | AMBER14SB 是蛋白质模拟的主流全原子力场；GBn2 是较新的 GB 变体 |
| `implicit_solvent` | `OBC2` | 决定加载哪个 implicit XML；OpenMM 8.x 里不是 `createSystem` 的参数 |
| `nonbonded_method` | `CutoffNonPeriodic` | 比 `NoCutoff` 快 3.6 倍（实测），1.0 nm 是隐式溶剂 MD 的惯例 |
| `cutoff_nm` | `1.0` | 惯例值 |
| `constraints` | `HBonds` | 固定 X–H 键，允许 2 fs 时间步 |

**`ensemble`（采样）**

| 项 | 默认 | 理由 |
|---|---|---|
| `torsion_levels` | `(1.0, 1.5, 2.0, 3.0)` | 4 个探索幅度层级 |
| `n_torsion_per_level` | `12` | 每层 12 个样本 → 48 帧/蛋白 |
| `torsion_min_deg` / `max_deg` | `55` / `165` | 从轻微到大幅构象变化 |
| `cartesian_jitter` | **`0.0`** | **必须为 0**：0.04 Å 就引入约 800 kcal/mol 的键长噪声（§5.1） |
| `minimise_fraction` | **`0.0`** | 二面角帧已覆盖低能区，部分最小化纯粹是时间开销（§5.6） |
| `reference_minimise_iterations` | `120` | 参考极小值定义能量零点，值得算准 |
| `max_relative_energy` | `250` | 二面角/NMR 帧的冲突筛选阈值 |
| `md_max_relative_energy` | `1500` | MD 帧的宽松阈值（均分定理，§5.7） |
| `md_temperatures` | `(300, 450)` | 覆盖热可及构象 |
| `md_ps_per_temperature` | `0.6` | 速度与覆盖的折中 |
| `max_nmr_models` | `3` | 每个 NMR 条目最多用 3 个模型（每个都要短最小化） |

**`features` / `model` / `train`**

| 项 | 默认 | 理由 |
|---|---|---|
| `cutoff` | `4.0` Å | 覆盖键（~1.5 Å）、角（~2.4 Å）与多数 1-4 对（~3.0–3.9 Å）；开销约随 cutoff³ 增长 |
| `n_rbf` | `32` | 0–5 Å 上约 0.161 Å 间隔；基函数本身稀疏，够用 |
| `max_neighbors` | `32` | 内存安全阀 |
| `use_bond_feature` | `True` | 键信息是化学先验，免费 |
| `kind` | `schnet` | 消息传递网络 |
| `hidden_dim` | `64` | 实测 128 维每步 15.6 s、64 维约 1/3 开销；750 个样本也撑不起大模型 |
| `n_interactions` | `3` | 感受野 ≈ 12 Å；层数是第二大开销来源 |
| `loss` | `huber` | 抗离群点（§8.4） |
| `split_mode` | `protein` | 避免数据泄漏（§9.1） |
| `cache_graphs` | `True` | 消掉每轮重复的建图开销（§8.2） |
| `batch_size` | `16` | 受内存限制；每 batch 的边数约 10⁵–10⁶ |
| `patience` | `40` | 早停耐心值 |

---

## 11. 实测结果

### 11.1 数据集

| 项目 | 值 |
|---|---|
| 蛋白数 | 14 |
| 总样本数 | 730 |
| 训练 / 验证 / 测试样本 | 323 / 199 / 208 |
| 切分方式 | 按蛋白质（`protein`） |
| 预测目标 | relative |
| 训练集目标均值 / 标准差 | 78.79 / 144.89 kcal/mol |

**测试集蛋白（训练中完全没有出现过）：** `1BDD, 2GB1, 3GB1, 5PTI`

**验证集蛋白：** `1PGB, 1RIS, 1UBQ, 1VII`

**训练集蛋白：** `1CRN, 1ENH, 1FME, 1L2Y, 1SHG, 2CI2`


**每个蛋白的规模与能量跨度**（按 ΔE 跨度降序）：

| PDB | 划分 | 残基数 | 原子数 | 帧数 | E_min (kcal/mol) | 最大 ΔE (kcal/mol) |
|---|---|---|---|---|---|---|
| 1RIS | val | 97 | 1647 | 50 | -6449.8 | 1169.6 |
| 1UBQ | val | 76 | 1231 | 49 | -3584.7 | 890.8 |
| 2CI2 | train | 65 | 1076 | 49 | -3063.8 | 817.0 |
| 1BDD | test | 60 | 941 | 54 | -2721.1 | 708.6 |
| 1ENH | train | 54 | 947 | 52 | -4318.9 | 700.5 |
| 1SHG | train | 57 | 955 | 54 | -2400.0 | 697.3 |
| 5PTI | test | 58 | 892 | 54 | -2814.7 | 683.1 |
| 2GB1 | test | 56 | 855 | 50 | -2108.1 | 681.7 |
| 1PGB | val | 56 | 855 | 46 | -2063.5 | 641.8 |
| 3GB1 | test | 56 | 855 | 50 | -2129.9 | 627.7 |
| 1CRN | train | 46 | 642 | 54 | -1456.4 | 529.7 |
| 1VII | val | 36 | 596 | 54 | -1307.5 | 404.9 |
| 1FME | train | 28 | 504 | 57 | -2047.4 | 404.0 |
| 1L2Y | train | 20 | 304 | 57 | -723.2 | 248.3 |

### 11.2 模型对比（测试集 = 完全没见过的蛋白）

| 模型 | 参数量 | 最佳轮次 | MAE | RMSE | R² | Spearman | 组内平均 ρ | 常数基线 MAE |
|---|---|---|---|---|---|---|---|---|
| SchNet（图神经网络） | 42,562 | 18 | 121.06 | 161.26 | 0.0622 | 0.4306 | 0.1467 | 107.29 |
| MLP（手工描述符基线） | 9,985 | 6 | 105.97 | 166.67 | -0.0018 | — | — | 107.29 |

> 单位除 R²/Spearman 外均为 kcal/mol。"常数基线"= 永远输出训练集平均 ΔE，任何模型都必须显著优于它。"组内平均 ρ"= 在同一个蛋白内部，预测排序与真实排序的平均 Spearman 相关，这才是打分函数真正需要的指标（§9.4）。

### 11.3 按采样来源分解（测试集）


**SchNet**

| 来源 | 样本数 | MAE | RMSE | Spearman |
|---|---|---|---|---|
| md450K | 12 | 494.42 | 496.25 | 0.720 |
| md300K | 12 | 279.79 | 283.38 | -0.189 |
| torsion | 181 | 86.66 | 90.29 | 0.360 |
| nmr | 3 | 68.23 | 70.35 | 0.500 |

**MLP 基线**

| 来源 | 样本数 | MAE | RMSE | Spearman |
|---|---|---|---|---|
| md450K | 12 | 550.28 | 553.11 | — |
| md300K | 12 | 325.65 | 327.31 | — |
| torsion | 181 | 63.09 | 67.16 | -0.351 |
| nmr | 3 | 37.55 | 41.83 | — |

### 11.4 按蛋白分解（测试集，SchNet）

| PDB | 样本数 | MAE | RMSE | R² | Spearman |
|---|---|---|---|---|---|
| 1BDD | 54 | 126.68 | 158.90 | 0.095 | 0.203 |
| 3GB1 | 50 | 122.85 | 162.62 | -0.040 | -0.121 |
| 2GB1 | 50 | 119.22 | 162.11 | 0.069 | 0.442 |
| 5PTI | 54 | 115.49 | 161.55 | 0.098 | 0.063 |

> 组内 Spearman 高，说明「能不能把好构象排在前面」这件事做得好；R² 甚至可能为负，说明该蛋白上的预测还不如直接用平均值。**这种分解是必需的**——总体指标会把这种失败完全掩盖（§9.5）。

### 11.5 消融实验：数据泄漏

**直接测量（不需要训练模型）**

| 切分方式 | 训练/测试帧数 | 最近邻同蛋白比例 | 最近邻距离中位数 | 最近邻距离最小值 | 近似重复比例 (<0.05) |
|---|---|---|---|---|---|
| 按蛋白质切分（本项目默认） | 323 / 208 | 0.0% | 4.775 | 4.250 | 0.0% |
| 按帧随机切分（常见错误） | 366 / 182 | 97.3% | 0.044 | 0.000 | 52.7% |

按帧切分时，**97% 的测试帧在训练集里的最近邻来自同一个蛋白**，其中 **53% 与训练样本的距离小于 0.05**（标准化描述符空间），也就是说**一半以上的「测试集」其实是训练集的近似副本**。而按蛋白质切分时这两个数字都是 **0%**。


**训练两种切分下的模型（同样的配置、同样的轮数）**

| 切分方式 | 模型 | 测试 MAE | 测试 RMSE | 测试 R² | 测试 Spearman | 最佳验证 MAE |
|---|---|---|---|---|---|---|
| 按蛋白质切分（正确） | mlp | 105.97 | 166.67 | -0.0018 | — | 117.29 |
| 按帧随机切分（泄漏） | mlp | 111.28 | 185.79 | -0.0083 | 0.0164 | 113.97 |

两个 arm 的测试 MAE 之比为 **0.95**（111.28 vs 105.97 kcal/mol）。


> **为什么 MAE 的差距没有想象中大？** 因为泄漏能被「利用」多少，取决于模型有没有能力把近似重复的帧**记下来**。这里用的是 1 万参数的描述符 MLP（跑得快），它本来就不擅长记忆单个构象，所以 MAE 差距被压缩了。
>
> 这恰恰是重点：**MAE 差距是一个依赖模型的间接证据，而上面那张「直接测量」表不依赖任何模型**——它数的是训练集和测试集到底有多重叠。看到 97%/53% 这两个数字，就不需要再靠训练模型来说服自己了。


### 11.6 产物文件

| 文件 | 内容 |
|---|---|
| `outputs/schnet_protein/checkpoint.pt` | 权重 + 模型配置 + 特征配置 + 归一化统计量 + 切分清单（自解释检查点） |
| `outputs/schnet_protein/eval/metrics.json` | 全部指标：总体 / 每蛋白 / 每来源 / 排序能力 / 常数基线 |
| `outputs/schnet_protein/eval/parity_test.png` | 预测 vs 真实散点图（按蛋白着色） |
| `outputs/schnet_protein/eval/residuals_test.png` | 残差分布 + 残差对真实值（查异方差） |
| `outputs/schnet_protein/eval/learning_curve.png` | 训练/验证损失、验证 MAE、学习率 |
| `outputs/schnet_protein/eval/per_protein_test.png` | 每个测试蛋白的 MAE |
| `outputs/schnet_protein/eval/per_source_test.png` | 每种采样来源的 MAE |
| `outputs/schnet_protein/eval/energy_distribution.png` | 目标分布与各来源的箱线图 |
| `outputs/ablation/split_leakage.json` | 泄漏消融的原始数字 |
| `outputs/predictions.json` | 对真实 PDB 文件的预测（可用 `--verify` 附上真值） |

### 11.7 自动诊断（直接由上面的数字推出）

**SchNet**

- 预测区间 84.1–186.3 kcal/mol，真实区间 0.0–708.6 kcal/mol；**预测只覆盖了真实跨度的 14%**，即典型的**回归到均值**。
- 测试 MAE 121.06 kcal/mol，常数基线 107.29 kcal/mol → **不如**常数预测器。
- 全局 Spearman ρ = 0.431，组内平均 ρ = 0.147（中位数 0.133，超过 0.5 的蛋白比例 0%）。
- 按来源：最准的是 `nmr`（MAE 68.2），最差的是 `md450K`（MAE 494.4）。

**MLP 基线**

- 预测区间 76.5–76.5 kcal/mol，真实区间 0.0–708.6 kcal/mol；**预测只覆盖了真实跨度的 0%**，即典型的**回归到均值**。
- 测试 MAE 105.97 kcal/mol，常数基线 107.29 kcal/mol → 预测值几乎不变，**它就是常数预测器本身**，差别只来自用哪个常数。
- 全局 Spearman ρ = nan，组内平均 ρ = nan（中位数 nan，超过 0.5 的蛋白比例 nan%）。
- 按来源：最准的是 `nmr`（MAE 37.5），最差的是 `md450K`（MAE 550.3）。

### 11.8 怎么读这些数字

1. **先看常数基线。** 如果 MAE 和常数基线差不多，模型实际上什么都没学到，
   后面的 R² 再好看也没有意义。
2. **再看预测区间。** 如果预测值只覆盖真实区间的一小部分，
   说明模型在**回归到均值**——这是"还没学会"的典型形态，而不是"学会了但不够准"。
3. **再看组内 ρ。** 这是"同一蛋白内部排序能力"，直接对应实际用途。
   全局 ρ 高但组内 ρ 低，意味着模型主要是学会了区分不同蛋白/不同采样来源，
   而不是学会了判断构象好坏。
4. **然后看按来源分解。** 低能区（`torsion`）通常明显比高温 MD 区准，
   因为高温区的构象形变更大、能量更高，而训练样本更少。
5. **最后看 Parity 图和残差图。** 点云被压向均值 → 回归到平均；
   残差随真实能量增大而增大 → 高能区拟合不足（异方差）。
6. **和泄漏消融对照。** 如果按帧切分的数字漂亮得多，
   那说明之前所有的乐观结论都不可信。

**本项目实测结论（诚实版）**：这条流水线是**完整可用**的——
从 PDB 下载、力场打标签、构象采样、按蛋白切分，到训练、评估、推理全部跑通，
产出的模型**确实带有一点真实的排序信号**（全局 Spearman 明显大于 0），
但**绝对精度不足以当作可用的打分函数**：预测区间被压缩、MAE 没有超过常数基线。

原因不是代码错误，而是**数据量**：只有 6 个训练蛋白、323 个构象，
而要泛化到没见过的蛋白，模型必须从几何细节里读出"热形变到什么程度"。
真实 MLIP 数据集是 $10^5$–$10^7$ 量级（§12.1）。
具体该往哪个方向补，见 §12.3。

---

## 12. 局限、失败模式与后续方向

### 12.1 本项目的局限

1. **标签是力场，不是实验，也不是量子化学。**
   我们学到的是"AMBER14+GBn2 会给出什么数"，而不是"真实能量"。
   力场本身有系统性误差（例如对无序区、金属位点、非标准残基都不准）。
   模型的上限就是力场的上限。

2. **cutoff 截断了长程物理。**
   4.5 Å 的消息传递半径意味着静电长程尾部学不到。
   对小的、紧凑的蛋白影响有限；对大蛋白或带电很多的体系会变差。

3. **只用了 14 个蛋白、约 750 个样本。**
   这足以验证方法（能训练、能泛化到新蛋白），
   但远不足以得到一个**可用**的势函数。
   真实 MLIP 数据集是 10⁵–10⁷ 量级（ANI-1x 是 5×10⁶，SPICE 是 1.1×10⁶）。

4. **没有预测力（forces）。**
   力 = $-\nabla E$，是 MD 和几何优化真正需要的东西。
   虽然神经网络可以自动微分直接给出解析梯度，
   但拿能量训练的模型，其**梯度精度往往明显差于能量精度**
   （这是 MLIP 领域的已知问题，通常要靠显式加入力的监督来修）。

5. **没有处理多链复合物、配体、金属、翻译后修饰。**
   清洗步骤会把它们全部丢掉。

6. **pH 依赖。** 质子化态由 `prepare.ph` 决定，是超参数。
   换 pH 会改变 His/Asp/Glu/Lys/Cys 的质子化，从而改变标签分布。

### 12.2 值得记录的失败模式

| 现象 | 根因 | 修法 |
|---|---|---|
| `No template found for residue 0 (THR)` | `write_pdb` 在每个残基间写了 `TER`，把每个残基变成独立链 | 只在链 ID 变化时写 `TER`（§3.9） |
| `argument 'implicitSolvent' was never used` | OpenMM 8.x 改用 XML 内的 `<Script>` | 去掉该参数，靠加载哪个 implicit XML 决定（§4.6） |
| `argument 'cutoff' was never used` | 参数名应为 `nonbondedCutoff`（命名参数），不是 `cutoff` | 用正确名字 |
| ΔE 最大 $5\times10^9$ kcal/mol | 二面角旋转造成原子重叠，LJ $r^{-12}$ 爆炸 | 逐帧能量筛选 `max_relative_energy`（§5.5） |
| 所有 MD 帧被筛掉（0/5） | MD 从"未最小化的加氢结构"出发；且 250 kcal/mol 的阈值低于均分能量 | 从极小值出发 + MD 用独立宽松阈值（§5.7） |
| 加 0.04 Å 噪声后 ΔE 中位数 797 | 键长拉伸噪声，且 HBonds 约束让它无法被最小化修掉 | `cartesian_jitter = 0`（§5.1） |
| `mat1 and mat2 shapes cannot be multiplied (x49 and 65x128)` | `features.n_rbf=48` 与 `model.n_gaussians=64` 不一致 | 单一来源：`edge_dims_for`（§8.11） |
| 第 1 轮验证 MAE 702 kcal/mol（比常数预测器还差 6 倍） | 读出头随机初始化，700 个原子求和把尺度放大了 26 倍 | 读出头最后一层置零（§8.12） |
| NMR 帧保留率 0/38 | 把 NMR 重原子坐标覆盖到别的体系继承来的氢上，C–H 键被拉开近 1 Å | 每个模型**重新加氢**，不要搬运氢（§5.8） |
| 一个大蛋白卡住十几分钟 | NMR 循环按"成功数"计数，被拒绝时会走遍全部 30+ 个模型 | 改为限制**尝试次数**（§5.8） |
| 每 epoch 110 秒 | 每轮重复 featurisation | 图缓存 + 把 RBF 挪进模型（§8.2） |
| 续训后前几轮明显变差 | 只恢复了权重、没恢复 AdamW 的两个动量，优化器"失忆重启" | 优化器状态存进 `train_state.pt`（§8.13） |
| 热启动后损失先炸再降 | 新旧数据集的目标 mean/std 不同，输出层按旧尺度标定 | 默认沿用检查点的归一化；必要时 `--recompute-normalisation`（§8.13） |
| 换了网络结构后续训"能跑但很差" | 形状不匹配时只加载了部分张量，其余保持随机 | 加载前逐字段比对配置并报错（§8.13） |
| 每加一批数据"测试指标就变好" | `split_proteins` 从蛋白列表推导切分，加数据会重新洗牌 | 第 1 轮冻结切分，新蛋白只进 train（§8.13） |

### 12.3 如果能继续做下去

> 下面按"实测证据"排序，而不是按"听起来高级"排序。
> 三条实测结论先说：**加数据没用、加轮数没用、加密特征有用但不解决根本问题**。
> 原始怀疑（RBF 分辨率太粗）和原始判断（数据量是根本瓶颈）都做了实验，
> 结果是一个被证实、一个被否掉。

**① 原始假设：RBF 分辨率太粗 —— 已实测，部分成立**

MD 帧的 ΔE 主要是**局部的键长/键角热形变**（能量均分）。
但本项目的径向基**间距是 0.161 Å**（`rbf_end - rbf_start = 5`，`n_rbf = 32`），
而 300–450 K 下键长的热涨落只有 **0.05–0.1 Å**——
**特征的几何分辨率比要检测的信号还粗**。这个怀疑是对的，加密后 MAE 确实降了（见 ③），
但它**没有解决排序能力**，所以不是全部答案。

顺带记一个坑：把 `rbf_end` 缩小到小于 `cutoff` 会**静默毁掉长程边**——
窄基函数在走到 cutoff 之前就已下溢为零（`n_rbf=64` 铺在 0–3 Å 时间距只有 0.048 Å，
离最后一个中心 1 Å 就是 $\exp(-220)$），于是 $(r_{\text{end}}, r_{\text{cutoff}}]$
的边只剩一个恒为 0 的键标记。**这看起来像"模型更粗"，实际是配置坏了。**
现在 `GaussianRBFExpansion` 会直接报错而不是默默变差。

**② 数据量：已实测——不是当前的瓶颈**

原本这里是"根本瓶颈"。**实测否掉了这个判断**，所以改写如下。

用 `scripts/learning_curve.py` 在**冻结切分**下把训练集按 25% / 50% / 100% 取子集
（val/test 蛋白固定，同种子、同轮数）：

| 训练构象数 | 训练损失 | 验证损失 | 验证 MAE |
|---|---|---|---|
| 81（25%） | 0.3052 | 2.0229 | 128.56 |
| 160（50%） | 0.3147 | 2.2538 | **143.02** |
| 323（100%，4 轮） | 0.3158 | — | 129.53 |
| 323（100%，20 轮，出厂模型） | — | — | 136.01 |

**数据翻倍，验证 MAE 反而变差**；而且**训练损失在任何数据量下都是 0.31**——
模型总能拟合手上的训练集，验证集却纹丝不动。这不是"数据不够"的形态，
是"学到的东西不迁移"的形态。轮数也不是答案：4 轮和 20 轮的结果一样。

> 结论：**在这个设置下，加 10 倍数据不会自动变好。** 先修 ① 和 ③，再谈加数据。

**③ 特征分辨率：实测有效，但代价被忽略了**

把基函数从 `n_rbf=32`（间距 0.161 Å）加密到 `n_rbf=96` 并把范围收到 `rbf_end=4.0`
（间距 0.042 Å，与 cutoff 对齐），**同数据、同切分、同 4 轮预算**：

| 变体 | val MAE | test MAE | test R² | **test Spearman** | 组内 ρ | **预测跨度** | train loss | 耗时 |
|---|---|---|---|---|---|---|---|---|
| `n_rbf=32, rbf_end=5.0`（出厂） | 129.53 | 121.39 | 0.002 | **0.430** | 0.169 | **2.4%** | 0.3158 | 358 s |
| `n_rbf=96, rbf_end=4.0` | **114.30** | **102.00** | 0.003 | **−0.036** | 0.184 | **2.0%** | 0.3093 | 868 s |
| 常数预测器 | 118.28 | 107.29 | — | — | — | — | — | — |

**这是本项目第一次有模型在 val 和 test 上同时超过常数预测器**，MAE 提升 12–16%，
而且训练损失几乎没变（0.3158 → 0.3093）——提升来自泛化，不是拟合。

**但是排序能力塌了：全局 Spearman 0.430 → −0.036。**
预测跨度仍然只有真实能量的 2%。

也就是说：**它靠"把预测压得更接近均值"降低了 MAE，代价是丢掉了原本那点排序信号。**

> ### 这是本项目最重要的一课
>
> **MAE 单独看会骗人。** 一个几乎只输出常数的模型，
> 只要那个常数比别人的常数稍微好一点，MAE 就能"赢过常数基线"。
> 真正的证据是**预测跨度**（2% ——说明它基本在输出常数）和**组内 Spearman**
> （0.17 ——说明它不会在同一个蛋白内部排序）。
>
> 所以本项目的 `metrics.json` / 消融输出同时给出这两个量，
> 而 §11.8 第 2 条要求**先看预测跨度再看 MAE**。
> 任何只报 MAE 的分子能量模型评测，都值得用这两个指标复核一遍。

**④ 目标的设计（下一步该动的地方）**

实测已经把注意力从"数据量"推到了"目标本身"。$\Delta E$ 现在混合了两件性质不同的东西：

* **热激发幅度**：MD 帧比极小值高 300–1000 kcal/mol（能量均分，§5.7），
  而且**随蛋白大小线性增长**；
* **构象形变**：二面角帧通常只有 0–100 kcal/mol。

模型要从**亚 0.1 Å 的几何细节**同时预测这两者，动态范围还跨越两个数量级——
难的不是"缺样本"，是**目标把"大小"和"形状"混在了一起**。可试：

* **按原子数归一**：目标是 $\Delta E / N$。这会消掉尺寸依赖，
  让模型专注学"形状有多紧张"，而不是"这分子有多大"；
* **砍掉高温尾巴**：只用 `torsion` + `md300K`，目标范围从 0–1200 收到 0–150，
  看高能尾巴是不是在毒化整个回归；
* **分开建模**：热形变项和构象差项各一个头；
* **排序损失**：用 pairwise ranking loss 直接优化组内 Spearman，
  因为那才是打分函数的实际用途（§9.4）。

**⑤ 模型与表示**

* 换成 **DimeNet++ / MACE / NequIP**，引入三体或等变特征；
* 加大 cutoff 或用多尺度（局部 GNN + 长程静电项）；
* 用**等变**表示而不是只给距离，对力的预测通常有显著帮助；
* **力监督** $\mathcal{L}=\lambda_E\mathcal{L}_E+\lambda_F\mathcal{L}_F$，
  这是 MLIP 的标准做法，对局部形变极其敏感。

**⑥ 工程**

* 用 `torch.compile`、混合精度、GPU 训练（本项目 20 轮 Schnet 用了 34 分钟 CPU）；
* 把图缓存换成内存映射格式（`np.memmap`）以支持更大数据集；
* 用 Weights & Biases / TensorBoard 记录实验；
* `max_neighbors` / `cutoff` 是最大的性能旋钮（开销约随 cutoff³ 增长，见 `benchmarks/`）。

---

## 13. 术语表与延伸阅读

### 13.1 术语表

| 中文 | English | 含义 |
|---|---|---|
| 构象 | conformation | 不改变化学键、只改变单键旋转得到的分子形状 |
| 势能面 | potential energy surface (PES) | 能量作为原子坐标的函数 $E(\mathbf{R})$ |
| 力场 | force field | 用经验函数近似 $E(\mathbf{R})$ 的模型 |
| 二面角 | dihedral / torsion angle | 四个原子绕中间键的相对扭转角 |
| 可旋转键 | rotatable bond | 能自由旋转的单键 |
| 隐式溶剂 | implicit solvent | 把溶剂当连续介质而不是显式分子 |
| 广义玻恩 | Generalized Born (GB) | 一种隐式溶剂的近似模型 |
| 玻恩半径 | Born radius | 原子上"有效"的溶剂暴露半径 |
| 能量最小化 | energy minimisation | 求解 $\min E(\mathbf{R})$，常用 L-BFGS |
| 分子动力学 | molecular dynamics (MD) | 数值积分运动方程得到构象轨迹 |
| 朗之万动力学 | Langevin dynamics | 带摩擦和随机力的 MD，起恒温器作用 |
| 正则系综 | canonical ensemble (NVT) | 固定粒子数、体积、温度的统计系综 |
| 能量均分 | equipartition | 每个二次自由度平均能量 $\frac12 k_BT$ |
| 消息传递 | message passing | GNN 里邻居间交换信息并聚合的框架 |
| 连续滤波卷积 | continuous-filter convolution | SchNet 的核心层，权重由距离决定 |
| 广延量 | extensive | 与体系大小成正比（能量就是） |
| 径向基函数 | radial basis function (RBF) | 把标量距离展开成平滑向量 |
| 数据泄漏 | data leakage | 训练集里含有测试集的信息，导致虚高指标 |
| 早停 | early stopping | 验证损失不再改善就停止训练 |
| 置换不变 | permutation invariant | 输入重排不改变输出 |

### 13.2 本项目方法在文献中的位置

**机器学习势函数（MLIP）主线**

* Behler & Parrinello (2007) — 高维神经网络势的开创工作，把能量写成原子贡献之和。
* Bartók et al. (2010) — SOAP 描述符，平滑重叠原子位置。
* Smith, Isayev, Roitberg (2017) — **ANI-1**，第一个大规模有机分子 MLIP 数据集
  （约 2×10⁷ 个 DFT 构象）。"用神经网络替代昂贵能量计算"的直接范例。
* Chmiela et al. (2017) — **sGDML**，强调**力**的监督和对称性。
* Unke & Meuwly (2019) — **SchNet 用于反应势能面**。
* Batzner et al. (2022) — **NequIP**，E(3) 等变网络。
* Batatia et al. (2022) — **MACE**，高阶多体消息传递。
* Eastman et al. (2023) — **NNP/MM**，把神经网络势与分子力学结合做药物设计。

**图神经网络主线**

* Schütt et al. (2017) — **SchNet**：连续滤波卷积。本项目采用的架构。
* Klicpera et al. (2020) — **DimeNet**：加入键角（三体）信息。
* Gilmer et al. (2017) — **MPNN**：消息传递的统一框架表述。
* Sanchez-Lengeling et al. (2021) — 图神经网络的综述性教程，适合入门。

**物理与工具**

* Allen & Tildesley, *Computer Simulation of Liquids* — 分子模拟的标准教科书。
* Frenkel & Smit, *Understanding Molecular Simulation* — 统计力学与模拟。
* Leach, *Molecular Modelling: Principles and Applications* — 力场与构象搜索。
* Case et al. — **AMBER** 力场系列论文（本项目的 `amber14`）。
* Nguyen, Roe, Simmerling (2013) — **GBn2** 隐式溶剂模型。
* Onufriev, Bashford, Case (2004) — **OBC** 玻恩半径模型。
* Eastman et al. (2017) — **OpenMM 7**，本项目使用的模拟引擎。
* **PDBFixer** — 结构修复与加氢工具。
* **wwPDB** 文件格式文档 — PDB/mmCIF 格式的权威说明。

**评估方法学**

* Kapoor & Narayanan (2023) — *Leakage and the Reproducibility Crisis in ML-based Science*。
  系统讨论了包括"随机切分泄漏"在内的数据泄漏类型，是 §9.1 的直接依据。

### 13.3 想自己动手改的话，从哪开始

1. **换力场**：改 `configs` 里的 `label.forcefield_files`
   （例如加 `implicit/obc2.xml` 对比 GBn2）。
2. **换采样策略**：改 `ensemble.md_temperatures`，或实现副本交换。
3. **换模型**：在 `models.py::build_model` 加一个分支，
   实现带键角特征的 DimeNet 风格层。
4. **加力监督**：让 OpenMM 同时返回力（`getState(getForces=True)`），
   在 `train.py` 里加一项 $\|\mathbf{F}_{\text{pred}} - \mathbf{F}_{\text{true}}\|^2$。
5. **扩大数据**：把 `DEFAULT_PDB_IDS` 换成几百个 PDB 条目，
   或者写一个从 AlphaFold DB 拉结构的下载器。

祝你玩得开心。这个项目的每一个"为什么"都在上面，
如果哪里读起来像黑魔法，那一定是文档没写清楚——那就去改代码，然后回来改这份文档。

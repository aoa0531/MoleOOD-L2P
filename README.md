# MoleOOD-L2P：基于 Prompt 的分子实例增量学习

本项目以 MoleOOD 为基础，引入 L2P 的 Prompt 池检索机制，用于分子性质预测中的有监督实例增量学习（Instance-Incremental Learning，IIL）。

所有阶段执行同一个二分类任务，训练样本分批到来，测试时不需要任务或阶段身份。


## 1. 方法简介

模型保留当前 MoleOOD 实现中的完整图编码器、子结构编码器和性质预测头，并新增可学习的 Prompt 池与对应 keys。

处理流程：

```text
输入分子
   │
   ├─ 不带 Prompt 的完整图编码器（冻结）
   │       ↓
   │   分子表示 query
   │       ↓
   │   与 Prompt keys 进行余弦相似度匹配
   │       ↓
   │   选择 Top-1 Prompt
   │
   ├─ 原子嵌入 + 所选 Prompt
   │       ↓
   │   完整图编码器（冻结，保留对 Prompt 的梯度）
   │       ↓
   │   带 Prompt 的完整图表示
   │
   └─ BRICS 子结构
           ↓
       子结构编码器（冻结）
           ↓
       子结构均值表示

完整图表示 + 子结构均值表示
           ↓
       性质预测头
           ↓
       二分类预测
```

### Prompt 注入方式

每个 Prompt 长度为 1，是与原子嵌入同维度的可学习向量。

在第一层 GNN 消息传递之前，将所选 Prompt 加到当前分子的每个原子嵌入上：

```text
新的原子嵌入 = 原始原子嵌入 + 当前分子选中的 Prompt
```

同一分子的所有原子共享所选 Prompt，不同分子可以选择不同 Prompt。

本实现借鉴 L2P 的检索机制，但使用加法式图 Prompt，不直接复现原版 Transformer 中的 Prompt token 拼接。

### Prompt 池

- 固定池大小，默认包含 10 个 Prompt。
- 每个 Prompt 配置一个独立的可学习 key。
- Prompt 和 keys 分别采用均匀分布随机初始化。
- 根据分子 query 逐样本选择 Top-1。
- 不根据阶段编号指定 Prompt，不动态扩展 Prompt 池。

## 2. 数据集

使用 MoleOOD 提供的 DrugOOD EC50 scaffold 数据：

[lbap_core_ec50_scaffold_brics.json](https://raw.githubusercontent.com/yangnianzu0515/MoleOOD/master/DrugOOD/data/ec50/lbap_core_ec50_scaffold_brics.json)

主要字段：

| 字段 | 含义 |
|---|---|
| `smiles` | 完整分子的 SMILES |
| `substructure` | BRICS 子结构 |
| `cls_label` | 二分类标签 |
| `reg_label` | 连续数值标签，本实现不用于回归训练 |
| `assay_id` | 实验测定编号 |
| `domain_id` | 数据中的域编号 |

本项目使用 `cls_label` 训练二分类模型。

### IIL 划分

仅将原始 `train` 划分为五个阶段：

| 阶段 | 比例 | 用途 |
|---|---:|---|
| D0 | 40% | 训练初始 MoleOOD |
| D1 | 15% | 第一批新增实例 |
| D2 | 15% | 第二批新增实例 |
| D3 | 15% | 第三批新增实例 |
| D4 | 15% | 第四批新增实例 |

采用固定随机种子，尽量保持各阶段类别比例接近，不按 scaffold 或 assay 分阶段。

相同 SMILES 字符串的记录归入同一阶段。当前没有对不同 SMILES 表达的等价分子进行结构标准化。

原始验证集和测试集保持独立，不参与训练。

## 3. 训练方式

### 初始阶段 D0

沿用 MoleOOD 的两阶段训练：

1. 训练环境推断相关模型。
2. 使用不变性目标训练主预测模型。

初始化只使用 D0，不提前访问后续阶段的训练样本。

### 增量阶段 D1—D4

冻结完整图和子结构编码器，包括 BatchNorm 统计量，并关闭编码器中的 Dropout。

更新以下参数：

- 当前 batch 选中的 Prompt；
- 对应的 keys；
- 性质预测头。

增量阶段损失由两部分组成：

```text
总损失 = 分类交叉熵损失 + λ × key 匹配损失
```

每阶段只使用当前阶段的数据，不回放旧训练样本。当前阶段可以训练多个 epoch。

测试时根据分子自动检索 Prompt，不需要任务身份，也不更新模型。

## 4. 项目结构

```text
MoleOOD-L2P/
├── models/
│   ├── Framework.py          # 整体模型及检索、预测数据流
│   ├── mygin.py              # GNN 与加法式 Prompt 注入
│   ├── prompt_pool.py        # Prompt 池、keys 和检索
│   └── dataset.py            # 分子、子结构及阶段数据加载
├── configs/
│   └── iil_scaffold.json     # IIL 实验配置
├── splits/
│   └── ec50_scaffold_iil_seed42.json
├── tests/
│   └── test_iil.py           # 功能和训练流程测试
├── prepare_iil_split.py      # 生成阶段划分
├── train_iil.py              # IIL 训练入口
├── evaluate.py              # 模型评估
├── requirements-iil.txt     # 本地验证使用的依赖版本
├── README_IIL.md            # 详细实现说明
└── main.py                  # 保留的原始训练入口
```

## 5. 环境准备

主要依赖：

- PyTorch
- PyTorch Geometric
- RDKit
- NumPy
- scikit-learn

在已有 PyTorch 的环境中安装：

```bash
pip install torch-geometric==2.6.1 rdkit numpy scikit-learn
```

本地验证版本见 `requirements-iil.txt`。

在 Colab 中建议保留运行时自带的 GPU 版 PyTorch，补充其他依赖。当前 IIL 训练入口不依赖旧版 DrugOOD 或 MMCV。

## 6. 准备数据

将数据文件放到：

```text
data/ec50/lbap_core_ec50_scaffold_brics.json
```

也可以使用 Python 下载：

```python
from pathlib import Path
from urllib.request import urlretrieve

path = Path("data/ec50/lbap_core_ec50_scaffold_brics.json")
path.parent.mkdir(parents=True, exist_ok=True)

urlretrieve(
    "https://raw.githubusercontent.com/yangnianzu0515/MoleOOD/"
    "master/DrugOOD/data/ec50/lbap_core_ec50_scaffold_brics.json",
    path,
)
```

仓库已提供默认阶段划分。训练时会核对数据文件的 SHA256。

如需生成新的划分：

```bash
python prepare_iil_split.py \
    --data data/ec50/lbap_core_ec50_scaffold_brics.json \
    --output splits/new_split.json
```

使用新划分时，需要同步修改配置中的 `split_path`。

## 7. 运行训练

在项目根目录执行：

```bash
python train_iil.py --config configs/iil_scaffold.json
```

默认使用 CPU。使用 GPU 时，将配置中的设备设置为：

```json
"device": "cuda:0"
```

主要配置包括：

| 参数 | 含义 |
|---|---|
| `data_path` | 数据文件路径 |
| `split_path` | 阶段划分文件 |
| `output_dir` | 检查点和指标保存目录 |
| `device` | CPU 或 CUDA 设备 |
| `emb_dim` | 隐藏表示维度 |
| `pool_size` | Prompt 池大小 |
| `batch_size` | 批大小 |
| `initial_epochs` | D0 主模型训练轮数 |
| `incremental_epochs` | 每个增量阶段训练轮数 |
| `key_loss_weight` | key 匹配损失系数 |

在 Colab 中，建议将 `output_dir` 设置为已挂载的 Google Drive 路径，以便保存训练结果。

## 8. 断点续训与评估

从第二个增量阶段结束的检查点继续：

```bash
python train_iil.py \
    --config configs/iil_scaffold.json \
    --resume log/iil_scaffold_seed42/stage_2.pt
```

恢复训练时，配置和阶段划分必须与原运行一致。

当前支持从已完成阶段恢复，不支持从 epoch 中途恢复。

评估最终模型：

```bash
python evaluate.py \
    --iil-checkpoint log/iil_scaffold_seed42/stage_4.pt \
    --splits iid_test ood_test \
    --device cuda:0
```

仅加载来源可信的检查点。

## 9. 输出文件

| 文件 | 内容 |
|---|---|
| `initial_unprompted.pt` | D0 训练后的无 Prompt 基线 |
| `stage_0.pt` | 初始模型加随机初始化的 Prompt 池 |
| `stage_N.pt` | 第 N 个增量阶段结束后的检查点 |
| `metrics.json` | 阶段评估指标和 Prompt 使用次数 |

评估指标包括：

- ROC-AUC；
- 平均精确率 AP，字段名为 `pr_auc_ap`；
- 平衡准确率；
- 准确率；
- 各 Prompt 的使用次数。

OOD 结果用于补充评估未见骨架上的泛化能力。固定测试集上的性能变化不能单独作为遗忘程度的度量。


## 10. 与原始方法的关系

本项目是在本地 MoleOOD 改版基础上的研究性扩展：

- 保留完整图与子结构均值融合；
- 使用 L2P 式 query–key 检索；
- 将 Prompt 注入方式改为原子嵌入相加；
- 面向有监督实例增量学习；
- 不属于原版 L2P 或 MoleOOD 论文的严格复现。

参考项目：

- [MoleOOD](https://github.com/yangnianzu0515/MoleOOD)
- [L2P](https://github.com/google-research/l2p)

使用或发表基于本项目的研究时，请引用对应原始工作。

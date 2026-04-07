# NanoSSL

**NanoSSL：面向固态纳米孔蛋白变体识别的注意力自监督学习框架**

NanoSSL 是一个面向纳米孔蛋白信号分析的自监督学习框架，目标是在标注数据有限、信号噪声较强的条件下，从单分子电流轨迹中学习稳定表示，并提升下游蛋白识别效果。

## 论文信息

本仓库对应如下论文：

> **NanoSSL: attention mechanism-based self-supervised learning method for protein variant identification using solid-state nanopores**  
> *Bioinformatics*, Volume 42, Issue 1, January 2026

- 论文页面：https://academic.oup.com/bioinformatics/article/42/1/btaf657/8371887
- DOI：https://doi.org/10.1093/bioinformatics/btaf657
- 代码仓库：https://github.com/YongXie66/NanoSSL

## 为什么是 NanoSSL？

纳米孔测得的蛋白信号通常噪声大、波动强，而且高质量标注成本高。NanoSSL 通过“两阶段训练”来缓解这个问题：先做自监督预训练，再做监督微调。

### 主要亮点

- 面向少标注场景的自监督预训练
- 使用 Transformer 注意力机制进行序列建模
- 专门针对纳米孔蛋白电流信号设计，包括固态纳米孔实验数据
- 在公开 ONT 数据和淀粉样蛋白 beta 变体识别任务上进行了验证

## 结果概览

根据论文结果，NanoSSL 在公开纳米孔基准任务上取得了较强表现：

- 条形码识别准确率 `0.951 +- 0.001`
- 二分类蛋白识别准确率 `0.982 +- 0.001`

论文还表明，NanoSSL 在固态纳米孔淀粉样蛋白 beta 变体识别任务上取得了领先结果，尤其在 `E22G` 和 `G37R` 等困难变体区分任务中表现突出。

## 方法概述

NanoSSL 采用两阶段训练策略：

### 1. 自监督预训练

将每条纳米孔信号切分为若干不重叠的局部片段，对其中一部分片段进行掩码，模型根据可见上下文学习恢复被掩码片段的表示。

### 2. 监督微调

完成预训练后，再将编码器用于下游分类任务微调，从而在有限标注数据条件下获得更好的识别效果。

### 代码中的核心思想

- 通过 `wave_length` 进行片段化 token 划分
- 使用 Transformer 编码器学习上下文表示
- 通过 `mask_ratio` 控制掩码式自监督目标
- 使用动量式目标编码器提升表示学习稳定性
- 使用分类头完成下游预测

## 仓库结构

```text
NanoSSL/
├── dataset/
│   └── ont/                  # 示例 ONT 数据集（.pt 文件）
├── model/
│   ├── NANOSSL.py            # NanoSSL 主模型
│   └── layers.py             # Transformer 及相关层
├── utils/
│   ├── args.py               # 参数解析与数据分发
│   ├── dataset.py            # 数据集封装
│   ├── datautils.py          # 数据加载工具
│   └── util.py               # 训练与评估辅助函数
├── train.py                  # 主训练入口
├── trainer.py                # 预训练与微调流程
├── train.sh                  # 示例训练脚本
├── requirements.txt
└── README.md
```

## 安装

克隆仓库：

```bash
git clone https://github.com/YongXie66/NanoSSL.git
cd NanoSSL
```

创建环境并安装依赖：

```bash
conda create -n nanossl python=3.10
conda activate nanossl
pip install -r requirements.txt
```

## 快速开始

### 运行仓库自带的 ONT 示例数据

```bash
python train.py \
  --dataset ont \
  --data_path dataset/ont/ \
  --save_path exp/ONT/test \
  --device cuda:0
```

或者直接使用仓库提供的脚本：

```bash
bash train.sh
```

### 在淀粉样蛋白 beta 变体数据上运行

```bash
python train.py \
  --dataset amyloid \
  --data_path data/amyloid/ \
  --mutation E22G \
  --save_path exp/amyloid/E22G \
  --device cuda:0
```

## 数据说明

### 仓库已包含数据

仓库内包含一个 ONT 示例数据集：

```text
dataset/ont/train.pt
dataset/ont/val.pt
dataset/ont/test.pt
```

### 淀粉样蛋白数据

对于淀粉样蛋白 beta 实验，代码默认读取如下命名格式的 MATLAB 文件：

```text
data/amyloid/
├── Native_BetaAmyloid.mat
├── E22G_BetaAmyloid.mat
├── G37R_BetaAmyloid.mat
└── ...
```

## 训练说明

- `train.py` 会执行完整流程：先预训练，再微调。
- 实际数据加载方式由 `--data_path` 决定。
- 主要输出会保存在 `--save_path` 下，包括配置、模型权重和结果摘要。
- 默认参数更适合 GPU 环境。

## 引用

如果这个仓库对你的研究有帮助，欢迎引用论文：

```bibtex
@article{xie2026nanossl,
  author = {Xie, Yong and Li, Jindong and Zhang, Ziyan and Meng, Bin and Dai, Shuaijian and Zhou, Yuchen and Kennedy, Eamonn and Jiao, Niandong and Chen, Haobin and Dong, Zhuxin},
  title = {NanoSSL: attention mechanism-based self-supervised learning method for protein variant identification using solid-state nanopores},
  journal = {Bioinformatics},
  volume = {42},
  number = {1},
  pages = {btaf657},
  year = {2026},
  doi = {10.1093/bioinformatics/btaf657}
}
```

## 致谢

如果 NanoSSL 对你的工作有帮助，欢迎给仓库点个 Star，并引用我们的论文。

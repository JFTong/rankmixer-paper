# RankMixer 论文复现

[![arXiv](https://img.shields.io/badge/arXiv-2507.15551-b31b1b.svg)](https://arxiv.org/abs/2507.15551)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org)

RankMixer 论文的完整算法实现，包含从论文原理到代码的全链路讲解。

## 论文概要

**RankMixer: Scaling Up Ranking Models in Industrial Recommenders**

字节跳动提出了 RankMixer——一种硬件友好的特征交互架构，用于工业级推荐和广告排序系统。

### 核心创新

| 模块 | 论文方案 | 效果 |
|------|---------|------|
| Token Mixing | 零参数跨 token 交互，替换 Self-Attention | MFU 4.5%→45% |
| Per-Token FFN | 每个特征 token 独立 MLP 参数 | 70× 参数，延迟不变 |
| Sparse MoE | ReLU Routing + DTSI-MoE | 1B 参数全量部署 |

### 关键结果
- **参数量**: 16M → 1B (70× 扩展)，推理延迟维持 14.3ms
- **实际收益**: 用户活跃天数 +0.3%，使用时长 +1.08%，广告 ADVV +3.9%

## 本实现

`rankmixer_implementation.ipynb` 包含：

1. ✅ **完整 RankMixer 架构**: Token Mixing + PFFN + Sparse MoE
2. ✅ **双数据源**: 合成数据 + 真实 Criteo 数据（一键切换）
3. ✅ **端到端训练**: 50K+ 训练样本，含验证和测试
4. ✅ **消融实验**: Token Mixing vs Self-Attention vs None
5. ✅ **MoE 扩展**: ReLU Routing 自适应专家激活

## 快速开始

```bash
# 安装依赖
pip install torch pandas numpy scikit-learn matplotlib tqdm nbformat jupyter

# 如需真实 Criteo 数据，额外安装:
pip install kagglehub  # Kaggle 数据源 (推荐)
# 或
pip install datasets   # HuggingFace Criteo 1TB (流式)

# 启动 Jupyter
jupyter notebook rankmixer_implementation.ipynb
```

所有单元格按顺序执行即可。默认使用合成数据，**无需额外下载**。

## 使用真实 Criteo 数据

修改 notebook Cell 6 中的一个开关即可切换：

```python
USE_REAL_DATA = True           # 改为 True 使用真实 Criteo 数据
REAL_DATA_SOURCE = 'kaggle'    # 'kaggle' | 'local' | 'hf' | 'tianchi'
REAL_DATA_ROWS = 5_000_000     # 加载行数
```

### 四种数据源

| 数据源 | 大小 | 速度 | 用途 |
|--------|------|------|------|
| **Kaggle** (`'kaggle'`) | 11GB, 4500万行 | ★★★ | **推荐**。一键下载，首次运行自动获取 |
| **HuggingFace** (`'hf'`) | 276GB, 24天数据 | ★★ | 流式读取，不占磁盘。适合大规模实验 |
| **本地文件** (`'local'`) | 任意 | ★★★★★ | 已下载的 `.txt` 文件 |
| **天池镜像** (`'tianchi'`) | 11GB | ★★★ | 国内下载更快 |

### 数据格式

Criteo 数据为 tab 分隔的文本文件：
```
<label>\t<I1>\t...\t<I13>\t<C1>\t...\t<C26>
```
- 13 个数值特征 (I1-I13)，含缺失值（空字段）
- 26 个类别特征 (C1-C26)，32-bit hash 值
- 缺失值为空字段（不是 NaN）

数据加载器 `criteo_loader.py` 自动处理格式转换，输出与合成数据完全一致的 DataFrame，后续预处理和模型训练**零改动**。

### 单独使用数据加载器

```python
from criteo_loader import load_criteo, download_and_cache

# 方式 1: 一键下载 + 缓存
df = download_and_cache(n_rows=5_000_000, source='kaggle')

# 方式 2: 直接加载
df = load_criteo('kaggle', n_rows=1_000_000)

# 方式 3: 本地文件
df = load_criteo('local', path='./data/train.txt')
```

## Notebook 结构

| # | 内容 | 论文章节 |
|---|------|---------|
| 1-2 | 论文概览 & 架构图解 | §1 |
| 3-4 | 环境配置 & 合成数据生成 | - |
| 5-6 | **📥 Criteo 真实数据加载** 🆕 | - |
| 7-8 | 数据预处理 | - |
| 9 | PyTorch Dataset & DataLoader | - |
| 10 | Feature Tokenization | §2.2, Eq.2 |
| 11 | Multi-Head Token Mixing | §2.3 |
| 12 | Per-Token FFN (PFFN) | §2.4, Eq.6-10 |
| 13 | RankMixer Block + 完整模型 | §2.1, Eq.1 |
| 14-16 | 训练 & 结果分析 | - |
| 17 | 消融实验 | - |
| 18 | Sparse MoE 扩展 | §2.5, Eq.11-12 |
| 19 | 总结 | - |

## 消融实验结果

| 变体 | AUC | 参数量 |
|------|-----|--------|
| **Token Mixing (原文)** | 最高 | 中等 |
| Self-Attention 替代 | 次之 | 更多 |
| 无 Token Mixing | 最低 | 最少 |

> 验证了论文核心主张：在异质特征空间中，Token Mixing 比 Self-Attention 更有效。

## 引用

```bibtex
@article{zhu2025rankmixer,
  title={RankMixer: Scaling Up Ranking Models in Industrial Recommenders},
  author={Zhu, Jie and Fan, Zhifang and Zhu, Xiaoxie and Jiang, Yuchen and
          Wang, Hangyu and Han, Xintian and Ding, Haoran and Wang, Xinmin and
          Zhao, Wenlin and Gong, Zhen and others},
  journal={arXiv preprint arXiv:2507.15551},
  year={2025}
}
```

## 许可

MIT License

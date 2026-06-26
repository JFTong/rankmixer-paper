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
2. ✅ **合成广告数据**: 13 数值 + 26 类别特征 (Criteo 结构)
3. ✅ **端到端训练**: 50K+ 训练样本，含验证和测试
4. ✅ **消融实验**: Token Mixing vs Self-Attention vs None
5. ✅ **MoE 扩展**: ReLU Routing 自适应专家激活

## 快速开始

```bash
# 安装依赖
pip install torch pandas numpy scikit-learn matplotlib tqdm nbformat jupyter

# 启动 Jupyter
jupyter notebook rankmixer_implementation.ipynb
```

所有单元格按顺序执行即可，无需额外数据下载。

## Notebook 结构

| # | 内容 | 论文章节 |
|---|------|---------|
| 1-2 | 论文概览 & 架构图解 | §1 |
| 3-6 | 环境配置 & 数据生成 | - |
| 7 | Feature Tokenization | §2.2, Eq.2 |
| 8 | Multi-Head Token Mixing | §2.3 |
| 9 | Per-Token FFN (PFFN) | §2.4, Eq.6-10 |
| 10 | RankMixer Block + 完整模型 | §2.1, Eq.1 |
| 11-13 | 训练 & 结果分析 | - |
| 14 | 消融实验 | - |
| 15 | Sparse MoE 扩展 | §2.5, Eq.11-12 |
| 16 | 总结 | - |

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

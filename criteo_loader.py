"""
Criteo 真实数据加载器
=====================
支持三种数据源，统一输出 (DataFrame, cat_dims) 供预处理 pipeline 使用。

数据格式: label + I1-I13 (数值) + C1-C26 (类别, 32-bit hash hex)

用法:
    from criteo_loader import load_criteo
    
    # Kaggle (推荐, 11GB)
    df, cat_dims = load_criteo(source='kaggle', n_rows=5_000_000)
    
    # 本地文件
    df, cat_dims = load_criteo(source='local', path='./data/train.txt')
    
    # HuggingFace Criteo 1TB (流式, 不占磁盘)
    df, cat_dims = load_criteo(source='hf', n_rows=1_000_000)
    
    # 天池镜像 (需先手动下载)
    df, cat_dims = load_criteo(source='tianchi', path='./data/train.txt')
"""

import numpy as np
import pandas as pd
import os
import glob

# Criteo 列名（与 notebook 预处理一致）
NUM_COLS = [f'I{i}' for i in range(1, 14)]   # 13 个数值特征
CAT_COLS = [f'C{i}' for i in range(1, 27)]   # 26 个类别特征
ALL_COLS = ['label'] + NUM_COLS + CAT_COLS    # 共 40 列


def _parse_criteo_file(filepath, n_rows=None):
    """解析 Criteo tab 分隔文件
    
    Criteo 原始格式:
        <label>\t<int1>\t...\t<int13>\t<cat1>\t...\t<cat26>
    缺失值为空字段。
    
    Args:
        filepath: train.txt 路径
        n_rows: 读取行数，None=全部
    
    Returns:
        df: 列名为 I1-I13, C1-C26, label 的 DataFrame
    """
    print(f"读取文件: {filepath}")
    
    df = pd.read_csv(
        filepath,
        sep='\t',
        header=None,
        names=ALL_COLS,
        nrows=n_rows,
        na_values=[''],         # 空字段 → NaN
        low_memory=False,
    )
    
    # 数值列类型转换（缺失值保留 NaN）
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # 类别列统一为字符串（缺失值填充为特殊标记）
    for c in CAT_COLS:
        df[c] = df[c].fillna('__MISSING__').astype(str)
    
    print(f"  ✓ 加载完成: {len(df):,} 行")
    print(f"  CTR: {df['label'].mean():.4f}")
    print(f"  数值缺失率: {df[NUM_COLS].isnull().mean().mean():.2%}")
    
    return df


def load_criteo_kaggle(n_rows=None, cache_dir=None):
    """从 Kaggle 下载 Criteo Display Advertising Challenge 数据集
    
    数据集: ~11GB, ~4500万行
    Kaggle: https://www.kaggle.com/c/criteo-display-ad-challenge
    
    首次运行会自动下载 (需接受 Kaggle 条款)。
    
    Args:
        n_rows: 读取行数，None=全部
        cache_dir: kagglehub 缓存目录
    
    Returns:
        df: DataFrame
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError("请先安装: pip install kagglehub")
    
    print("从 Kaggle 下载 Criteo 数据集...")
    path = kagglehub.dataset_download(
        "criteo/criteo-display-ad-challenge",
        path=cache_dir,
    )
    
    # 查找 train.txt（可能在子目录）
    matches = glob.glob(f"{path}/**/train.txt", recursive=True)
    if not matches:
        raise FileNotFoundError(f"未找到 train.txt，下载路径: {path}")
    
    return _parse_criteo_file(matches[0], n_rows=n_rows)


def load_criteo_local(filepath, n_rows=None):
    """从本地文件加载 Criteo 数据
    
    Args:
        filepath: tab 分隔文件路径
        n_rows: 读取行数
    
    Returns:
        df: DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    return _parse_criteo_file(filepath, n_rows=n_rows)


def load_criteo_hf(n_rows=None, day_files=None):
    """从 HuggingFace 加载 Criteo 1TB 数据集 (流式, 不占磁盘)
    
    数据集: ~276GB, 24天, 每天一个文件
    HuggingFace: https://huggingface.co/datasets/criteo/CriteoClickLogs
    
    HF 版本列名与 Kaggle 不同，需要重命名:
        label → label
        integer_feature_1..13 → I1..I13
        categorical_feature_1..26 → C1..C26
    
    Args:
        n_rows: 读取行数（在整个 dataset 上 shuffle 后采样）
        day_files: 要加载的天数文件列表，None=全部24天
    
    Returns:
        df: DataFrame
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请先安装: pip install datasets")
    
    print("从 HuggingFace 加载 Criteo 1TB 数据集 (流式)...")
    
    ds = load_dataset(
        "criteo/CriteoClickLogs",
        split="train",
        streaming=True,
    )
    
    if day_files:
        ds = ds.filter(lambda x, files=day_files: x['split'] in files)
    
    if n_rows:
        print(f"  流式读取 {n_rows:,} 行...")
        ds = ds.take(n_rows)
    else:
        print("  读取全部 24 天数据 (~4B 行)，可能需要数小时...")
    
    # 转为列表后构建 DataFrame
    rows = []
    col_map = {f'integer_feature_{i}': f'I{i}' for i in range(1, 14)}
    col_map.update({f'categorical_feature_{i}': f'C{i}' for i in range(1, 27)})
    
    for i, row in enumerate(ds):
        record = {'label': row['label']}
        for old, new in col_map.items():
            record[new] = row.get(old)
        rows.append(record)
        
        if (i + 1) % 100_000 == 0:
            print(f"  已读取 {i+1:,} 行...")
    
    df = pd.DataFrame(rows)
    
    # 数值列转换
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # 类别列填充
    for c in CAT_COLS:
        df[c] = df[c].fillna('__MISSING__').astype(str)
    
    print(f"  ✓ 加载完成: {len(df):,} 行")
    print(f"  CTR: {df['label'].mean():.4f}")
    
    return df


def load_criteo(source='kaggle', n_rows=None, path=None, **kwargs):
    """统一入口: 加载 Criteo 真实数据
    
    支持四种数据源，返回统一格式的 DataFrame。
    
    Args:
        source: 'kaggle' | 'local' | 'hf' | 'tianchi'
        n_rows: 读取行数限制。推荐: 100万 (快速实验) / 500万 (充分训练)
        path: 本地文件路径 (source='local'/'tianchi' 时必填)
        **kwargs: 传递给底层加载器
    
    Returns:
        df: 包含 label, I1-I13, C1-C26 的 DataFrame
    
    Examples:
        >>> df = load_criteo(source='kaggle', n_rows=5_000_000)
        >>> df = load_criteo(source='local', path='./criteo_data/train.txt')
        >>> df = load_criteo(source='hf', n_rows=1_000_000)
    """
    loaders = {
        'kaggle': lambda: load_criteo_kaggle(n_rows=n_rows, **kwargs),
        'local':  lambda: load_criteo_local(path, n_rows=n_rows),
        'hf':     lambda: load_criteo_hf(n_rows=n_rows, **kwargs),
        'tianchi': lambda: load_criteo_local(path, n_rows=n_rows),  # 天池格式同 Kaggle
    }
    
    if source not in loaders:
        raise ValueError(f"不支持的数据源: {source}，可选: {list(loaders.keys())}")
    
    if source in ('local', 'tianchi') and path is None:
        raise ValueError(f"source='{source}' 必须指定 path 参数")
    
    return loaders[source]()


# ============================================================
# 便捷函数: 下载 + 缓存到本地
# ============================================================

def download_and_cache(output_dir='./criteo_data', n_rows=None, source='kaggle'):
    """下载 Criteo 数据并缓存为本地 parquet 文件（后续加载更快）
    
    Args:
        output_dir: 输出目录
        n_rows: 下载行数
        source: 数据源
    
    Returns:
        df: DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, 'criteo_train.parquet')
    
    if os.path.exists(cache_path):
        print(f"从缓存加载: {cache_path}")
        return pd.read_parquet(cache_path)
    
    df = load_criteo(source=source, n_rows=n_rows)
    
    print(f"缓存到: {cache_path}")
    df.to_parquet(cache_path, index=False)
    
    # 同时保存统计信息
    stats_path = os.path.join(output_dir, 'criteo_stats.txt')
    with open(stats_path, 'w') as f:
        f.write(f"行数: {len(df):,}\n")
        f.write(f"CTR: {df['label'].mean():.4f}\n")
        f.write(f"数值缺失率: {df[NUM_COLS].isnull().mean().mean():.2%}\n")
        f.write(f"类别缺失率: {(df[CAT_COLS]=='__MISSING__').mean().mean():.2%}\n")
    
    return df


# ============================================================
# 数据源指南
# ============================================================

if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════╗
║          Criteo 真实数据加载器 — 使用指南                ║
╠══════════════════════════════════════════════════════════╣
║                                                        ║
║  数据源        大小      速度    推荐用途               ║
║  ─────────     ────      ────    ──────────             ║
║  kaggle        11GB      ★★★    最推荐，国内需代理      ║
║  hf           276GB      ★★     按需流式，不占磁盘      ║
║  local        任意       ★★★★★  已下载的本地文件        ║
║  tianchi      11GB      ★★★    天池镜像，国内快         ║
║                                                        ║
║  快速开始:                                             ║
║    from criteo_loader import load_criteo                ║
║    df = load_criteo('kaggle', n_rows=1_000_000)         ║
║                                                        ║
║  数据格式 (tab 分隔):                                   ║
║    label + I1..I13 (数值) + C1..C26 (类别 hash)        ║
║    缺失值为空字段                                       ║
║                                                        ║
╚══════════════════════════════════════════════════════════╝
    """)

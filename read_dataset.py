"""
遍历 UniMER-Test 下所有子数据集并输出配对数组（不做预测）

数据结构约定（按当前目录结构自动识别）：
- 子数据集目录位于：<DATASET_ROOT>/<name>/
- 对应文本文件位于：<DATASET_ROOT>/<name>.txt（每行一条文本）

输出格式（聚合所有子数据集）：
- [{"dataset": "cpe", "img_path": "...", "text": "...", "pred": ""}, ...]

使用：
- `python read_dataset.py`
- 指定数据集目录：`TEXO_UNIMER_DIR=./data/dataset/hf_datasets/UniMER-Test python read_dataset.py`
"""

import os
import sys
import json
from typing import List, Dict, Tuple

# 注入本地 src，用于导入 texo 本地包
PROJECT_ROOT = os.path.dirname(__file__)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 当前脚本不依赖模型，仅遍历并输出配对数组


def resolve_dataset_dir() -> str:
    """解析数据集目录：优先读取环境变量 TEXO_UNIMER_DIR，其次回退到默认路径。"""
    return os.environ.get(
        "TEXO_UNIMER_DIR",
        os.path.join(PROJECT_ROOT, "data/dataset/hf_datasets/UniMER-Test"),
    )


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def list_images(folder: str) -> List[str]:
    """列出指定目录下的所有图片路径，按文件名排序。

    支持扩展名：png/jpg/jpeg/webp。
    """
    if not os.path.isdir(folder):
        return []
    names = [n for n in os.listdir(folder) if os.path.splitext(n)[1].lower() in IMAGE_EXTS]
    names.sort()
    return [os.path.join(folder, n) for n in names]


def read_texts(txt_path: str) -> List[str]:
    """读取文本文件，每行作为一条文本，去除尾部换行。"""
    if not os.path.isfile(txt_path):
        return []
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f.readlines()]
    return lines


def find_dataset_groups(root: str) -> List[Tuple[str, str, str]]:
    """枚举所有子数据集组。

    规则：在 root 下查找所有一级子目录 name，同时存在同级文件 name.txt。
    返回三元组列表 (name, folder_path, txt_path)。
    """
    groups: List[Tuple[str, str, str]] = []
    if not os.path.isdir(root):
        return groups
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        txt = os.path.join(root, f"{name}.txt")
        if os.path.isdir(folder) and os.path.isfile(txt):
            groups.append((name, folder, txt))
    return groups

def build_pairs_for_group(name: str, folder: str, txt_path: str) -> List[Dict[str, str]]:
    """构建单个数据集组的配对数组。

    按顺序配对 <folder> 下的图片与 <txt_path> 中的每行文本；数量不一致时按最小长度配对。
    返回元素包含：dataset 名称、图片路径、文本与占位的 pred。"""
    images = list_images(folder)
    texts = read_texts(txt_path)

    n_img = len(images)
    n_txt = len(texts)
    if n_img == 0:
        print(f"[ERROR] 未在目录中找到图片：{folder}")
        return []
    if n_txt == 0:
        print(f"[ERROR] 未找到或为空：{txt_path}")
        return []
    if n_img != n_txt:
        print(f"[WARN] [{name}] 图像与文本数量不一致：images={n_img}, texts={n_txt}，将按最小长度配对。")

    k = min(n_img, n_txt)
    return [
        {"dataset": name, "img_path": images[i], "text": texts[i]}
        for i in range(k)
    ]


def build_all_pairs(root: str) -> List[Dict[str, str]]:
    """聚合所有子数据集的配对数组。"""
    pairs: List[Dict[str, str]] = []
    groups = find_dataset_groups(root)
    if not groups:
        print(f"[ERROR] 未在目录中找到任何子数据集组：{root}")
        return pairs
    for name, folder, txt in groups:
        pairs.extend(build_pairs_for_group(name, folder, txt))
    return pairs


def read_dataset() -> List[Dict[str, str]]:
    """主流程：解析路径、生成聚合配对数组并以 JSON 输出。"""
    dataset_root = resolve_dataset_dir()
    pairs = build_all_pairs(dataset_root)
    # print(json.dumps(pairs, ensure_ascii=False))
    return pairs

# if __name__ == "__main__":
#     read_dataset()
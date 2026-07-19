#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第31课：多模态 Embedding
========================

本课程讲解文本之外的 Embedding——图像、音频、多模态（图文联合）。
Embedding 不只是文本的专利，CLIP 等模型让"用文字搜图片"成为可能。

学习目标：
1. 理解多模态 Embedding 的核心思想（共享向量空间）
2. 掌握 CLIP 模型的原理和使用
3. 了解图像检索、文搜图、图搜图的应用场景
4. 能用 CLIP 构建简单的多模态检索系统

"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# Part 1: 什么是多模态 Embedding？
# ============================================================

def demo_what_is_multimodal():
    """
    多模态 Embedding 的核心思想：把不同类型的输入（文本、图像、音频）
    映射到同一个向量空间，使得语义相近的内容距离近。
    """
    print("=" * 60)
    print("Part 1: 什么是多模态 Embedding？")
    print("=" * 60)

    print("""
    传统 Embedding（单模态）：

    文本 ──→ [0.1, 0.3, 0.8, ...]  文本向量空间
    图片 ──→ 无法处理 ❌

    多模态 Embedding：

    文本 "一只猫"  ──→ [0.1, 0.3, 0.8, ...]  ┐
                                              ├──→ 同一个向量空间！
    图片（猫的照片）──→ [0.2, 0.4, 0.7, ...]  ┘

    因为在同一个空间，所以可以：
    ├── 用文字搜图片："猫" → 找到猫的照片
    ├── 用图片搜文字：猫的照片 → 找到"猫"、"宠物"、"猫咪"等文字
    ├── 图片搜图片：猫的照片 → 找到其他猫的照片
    └── 跨模态聚类：把猫的文字和图片聚在一起
    """)

    print("""
    为什么这很重要？

    传统搜索：用关键词匹配图片文件名/标签
    → 只能搜到标签里写了"猫"的图片
    → 标签不全就搜不到

    多模态搜索：用语义理解图片内容
    → 图片里有一只猫，即使没有标签，也能被"猫"搜到
    → 甚至能搜到"毛茸茸的小动物"这种语义相关的图片
    """)


# ============================================================
# Part 2: CLIP 模型原理
# ============================================================

def demo_clip_principle():
    """
    CLIP（Contrastive Language-Image Pre-training）是 OpenAI 发布的
    多模态基础模型，也是目前最流行的图文 Embedding 模型。
    """
    print("\n" + "=" * 60)
    print("Part 2: CLIP 模型原理")
    print("=" * 60)

    print("""
    CLIP 的架构：

    ┌─────────────┐              ┌─────────────┐
    │  文本编码器  │              │  图像编码器  │
    │ (Text       │              │ (Vision     │
    │  Encoder)   │              │  Encoder)   │
    │  Transformer│              │  ViT/ResNet │
    └──────┬──────┘              └──────┬──────┘
           │                            │
           ▼                            ▼
    [文本向量 512d]              [图像向量 512d]
           │                            │
           └──────────┬─────────────────┘
                      │
                      ▼
              对比学习：让匹配的图文对
              向量距离近，不匹配的距离远
    """)

    print("""
    CLIP 的训练方式（对比学习）：

    训练数据：互联网上的图文配对（4 亿对）

    正样本：一张猫的图片 + "一只猫坐在沙发上"  → 拉近距离
    负样本：一张猫的图片 + "一辆红色的汽车"    → 推远距离

    训练目标：在一个 batch 中，让每张图片找到它对应的文字，
             让每段文字找到它对应的图片。

    训练完成后：
    - 文本编码器：能把任何文本转为 512 维向量
    - 图像编码器：能把任何图片转为 512 维向量
    - 两个向量在同一空间，可以直接计算相似度
    """)

    print("""
    CLIP 的关键特性：

    1. 零样本分类（Zero-shot Classification）
       不需要训练，直接用文字描述类别就能分类图片
       类别描述："一张狗的照片"、"一张猫的照片"
       图片 → 分别算和每个类别的相似度 → 选最高的

    2. 开放词汇（Open Vocabulary）
       不限于固定的类别，任何文字描述都可以
       "一只戴着墨镜的狗" → 即使训练时没见过这个描述，也能工作

    3. 跨语言
       中文、英文、日文... 都能映射到同一空间
       "一只猫" ≈ "a cat" ≈ "猫の写真"（向量相近）
    """)


# ============================================================
# Part 3: CLIP 实战
# ============================================================

def demo_clip_usage():
    """
    CLIP 实战：安装和基本使用。
    """
    print("\n" + "=" * 60)
    print("Part 3: CLIP 实战")
    print("=" * 60)

    print("""
    安装：
    pip install transformers torch Pillow

    或使用 OpenAI 的原版：
    pip install open-clip-torch
    """)

    # 检查是否可以导入
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False

    if has_torch:
        print("✅ PyTorch 已安装，可以运行 CLIP")
    else:
        print("⚠️ PyTorch 未安装，以下为代码示例")

    print("""
    方式一：使用 Hugging Face Transformers

    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image

    # 加载模型
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 图文编码
    image = Image.open("cat.jpg")
    inputs = processor(
        text=["一只猫", "一只狗", "一辆汽车"],
        images=image,
        return_tensors="pt",
        padding=True,
    )

    outputs = model(**inputs)
    logits = outputs.logits_per_image  # 图文匹配分数
    probs = logits.softmax(dim=1)       # 概率

    print(probs)  # 例如 [0.85, 0.10, 0.05] → 猫的概率最高


    方式二：使用 open-clip-torch（更灵活）

    import open_clip
    import torch
    from PIL import Image

    # 加载模型
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # 图像编码
    image = preprocess(Image.open("cat.jpg")).unsqueeze(0)
    image_features = model.encode_image(image)

    # 文本编码
    text = tokenizer(["一只猫", "一只狗", "一辆汽车"])
    text_features = model.encode_text(text)

    # 计算相似度
    similarity = torch.cosine_similarity(image_features, text_features)
    print(similarity)  # [0.28, 0.15, 0.05]
    """)


# ============================================================
# Part 4: 多模态检索系统
# ============================================================

class MultimodalSearchDemo:
    """
    多模态检索系统演示。

    支持三种检索模式：
    1. 文搜图（Text → Image）
    2. 图搜图（Image → Image）
    3. 图搜文（Image → Text）
    """

    def __init__(self):
        self.image_embeddings = {}  # {image_id: embedding}
        self.text_embeddings = {}   # {text_id: embedding}
        self.image_metadata = {}
        self.text_metadata = {}

    def add_image(self, image_id: str, embedding: np.ndarray, metadata: Dict = None):
        """添加图片向量"""
        self.image_embeddings[image_id] = embedding / np.linalg.norm(embedding)
        self.image_metadata[image_id] = metadata or {}

    def add_text(self, text_id: str, embedding: np.ndarray, metadata: Dict = None):
        """添加文本向量"""
        self.text_embeddings[text_id] = embedding / np.linalg.norm(embedding)
        self.text_metadata[text_id] = metadata or {}

    def search_images_by_text(self, text_embedding: np.ndarray,
                               top_k: int = 5) -> List[Tuple[str, float]]:
        """文搜图：用文本向量搜索最相似的图片"""
        query = text_embedding / np.linalg.norm(text_embedding)
        results = []
        for img_id, img_emb in self.image_embeddings.items():
            sim = float(np.dot(query, img_emb))
            results.append((img_id, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_images_by_image(self, image_embedding: np.ndarray,
                                top_k: int = 5) -> List[Tuple[str, float]]:
        """图搜图：用图片向量搜索最相似的图片"""
        query = image_embedding / np.linalg.norm(image_embedding)
        results = []
        for img_id, img_emb in self.image_embeddings.items():
            sim = float(np.dot(query, img_emb))
            results.append((img_id, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_texts_by_image(self, image_embedding: np.ndarray,
                               top_k: int = 5) -> List[Tuple[str, float]]:
        """图搜文：用图片向量搜索最相似的文本"""
        query = image_embedding / np.linalg.norm(image_embedding)
        results = []
        for txt_id, txt_emb in self.text_embeddings.items():
            sim = float(np.dot(query, txt_emb))
            results.append((txt_id, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


def demo_multimodal_search():
    """演示多模态检索"""
    print("\n" + "=" * 60)
    print("Part 4: 多模态检索系统演示")
    print("=" * 60)

    search = MultimodalSearchDemo()

    # 模拟图片和文本的 Embedding（实际用 CLIP 生成）
    np.random.seed(42)
    dim = 512

    # 模拟：语义相近的内容向量也相近
    # "猫" 的基础向量
    cat_base = np.random.randn(dim)
    cat_base = cat_base / np.linalg.norm(cat_base)

    # "狗" 的基础向量（和猫有一定相似度）
    dog_base = cat_base * 0.5 + np.random.randn(dim) * 0.5
    dog_base = dog_base / np.linalg.norm(dog_base)

    # "汽车" 的基础向量（和猫狗差异大）
    car_base = np.random.randn(dim)
    car_base = car_base / np.linalg.norm(car_base)

    # 添加图片 Embedding
    search.add_image("cat_photo_1", cat_base + np.random.randn(dim) * 0.1,
                     {"description": "一只橘猫"})
    search.add_image("cat_photo_2", cat_base + np.random.randn(dim) * 0.1,
                     {"description": "一只黑猫"})
    search.add_image("dog_photo_1", dog_base + np.random.randn(dim) * 0.1,
                     {"description": "一只金毛犬"})
    search.add_image("dog_photo_2", dog_base + np.random.randn(dim) * 0.1,
                     {"description": "一只柯基"})
    search.add_image("car_photo_1", car_base + np.random.randn(dim) * 0.1,
                     {"description": "一辆红色跑车"})

    # 添加文本 Embedding
    search.add_text("text_cat", cat_base + np.random.randn(dim) * 0.05,
                    {"text": "一只可爱的猫咪"})
    search.add_text("text_dog", dog_base + np.random.randn(dim) * 0.05,
                    {"text": "一只忠诚的狗狗"})
    search.add_text("text_car", car_base + np.random.randn(dim) * 0.05,
                    {"text": "一辆高速行驶的汽车"})
    search.add_text("text_pet", (cat_base + dog_base) / 2 + np.random.randn(dim) * 0.05,
                    {"text": "可爱的宠物们"})

    # 文搜图
    print("\n🔍 文搜图：查询 '一只猫咪'")
    results = search.search_images_by_text(cat_base, top_k=3)
    for img_id, score in results:
        desc = search.image_metadata[img_id]["description"]
        print(f"   [{score:.4f}] {img_id} - {desc}")

    # 图搜图
    print("\n🔍 图搜图：用 'cat_photo_1' 搜相似图片")
    results = search.search_images_by_image(search.image_embeddings["cat_photo_1"], top_k=3)
    for img_id, score in results:
        desc = search.image_metadata[img_id]["description"]
        print(f"   [{score:.4f}] {img_id} - {desc}")

    # 图搜文
    print("\n🔍 图搜文：用 'cat_photo_1' 搜相关文字")
    results = search.search_texts_by_image(search.image_embeddings["cat_photo_1"], top_k=3)
    for txt_id, score in results:
        text = search.text_metadata[txt_id]["text"]
        print(f"   [{score:.4f}] {txt_id} - {text}")


# ============================================================
# Part 5: 主流多模态 Embedding 模型
# ============================================================

def demo_multimodal_models():
    """主流多模态 Embedding 模型对比"""
    print("\n" + "=" * 60)
    print("Part 5: 主流多模态 Embedding 模型")
    print("=" * 60)

    models = [
        {
            "name": "CLIP ViT-B/32",
            "provider": "OpenAI",
            "modalities": "文本 + 图像",
            "dimension": 512,
            "params": "150M",
            "strengths": "经典模型，社区资源丰富，轻量",
            "weaknesses": "中文支持弱，精度不如新模型",
        },
        {
            "name": "CLIP ViT-L/14",
            "provider": "OpenAI",
            "modalities": "文本 + 图像",
            "dimension": 768,
            "params": "428M",
            "strengths": "精度更高，支持 336px 输入",
            "weaknesses": "需要更多显存",
        },
        {
            "name": "SigLIP",
            "provider": "Google",
            "modalities": "文本 + 图像",
            "dimension": "768-1152",
            "params": "400M-878M",
            "strengths": "不需要负样本，训练更稳定，多语言好",
            "weaknesses": "社区资源不如 CLIP",
        },
        {
            "name": "E5-V",
            "provider": "Microsoft",
            "modalities": "文本 + 图像",
            "dimension": 1024,
            "params": "2B",
            "strengths": "图文检索 MTEB 多模态榜单 Top",
            "weaknesses": "模型较大，需要较强 GPU",
        },
        {
            "name": "Jina CLIP v2",
            "provider": "Jina AI",
            "modalities": "文本 + 图像",
            "dimension": 1024,
            "params": "900M",
            "strengths": "中英文支持好，Matryoshka 维度裁剪",
            "weaknesses": "较新，社区较小",
        },
        {
            "name": "Chinese-CLIP",
            "provider": "OFA-Sys/阿里",
            "modalities": "文本 + 图像",
            "dimension": 512-768,
            "params": "188M-406M",
            "strengths": "中文优化，基于 CLIP 微调",
            "weaknesses": "英文能力不如原版 CLIP",
        },
        {
            "name": "BGE-M3",
            "provider": "BAAI（智源）",
            "modalities": "文本（可扩展到多模态）",
            "dimension": 1024,
            "params": "568M",
            "strengths": "文本为主，多语言强",
            "weaknesses": "不是原生多模态模型",
        },
    ]

    for m in models:
        print(f"\n{'─' * 55}")
        print(f"📦 {m['name']}  ({m['provider']})")
        print(f"   模态: {m['modalities']}  |  维度: {m['dimension']}  |  参数: {m['params']}")
        print(f"   ✅ {m['strengths']}")
        print(f"   ❌ {m['weaknesses']}")

    print(f"\n{'─' * 55}")
    print("""
    选型建议：

    图文检索（中文）→ Chinese-CLIP / Jina CLIP v2
    图文检索（英文）→ CLIP ViT-L/14 / SigLIP
    追求极致精度   → E5-V（需要强 GPU）
    轻量/移动端    → CLIP ViT-B/32
    """)


# ============================================================
# Part 6: 应用场景
# ============================================================

def demo_applications():
    """多模态 Embedding 的实际应用场景"""
    print("\n" + "=" * 60)
    print("Part 6: 实际应用场景")
    print("=" * 60)

    scenarios = [
        {
            "name": "电商图搜商品",
            "description": "用户拍一张鞋的照片，找到同款或相似商品",
            "flow": "图片 → CLIP 编码 → 向量检索商品库 → 返回相似商品",
            "key": "需要商品图片库的 Embedding 索引",
        },
        {
            "name": "设计素材搜索",
            "description": "设计师输入'温暖的夕阳风景'，找到合适的素材图",
            "flow": "文本 → CLIP 编码 → 向量检索素材库 → 返回匹配图片",
            "key": "文本描述的多样性（同义词、风格描述）",
        },
        {
            "name": "图片去重/聚类",
            "description": "检测相似图片、自动分组",
            "flow": "图片批量 → CLIP 编码 → 相似度矩阵 → 聚类",
            "key": "阈值设定，区分'相同内容不同角度'和'不同内容'",
        },
        {
            "name": "多模态 RAG",
            "description": "文档中包含图片和文字，统一检索",
            "flow": "文档拆分 → 文字用文本模型、图片用 CLIP → 统一向量库",
            "key": "文字和图片的索引策略、结果融合",
        },
        {
            "name": "内容审核",
            "description": "检测图片是否包含违规内容",
            "flow": "图片 → CLIP 编码 → 和违规描述库比对 → 标记风险",
            "key": "描述库的覆盖面、阈值调优",
        },
    ]

    for s in scenarios:
        print(f"\n🎯 {s['name']}")
        print(f"   场景：{s['description']}")
        print(f"   流程：{s['flow']}")
        print(f"   关键：{s['key']}")


# ============================================================
# Part 7: 从文本 RAG 到多模态 RAG
# ============================================================

def demo_multimodal_rag():
    """
    多模态 RAG：把文本 RAG 扩展到图文混合检索。
    """
    print("\n" + "=" * 60)
    print("Part 7: 从文本 RAG 到多模态 RAG")
    print("=" * 60)

    print("""
    文本 RAG（你已经学过的）：

    文档 → 分块 → 文本 Embedding → 向量库 → 检索 → LLM 回答

    多模态 RAG：

    文档（含图片）──→ 分块 ──┬→ 文本块 → 文本 Embedding ──┐
                            │                              ├→ 统一向量库 → 检索
                            └→ 图片块 → CLIP Embedding ────┘
                                                          │
                                                          ▼
                                                    LLM 回答（引用图片）
    """)

    print("""
    实现多模态 RAG 的三个关键步骤：

    步骤一：图文分离
    ├── PDF/网页中提取文字和图片
    ├── 文字走文本分块流程
    └── 图片走 CLIP 编码，附加周围文字作为描述

    步骤二：统一索引
    ├── 文本块用文本 Embedding（如千问 v4）
    ├── 图片块用 CLIP Embedding
    ├── 两者存入同一个向量库
    └── 元数据标记类型（text/image）

    步骤三：混合检索
    ├── 用户查询用文本 Embedding 编码
    ├── 同时检索文本块和图片块
    ├── 结果融合排序
    └── LLM 生成回答时可以引用图片
    """)

    print("""
    当前的局限：

    1. LLM 看图能力
       ├── 需要多模态 LLM（GPT-4o、Claude 3.5、Qwen-VL）才能理解图片
       └── 普通 LLM 只能读图片的文字描述，不能看图片本身

    2. 图片分块
       ├── 文本有成熟的分块策略，图片没有
       └── 一张图应该作为一个完整块，还是切割？

    3. 跨模态对齐
       ├── 文本 Embedding 和图片 Embedding 的尺度可能不同
       └── 需要归一化或专门的融合策略

    4. 成本
       ├── CLIP 编码图片比文本 Embedding 慢
       └── 图片向量通常比文本向量占更多存储
    """)


# ============================================================
# 主函数
# ============================================================

def main():
    """运行所有演示"""
    demo_what_is_multimodal()
    demo_clip_principle()
    demo_clip_usage()
    demo_multimodal_search()
    demo_multimodal_models()
    demo_applications()
    demo_multimodal_rag()

    print("\n" + "=" * 60)
    print("📝 本课要点总结")
    print("=" * 60)
    print("""
    1. 多模态 Embedding 把文本、图片映射到同一个向量空间
    2. CLIP 是最流行的图文 Embedding 模型（对比学习训练）
    3. 核心能力：文搜图、图搜图、图搜文、零样本分类
    4. 中文场景推荐 Chinese-CLIP 或 Jina CLIP v2
    5. 多模态 RAG = 文本 RAG + 图片 CLIP 编码 + 统一索引
    6. 需要多模态 LLM（GPT-4o 等）才能真正"看懂"图片
    """)


if __name__ == "__main__":
    main()

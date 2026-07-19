#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第28课：Embedding 模型对比与选型
================================

本课程对比主流 Embedding 模型，帮助你根据场景选择最合适的模型。
你项目目前只用了千问 text-embedding-v4，了解全貌后才能做出最佳选型。

学习目标：
1. 了解主流 Embedding 模型及其特点
2. 掌握 MTEB 排行榜的解读方法
3. 能根据场景（语言、成本、维度、延迟）选择模型
4. 理解本地模型 vs API 模型的权衡

"""

import time
import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# Part 1: Embedding 模型全景
# ============================================================

def demo_model_landscape():
    """
    主流 Embedding 模型全景图。

    按提供商分类：OpenAI、Cohere、阿里（千问）、
    开源（BGE、E5、GTE、Jina）、Google、Voyage。
    """
    print("=" * 60)
    print("Part 1: Embedding 模型全景")
    print("=" * 60)

    models = [
        # ---- 商业 API 模型 ----
        {
            "name": "text-embedding-3-large",
            "provider": "OpenAI",
            "dimension": 3072,
            "max_tokens": 8191,
            "languages": "多语言（100+）",
            "mteb_rank": "Top 5",
            "cost": "$0.13/1M tokens",
            "strengths": "综合性能强，支持维度裁剪（Matryoshka）",
            "weaknesses": "价格较高，依赖 OpenAI",
        },
        {
            "name": "text-embedding-3-small",
            "provider": "OpenAI",
            "dimension": 1536,
            "max_tokens": 8191,
            "languages": "多语言",
            "mteb_rank": "Top 15",
            "cost": "$0.02/1M tokens",
            "strengths": "性价比极高，性能够用",
            "weaknesses": "精度不如 large",
        },
        {
            "name": "embed-v4",
            "provider": "Google (Gemini)",
            "dimension": "动态 (256-3072)",
            "max_tokens": 8192,
            "languages": "多语言（100+）",
            "mteb_rank": "Top 3",
            "cost": "$0.15/1M tokens",
            "strengths": "多语言最强，支持代码，支持任务类型指定",
            "weaknesses": "Google Cloud 依赖",
        },
        {
            "name": "embed-v3",
            "provider": "Cohere",
            "dimension": 1024,
            "max_tokens": 512,
            "languages": "多语言（100+）",
            "mteb_rank": "Top 5",
            "cost": "$0.10/1M tokens",
            "strengths": "多语言优秀，内置压缩，搜索和分类两种模式",
            "weaknesses": "max_tokens 较小",
        },
        {
            "name": "voyage-3-large",
            "provider": "Voyage AI",
            "dimension": 1024,
            "max_tokens": 32000,
            "languages": "多语言",
            "mteb_rank": "Top 3",
            "cost": "$0.18/1M tokens",
            "strengths": "代码和法律领域表现突出，上下文窗口大",
            "weaknesses": "知名度较低",
        },
        # ---- 阿里千问 ----
        {
            "name": "text-embedding-v4",
            "provider": "阿里（千问）",
            "dimension": "可选 (64-2048)",
            "max_tokens": 8192,
            "languages": "多语言，中文优化",
            "mteb_rank": "中文 Top 3",
            "cost": "¥0.7/1M tokens",
            "strengths": "中文最强之一，8种维度可选，性价比高",
            "weaknesses": "英文稍逊于 OpenAI/Cohere",
        },
        # ---- 开源模型 ----
        {
            "name": "BGE-M3",
            "provider": "BAAI（智源）",
            "dimension": 1024,
            "max_tokens": 8192,
            "languages": "多语言（100+）",
            "mteb_rank": "开源 Top 3",
            "cost": "免费（自部署）",
            "strengths": "多语言+多粒度+多功能，支持稠密/稀疏/ColBERT 三种检索",
            "weaknesses": "需要 GPU，推理速度依赖硬件",
        },
        {
            "name": "E5-large-v2",
            "provider": "Microsoft",
            "dimension": 1024,
            "max_tokens": 512,
            "languages": "英文为主",
            "mteb_rank": "英文 Top 5",
            "cost": "免费（自部署）",
            "strengths": "英文性能强，轻量",
            "weaknesses": "多语言支持弱，上下文短",
        },
        {
            "name": "GTE-large",
            "provider": "阿里 DAMO",
            "dimension": 1024,
            "max_tokens": 8192,
            "languages": "多语言",
            "mteb_rank": "开源 Top 5",
            "cost": "免费（自部署）",
            "strengths": "中文表现好，上下文长",
            "weaknesses": "社区活跃度不如 BGE",
        },
        {
            "name": "Jina-embeddings-v3",
            "provider": "Jina AI",
            "dimension": "可选 (32-1024)",
            "max_tokens": 8192,
            "languages": "多语言（89）",
            "mteb_rank": "开源 Top 3",
            "cost": "免费（自部署）/ API 付费",
            "strengths": "支持维度裁剪，多语言强，有 API 和本地两种",
            "weaknesses": "大模型需要较强 GPU",
        },
        {
            "name": "nomic-embed-text-v1.5",
            "provider": "Nomic AI",
            "dimension": "可选 (64-768)",
            "max_tokens": 8192,
            "languages": "英文为主",
            "mteb_rank": "开源 Top 10",
            "cost": "免费（自部署）",
            "strengths": "轻量（137M 参数），支持 Matryoshka 维度裁剪",
            "weaknesses": "多语言支持有限",
        },
    ]

    for m in models:
        print(f"\n{'─' * 55}")
        print(f"📦 {m['name']}  ({m['provider']})")
        print(f"   维度: {m['dimension']}  |  上限: {m['max_tokens']} tokens")
        print(f"   语言: {m['languages']}  |  MTEB: {m['mteb_rank']}")
        print(f"   成本: {m['cost']}")
        print(f"   ✅ {m['strengths']}")
        print(f"   ❌ {m['weaknesses']}")


# ============================================================
# Part 2: MTEB 排行榜解读
# ============================================================

def demo_mteb():
    """
    MTEB（Massive Text Embedding Benchmark）是 Embedding 模型的标准评测。

    官网：https://huggingface.co/spaces/mteb/leaderboard

    MTEB 测试 8 类任务：
    """
    print("\n" + "=" * 60)
    print("Part 2: MTEB 排行榜解读")
    print("=" * 60)

    tasks = [
        ("Classification", "文本分类", "用 Embedding 做分类任务的准确率"),
        ("Clustering", "聚类", "Embedding 的聚类纯度"),
        ("Pair Classification", "句对分类", "判断两个句子是否等价/蕴含"),
        ("Reranking", "重排序", "对候选文档重新排序的质量"),
        ("Retrieval", "检索", "⭐ RAG 最关心的指标，检索相关文档的能力"),
        ("STS", "语义文本相似度", "计算句子相似度的准确度"),
        ("Summarization", "摘要", "Embedding 质量对摘要任务的影响"),
        ("Bitext Mining", "双语挖掘", "跨语言句子对齐能力"),
    ]

    print("\nMTEB 评测的 8 类任务：")
    print("─" * 60)
    for task_name, cn_name, description in tasks:
        print(f"  {task_name:<25} ({cn_name})")
        print(f"    → {description}")

    print(f"\n{'─' * 60}")
    print("""
    看排行榜的注意事项：

    1. 看 Retrieval 分数（做 RAG 最关键）
       不是所有 Top 1 模型在检索任务上都最强

    2. 看你的语言
       英文 Top 1 不一定是中文 Top 1
       MTEB 有专门的中文排行榜（C-MTEB）

    3. 看任务类型
       检索、分类、聚类对 Embedding 的要求不同
       检索需要"找到相关文档"，分类需要"区分类别"

    4. 看推理速度和成本
       MTEB 只评质量，不评速度和价格
       生产环境要综合考虑质量、速度、成本

    5. MTEB 分数差距 < 2% 时，实际体验差异不大
       不要为了 0.5% 的提升付出 10 倍成本
    """)


# ============================================================
# Part 3: 选型决策框架
# ============================================================

def demo_selection_framework():
    """
    选型决策框架：根据你的场景选择最合适的模型。
    """
    print("\n" + "=" * 60)
    print("Part 3: 选型决策框架")
    print("=" * 60)

    print("""
    选型要考虑的 5 个维度：

    ┌──────────────────────────────────────────────────┐
    │  1. 语言需求                                      │
    │     纯中文 → 千问 v4 / GTE                        │
    │     纯英文 → OpenAI / E5                          │
    │     中英混合 → 千问 v4 / BGE-M3 / OpenAI          │
    │     小语种 → BGE-M3 / Cohere embed-v3             │
    ├──────────────────────────────────────────────────┤
    │  2. 部署方式                                      │
    │     用 API（省事）→ OpenAI / 千问 / Cohere        │
    │     本地部署（隐私/离线）→ BGE-M3 / GTE / Jina    │
    │     边缘设备（手机/嵌入式）→ nomic-embed / 小模型   │
    ├──────────────────────────────────────────────────┤
    │  3. 预算                                          │
    │     零预算 → 开源自部署（BGE / E5 / GTE）          │
    │     低预算 → 千问 v4（¥0.7/百万token，性价比之王）  │
    │     不差钱 → OpenAI large / Voyage large           │
    ├──────────────────────────────────────────────────┤
    │  4. 数据规模                                      │
    │     < 10 万条 → 任何模型都可以                     │
    │     10-100 万 → 关注推理速度                       │
    │     > 100 万 → 需要维度小的模型（减少存储和计算）   │
    ├──────────────────────────────────────────────────┤
    │  │  5. 领域特殊性                                 │
    │     通用文本 → 通用模型                            │
    │     代码 → Voyage-3-large / CodeBERT              │
    │     法律/医疗 → 领域微调模型                       │
    │     多模态 → CLIP / Voyage multimodal             │
    └──────────────────────────────────────────────────┘
    """)

    # 场景推荐
    scenarios = [
        {
            "scenario": "中文 RAG 系统，预算有限",
            "recommended": "千问 text-embedding-v4",
            "reason": "中文性能 Top 3，¥0.7/百万 token 极便宜，8种维度可选",
        },
        {
            "scenario": "中英混合的企业知识库",
            "recommended": "OpenAI text-embedding-3-small 或 BGE-M3",
            "reason": "多语言均衡，small 性价比高，BGE-M3 免费且多语言强",
        },
        {
            "scenario": "数据不能出内网（金融/医疗/政务）",
            "recommended": "BGE-M3 或 GTE-large（本地部署）",
            "reason": "开源免费，数据不出网，GPU 推理性能可接受",
        },
        {
            "scenario": "英文代码搜索",
            "recommended": "Voyage-3-large",
            "reason": "代码领域专门优化，MTEB 代码任务 Top 1",
        },
        {
            "scenario": "手机/嵌入式设备",
            "recommended": "nomic-embed-text-v1.5",
            "reason": "137M 参数极轻量，支持 Matryoshka 可用 64 维",
        },
        {
            "scenario": "追求最高质量，不计成本",
            "recommended": "OpenAI text-embedding-3-large + Reranker",
            "reason": "综合 MTEB Top 5，配合 Reranker 达到最佳效果",
        },
    ]

    print("\n📊 场景推荐：")
    print("─" * 55)
    for s in scenarios:
        print(f"\n🎯 {s['scenario']}")
        print(f"   推荐：{s['recommended']}")
        print(f"   理由：{s['reason']}")


# ============================================================
# Part 4: API 模型 vs 本地模型
# ============================================================

def demo_api_vs_local():
    """API 模型和本地部署模型的详细对比"""
    print("\n" + "=" * 60)
    print("Part 4: API 模型 vs 本地模型")
    print("=" * 60)

    print(f"\n{'维度':<15} {'API 模型':<30} {'本地模型':<30}")
    print("─" * 75)

    comparisons = [
        ("代表", "OpenAI / 千问 / Cohere", "BGE-M3 / GTE / E5"),
        ("上手难度", "极低，一个 API 调用", "中等，需要配置环境"),
        ("质量", "通常较高（持续更新）", "部分模型质量相当"),
        ("速度", "取决于网络和限流", "取决于 GPU 硬件"),
        ("成本", "按 token 计费，持续支出", "一次性硬件成本"),
        ("数据隐私", "数据发送到第三方", "数据不出本机"),
        ("离线使用", "需要网络", "完全离线可用"),
        ("定制能力", "有限（只能选模型）", "可微调、可裁剪"),
        ("维护成本", "零（提供商维护）", "需要自己更新/监控"),
        ("适合规模", "小到中等", "中等到大规模"),
    ]

    for dim, api, local in comparisons:
        print(f"{dim:<15} {api:<30} {local:<30}")

    print(f"\n{'─' * 75}")
    print("""
    决策建议：

    ┌─────────────────────────────┐
    │  数据敏感吗？                │
    │  ├── 是 → 本地部署           │
    │  └── 否 ↓                   │
    │                              │
    │  日调用量大吗？（> 10 万次/日）│
    │  ├── 是 → 本地部署（省钱）    │
    │  └── 否 ↓                   │
    │                              │
    │  有 GPU 资源吗？             │
    │  ├── 没有 → API              │
    │  └── 有 ↓                   │
    │                              │
    │  需要最新模型吗？            │
    │  ├── 是 → API（持续更新）    │
    │  └── 否 → 本地部署           │
    └─────────────────────────────┘
    """)


# ============================================================
# Part 5: 本地模型部署实战
# ============================================================

def demo_local_deployment():
    """
    本地 Embedding 模型部署实战。

    以 BGE-M3 为例，展示用 sentence-transformers 部署本地模型。
    """
    print("\n" + "=" * 60)
    print("Part 5: 本地模型部署实战")
    print("=" * 60)

    print("""
    最简单的方式：sentence-transformers

    安装：
    pip install sentence-transformers

    使用：
    """)

    print("""
    from sentence_transformers import SentenceTransformer

    # 加载模型（首次会自动下载）
    model = SentenceTransformer("BAAI/bge-m3")

    # 生成 Embedding
    texts = ["你好世界", "Hello world"]
    embeddings = model.encode(texts)

    print(embeddings.shape)  # (2, 1024)
    """)

    print("""
    常用本地模型及推荐配置：

    ┌────────────────────┬──────────┬────────┬────────────────┐
    │ 模型                │ 参数量   │ 显存   │ 推荐 GPU       │
    ├────────────────────┼──────────┼────────┼────────────────┤
    │ bge-small-zh-v1.5  │ 24M      │ 0.5GB  │ 任何 GPU/CPU   │
    │ bge-base-zh-v1.5   │ 102M     │ 1GB    │ 任何 GPU       │
    │ bge-large-zh-v1.5  │ 326M     │ 2GB    │ GTX 1060+      │
    │ BGE-M3             │ 568M     │ 3GB    │ RTX 3060+      │
    │ GTE-large          │ 335M     │ 2GB    │ GTX 1060+      │
    │ jina-embeddings-v3 │ 572M     │ 3GB    │ RTX 3060+      │
    │ nomic-embed-v1.5   │ 137M     │ 1GB    │ 任何 GPU/CPU   │
    └────────────────────┴──────────┴────────┴────────────────┘

    你的 RTX 4090 Laptop (16GB) 可以轻松跑所有本地 Embedding 模型。
    甚至可以同时跑 Embedding + Reranker + 小规模 LLM 推理。
    """)

    # 模拟性能对比
    print("\n📊 本地模型推理速度估算（RTX 4090 Laptop）：")
    print("─" * 50)
    estimates = [
        ("bge-small-zh", "24M", "~50,000 条/秒"),
        ("bge-base-zh", "102M", "~10,000 条/秒"),
        ("BGE-M3", "568M", "~3,000 条/秒"),
        ("GTE-large", "335M", "~5,000 条/秒"),
    ]
    for name, params, speed in estimates:
        print(f"   {name:<20} ({params}) → {speed}")

    print("\n   对比 API：千问 Embedding API 约 500-1000 条/秒（含网络延迟）")
    print("   本地部署在批量场景下速度优势明显。")


# ============================================================
# Part 6: 维度选择策略
# ============================================================

def demo_dimension_strategy():
    """
    Embedding 维度选择：不是越大越好。

    高维度：信息更丰富，但存储和计算成本更高
    低维度：速度快、省空间，但可能损失精度
    """
    print("\n" + "=" * 60)
    print("Part 6: 维度选择策略")
    print("=" * 60)

    print("""
    维度与质量的关系：

    维度    质量损失    存储(100万条)   检索速度    适用场景
    ────    ────────    ────────────   ────────    ────────
    2048    基准        ~8 GB          最慢        追求极致质量
    1024    < 1%        ~4 GB          较慢        通用推荐
    768     1-2%        ~3 GB          中等        平衡之选
    512     2-3%        ~2 GB          较快        资源受限
    256     3-5%        ~1 GB          快          边缘设备/大规模
    128     5-8%        ~0.5 GB        很快        快速原型/移动端
    64      8-15%       ~0.25 GB       最快        概念验证
    """)

    print("""
    Matryoshka 维度裁剪：

    部分模型（OpenAI text-embedding-3、Jina v3、千问 v4）支持"维度裁剪"——
    用高维度训练的模型，直接截取前 N 维使用，质量损失比从头训练小模型要少。

    例如千问 text-embedding-v4：
    - 2048 维：最高精度
    - 1024 维：推荐默认
    - 768 维：平衡
    - 256 维：轻量

    使用方式（千问 API）：
    response = client.embeddings.create(
        model="text-embedding-v4",
        input=["文本"],
        dimensions=768   # 指定维度
    )
    """)

    # 实际演示维度裁剪
    print("\n📊 模拟维度裁剪效果：")
    print("─" * 50)

    np.random.seed(42)
    full_dim = 1024
    vec_a = np.random.randn(full_dim)
    vec_b = vec_a + np.random.randn(full_dim) * 0.1  # 相似的向量
    vec_c = np.random.randn(full_dim)  # 不相似的向量

    dims_to_test = [1024, 768, 512, 256, 128, 64]

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    print(f"{'维度':<8} {'A-B 相似度':<15} {'A-C 相似度':<15} {'区分度':<10}")
    print("─" * 48)
    for dim in dims_to_test:
        a, b, c = vec_a[:dim], vec_b[:dim], vec_c[:dim]
        sim_ab = cosine_sim(a, b)
        sim_ac = cosine_sim(a, c)
        gap = sim_ab - sim_ac
        print(f"{dim:<8} {sim_ab:<15.4f} {sim_ac:<15.4f} {gap:<10.4f}")

    print("\n结论：降到 256 维，区分度仍然够用。降到 64 维开始有明显损失。")


# ============================================================
# Part 7: 统一 Embedding 接口
# ============================================================

class UnifiedEmbedding:
    """
    统一的 Embedding 接口，支持多种模型切换。

    使用方式：
        emb = UnifiedEmbedding(model="dashscope/text-embedding-v4")
        vectors = emb.embed(["文本1", "文本2"])
    """

    # 预定义的模型配置
    MODEL_CONFIGS = {
        "dashscope/text-embedding-v4": {
            "provider": "dashscope",
            "model": "text-embedding-v4",
            "default_dim": 1024,
            "max_batch": 10,
            "max_tokens": 8192,
        },
        "openai/text-embedding-3-small": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "default_dim": 1536,
            "max_batch": 2048,
            "max_tokens": 8191,
        },
        "openai/text-embedding-3-large": {
            "provider": "openai",
            "model": "text-embedding-3-large",
            "default_dim": 3072,
            "max_batch": 2048,
            "max_tokens": 8191,
        },
        "local/BAAI/bge-m3": {
            "provider": "local",
            "model": "BAAI/bge-m3",
            "default_dim": 1024,
            "max_batch": 32,
            "max_tokens": 8192,
        },
        "mock": {
            "provider": "mock",
            "model": "mock",
            "default_dim": 1024,
            "max_batch": 999,
            "max_tokens": 99999,
        },
    }

    def __init__(self, model: str = "mock", dimension: Optional[int] = None):
        self.model_key = model
        config = self.MODEL_CONFIGS.get(model)
        if not config:
            raise ValueError(f"未知模型: {model}，可选: {list(self.MODEL_CONFIGS.keys())}")

        self.provider = config["provider"]
        self.model_name = config["model"]
        self.dimension = dimension or config["default_dim"]
        self.max_batch = config["max_batch"]
        self._local_model = None
        self._client = None

    def _init_provider(self):
        """延迟初始化 provider"""
        if self._client is not None or self._local_model is not None:
            return

        if self.provider == "dashscope":
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.environ.get("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        elif self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI()
        elif self.provider == "local":
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """生成 Embedding，自动处理分批"""
        if self.provider == "mock":
            vectors = []
            for text in texts:
                np.random.seed(hash(text) % 2**31)
                vec = np.random.randn(self.dimension).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                vectors.append(vec)
            return np.array(vectors)

        self._init_provider()

        all_vectors = []
        for i in range(0, len(texts), self.max_batch):
            batch = texts[i:i + self.max_batch]

            if self.provider == "local":
                vecs = self._local_model.encode(batch)
            else:
                kwargs = {"model": self.model_name, "input": batch}
                if self.provider in ("dashscope", "openai"):
                    kwargs["dimensions"] = self.dimension
                response = self._client.embeddings.create(**kwargs)
                vecs = [item.embedding for item in response.data]

            all_vectors.extend(vecs)

        return np.array(all_vectors, dtype=np.float32)

    def info(self) -> Dict:
        """返回模型信息"""
        return {
            "model": self.model_key,
            "provider": self.provider,
            "dimension": self.dimension,
            "max_batch": self.max_batch,
        }


def demo_unified_interface():
    """演示统一接口的使用"""
    print("\n" + "=" * 60)
    print("Part 7: 统一 Embedding 接口")
    print("=" * 60)

    # 切换不同模型只需要改一行
    models_to_test = ["mock", "mock"]  # 实际可换成真实模型

    for model_key in models_to_test:
        emb = UnifiedEmbedding(model=model_key)
        print(f"\n📦 模型：{emb.info()}")

        texts = ["Python 是编程语言", "JavaScript 用于前端开发", "今天天气很好"]
        vectors = emb.embed(texts)
        print(f"   输出：{vectors.shape}")

        # 计算相似度
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        print(f"   'Python' vs 'JavaScript'：{cosine_sim(vectors[0], vectors[1]):.4f}")
        print(f"   'Python' vs '天气'：{cosine_sim(vectors[0], vectors[2]):.4f}")

    print(f"\n📋 可用模型列表：")
    for key in UnifiedEmbedding.MODEL_CONFIGS:
        print(f"   - {key}")


# ============================================================
# 主函数
# ============================================================

def main():
    """运行所有演示"""
    demo_model_landscape()
    demo_mteb()
    demo_selection_framework()
    demo_api_vs_local()
    demo_local_deployment()
    demo_dimension_strategy()
    demo_unified_interface()

    print("\n" + "=" * 60)
    print("📝 本课要点总结")
    print("=" * 60)
    print("""
    1. 主流 Embedding 模型：OpenAI、千问、Cohere、BGE-M3、GTE、Jina
    2. MTEB 排行榜看 Retrieval 分数，注意语言和任务类型
    3. 中文 RAG 选千问 v4（性价比之王），数据敏感选 BGE-M3（本地部署）
    4. API vs 本地：数据敏感/大规模选本地，快速原型选 API
    5. 维度不是越大越好，256-1024 维通常够用
    6. 用统一接口封装多模型，切换成本为零
    """)


if __name__ == "__main__":
    main()

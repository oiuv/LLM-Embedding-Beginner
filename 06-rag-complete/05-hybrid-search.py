#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第29课：混合检索 (Hybrid Search)
================================

本课程讲解稠密向量检索 + 稀疏向量检索的组合策略。
纯向量检索在某些场景下会"翻车"，混合检索是生产系统的标配。

学习目标：
1. 理解纯向量检索的局限性
2. 掌握 BM25 稀疏检索的原理
3. 理解稠密 + 稀疏混合检索的策略
4. 能实现一个混合检索系统

"""

import numpy as np
import re
import math
from typing import List, Dict, Tuple, Optional
from collections import Counter
from dataclasses import dataclass, field


# ============================================================
# Part 1: 纯向量检索会"翻车"的场景
# ============================================================

def demo_vector_search_failures():
    """
    向量检索不是万能的，有几种典型失败场景。
    """
    print("=" * 60)
    print("Part 1: 纯向量检索的局限性")
    print("=" * 60)

    print("""
    场景一：精确关键词匹配

    查询："ERROR_CODE_4035"
    文档A："系统遇到 ERROR_CODE_4035 时需要重启服务"  ← 应该找到
    文档B："系统出错时的通用处理方法"                 ← 不相关

    向量检索的问题：
    "ERROR_CODE_4035" 的 Embedding 可能和"系统出错"更相似（语义接近）
    而不是和完全匹配的"ERROR_CODE_4035"（字符串精确匹配）
    → 向量检索可能返回 B 而不是 A

    场景二：罕见词/专业术语

    查询："HbA1c 糖化血红蛋白"
    文档A："HbA1c（糖化血红蛋白）是糖尿病监测的重要指标"  ← 精确匹配
    文档B："血糖检测的常用方法"                          ← 语义相关

    向量检索可能认为 B 更"语义相似"（都是关于血糖检测）
    但用户要的是精确包含"HbA1c"的文档

    场景三：否定词和反义

    查询："不要使用 deprecated API"
    文档A："本系统已移除所有 deprecated API"  ← 用户想要的
    文档B："deprecated API 的使用指南"        ← 包含关键词但意思相反

    向量检索难以区分"包含 deprecated API"和"不要使用 deprecated API"
    """)

    print("""
    核心问题：
    向量检索擅长"语义相似"，不擅长"精确匹配"。
    当查询包含专业术语、代码、ID、否定词时，向量检索可能失效。

    解决方案：
    稀疏检索（BM25）擅长精确匹配。
    两者组合 = 混合检索（Hybrid Search）。
    """)


# ============================================================
# Part 2: BM25 稀疏检索
# ============================================================

class BM25:
    """
    BM25（Best Matching 25）经典检索算法。

    原理：
    1. 文档分词
    2. 计算每个词在文档中的 TF（词频）
    3. 计算每个词的 IDF（逆文档频率）
    4. 查询时，把查询也分词，计算每个词的 BM25 分数，加权求和

    BM25 公式：
    score(q, d) = Σ IDF(qi) * (f(qi, d) * (k1 + 1)) / (f(qi, d) + k1 * (1 - b + b * |d|/avgdl))

    其中：
    - f(qi, d) = 词 qi 在文档 d 中的词频
    - |d| = 文档 d 的长度
    - avgdl = 平均文档长度
    - k1, b = 调节参数（通常 k1=1.5, b=0.75）

    优点：精确匹配能力强，不需要 GPU，速度快
    缺点：不理解语义（"电脑"和"计算机"是不同的词）
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_freqs = {}    # 每个词在多少文档中出现
        self.doc_lengths = []  # 每个文档的长度
        self.avg_doc_length = 0
        self.num_docs = 0
        self.tokenized_docs = []

    def _tokenize(self, text: str) -> List[str]:
        """
        中英文分词。

        简化实现：中文按字符，英文按空格。
        生产环境建议用 jieba 或其他分词器。
        """
        # 英文按空格分词，转小写
        text = text.lower()
        # 中文按 bigram（两个字符一组）
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
        english_words = re.findall(r'[a-z0-9_]+', text)

        tokens = list(english_words)
        for phrase in chinese_chars:
            # 中文 bigram
            for i in range(len(phrase) - 1):
                tokens.append(phrase[i:i+2])
            # 也保留单字
            for char in phrase:
                tokens.append(char)

        return tokens

    def index(self, documents: List[str]):
        """建立 BM25 索引"""
        self.documents = documents
        self.num_docs = len(documents)
        self.tokenized_docs = []
        df = Counter()  # document frequency

        for doc in documents:
            tokens = self._tokenize(doc)
            self.tokenized_docs.append(tokens)
            self.doc_lengths.append(len(tokens))
            # 每个词只计一次（是否出现在该文档中）
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1

        self.avg_doc_length = sum(self.doc_lengths) / self.num_docs if self.num_docs > 0 else 0
        self.doc_freqs = df

    def _idf(self, term: str) -> float:
        """计算 IDF（逆文档频率）"""
        df = self.doc_freqs.get(term, 0)
        # IDF 公式：log((N - df + 0.5) / (df + 0.5) + 1)
        return math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """搜索最相关的文档"""
        query_tokens = self._tokenize(query)
        scores = []

        for doc_idx, doc_tokens in enumerate(self.tokenized_docs):
            score = 0.0
            doc_len = self.doc_lengths[doc_idx]
            tf_counter = Counter(doc_tokens)

            for qt in query_tokens:
                if qt not in tf_counter:
                    continue
                tf = tf_counter[qt]
                idf = self._idf(qt)
                # BM25 公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                score += idf * numerator / denominator

            scores.append((doc_idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def demo_bm25():
    """演示 BM25 检索"""
    print("\n" + "=" * 60)
    print("Part 2: BM25 稀疏检索")
    print("=" * 60)

    documents = [
        "Python 是一种高级编程语言，广泛用于数据科学和机器学习",
        "JavaScript 是 Web 前端开发的核心语言，也用于后端开发",
        "ERROR_CODE_4035 表示数据库连接超时，需要检查网络配置",
        "系统出错时应该先检查日志文件，再联系技术支持",
        "HbA1c 糖化血红蛋白是糖尿病监测的重要指标",
        "血糖检测的常用方法包括空腹血糖和餐后血糖",
        "Git commit 和 Git push 是版本控制的基本操作",
        "Docker 容器化部署可以提高应用的可移植性",
    ]

    bm25 = BM25()
    bm25.index(documents)

    queries = [
        "ERROR_CODE_4035",
        "HbA1c 糖化血红蛋白",
        "Python 数据科学",
        "Git 操作",
    ]

    for query in queries:
        print(f"\n🔍 查询：'{query}'")
        results = bm25.search(query, top_k=3)
        for doc_idx, score in results:
            print(f"   [{score:.2f}] {documents[doc_idx][:50]}...")


# ============================================================
# Part 3: 混合检索策略
# ============================================================

class HybridSearch:
    """
    混合检索：BM25（稀疏）+ 向量检索（稠密）的组合。

    三种融合策略：
    1. RRF（Reciprocal Rank Fusion）：基于排名融合，最简单
    2. 加权分数融合：把两种分数归一化后加权求和
    3. 重排融合：先各取 Top-K，合并后用 Reranker 重排
    """

    def __init__(self, documents: List[str], embeddings: np.ndarray,
                 dimension: int = 1024):
        self.documents = documents
        self.embeddings = embeddings

        # BM25 索引
        self.bm25 = BM25()
        self.bm25.index(documents)

        # 向量索引
        self.vectors = embeddings.astype(np.float32)

    def _vector_search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """向量相似度搜索"""
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_embedding)
        similarities = np.dot(self.vectors, query_embedding) / (norms + 1e-10)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    # ---- 策略一：RRF（Reciprocal Rank Fusion） ----
    def search_rrf(self, query: str, query_embedding: np.ndarray,
                   top_k: int = 5, k: int = 60) -> List[Tuple[int, float]]:
        """
        RRF 融合：基于排名而非分数。

        公式：RRF_score(d) = Σ 1 / (k + rank_i(d))

        优点：不需要归一化分数，简单有效
        k=60 是论文推荐值
        """
        # 获取两种检索的排名
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        vector_results = self._vector_search(query_embedding, top_k=top_k * 2)

        # 计算 RRF 分数
        rrf_scores = {}

        for rank, (doc_idx, _) in enumerate(bm25_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (k + rank + 1)

        for rank, (doc_idx, _) in enumerate(vector_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (k + rank + 1)

        # 排序
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    # ---- 策略二：加权分数融合 ----
    def search_weighted(self, query: str, query_embedding: np.ndarray,
                        top_k: int = 5, alpha: float = 0.5) -> List[Tuple[int, float]]:
        """
        加权融合：归一化分数后加权求和。

        score(d) = α * score_vector(d) + (1-α) * score_bm25(d)

        alpha：向量检索的权重（0-1）
        - alpha > 0.5：更偏向语义检索
        - alpha < 0.5：更偏向关键词检索
        - alpha = 0.5：等权混合
        """
        # 获取所有文档的 BM25 分数
        bm25_all = self.bm25.search(query, top_k=len(self.documents))
        bm25_scores = {idx: score for idx, score in bm25_all}

        # 获取所有文档的向量分数
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_embedding)
        vector_scores_arr = np.dot(self.vectors, query_embedding) / (norms + 1e-10)

        # 归一化到 [0, 1]
        bm25_values = list(bm25_scores.values())
        bm25_min, bm25_max = min(bm25_values), max(bm25_values)
        vec_min, vec_max = float(vector_scores_arr.min()), float(vector_scores_arr.max())

        combined_scores = []
        for i in range(len(self.documents)):
            # 归一化 BM25
            bm25_norm = (bm25_scores.get(i, 0) - bm25_min) / (bm25_max - bm25_min + 1e-10)
            # 归一化向量
            vec_norm = (vector_scores_arr[i] - vec_min) / (vec_max - vec_min + 1e-10)
            # 加权融合
            combined = alpha * vec_norm + (1 - alpha) * bm25_norm
            combined_scores.append((i, combined))

        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores[:top_k]

    # ---- 策略三：各取 Top-K 后合并 ----
    def search_merge(self, query: str, query_embedding: np.ndarray,
                     top_k: int = 5) -> List[Tuple[int, float]]:
        """
        合并策略：BM25 取 Top-K，向量取 Top-K，合并去重后返回。
        适合配合 Reranker 使用（先粗召，再精排）。
        """
        bm25_results = self.bm25.search(query, top_k=top_k)
        vector_results = self._vector_search(query_embedding, top_k=top_k)

        # 合并去重
        seen = set()
        merged = []
        for doc_idx, score in bm25_results + vector_results:
            if doc_idx not in seen:
                seen.add(doc_idx)
                merged.append((doc_idx, score))

        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:top_k]


def demo_hybrid_search():
    """演示混合检索"""
    print("\n" + "=" * 60)
    print("Part 3: 混合检索对比")
    print("=" * 60)

    documents = [
        "Python 是一种高级编程语言，广泛用于数据科学和机器学习",
        "JavaScript 是 Web 前端开发的核心语言，也用于后端开发",
        "ERROR_CODE_4035 表示数据库连接超时，需要检查网络配置",
        "系统出错时应该先检查日志文件，再联系技术支持",
        "HbA1c 糖化血红蛋白是糖尿病监测的重要指标",
        "血糖检测的常用方法包括空腹血糖和餐后血糖",
        "Python 的虚拟环境用于隔离不同项目的依赖",
        "Docker 容器化部署可以提高应用的可移植性",
    ]

    np.random.seed(42)
    dimension = 128
    embeddings = np.random.randn(len(documents), dimension).astype(np.float32)

    hybrid = HybridSearch(documents, embeddings, dimension)

    # 模拟 query embedding
    query = "ERROR_CODE_4035"
    query_embedding = np.random.randn(dimension).astype(np.float32)

    print(f"\n🔍 查询：'{query}'")
    print(f"\n{'─' * 55}")

    # 纯 BM25
    print("\n📋 纯 BM25（关键词匹配）：")
    bm25 = BM25()
    bm25.index(documents)
    for idx, score in bm25.search(query, top_k=3):
        print(f"   [{score:.2f}] {documents[idx][:50]}...")

    # 纯向量
    print("\n📋 纯向量检索（语义匹配）：")
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    sims = np.dot(embeddings, query_embedding) / (norms + 1e-10)
    top_indices = np.argsort(sims)[-3:][::-1]
    for idx in top_indices:
        print(f"   [{sims[idx]:.4f}] {documents[idx][:50]}...")

    # 混合 RRF
    print("\n📋 混合检索 RRF：")
    for idx, score in hybrid.search_rrf(query, query_embedding, top_k=3):
        print(f"   [{score:.4f}] {documents[idx][:50]}...")

    print(f"\n{'─' * 55}")
    print("观察：BM25 精确找到了 ERROR_CODE_4035，向量检索可能找不准。")
    print("混合检索结合了两者优势。")


# ============================================================
# Part 4: alpha 参数调优
# ============================================================

def demo_alpha_tuning():
    """
    混合检索的 alpha 参数怎么调？

    alpha = 向量检索权重
    1-alpha = BM25 权重

    不同场景需要不同的 alpha。
    """
    print("\n" + "=" * 60)
    print("Part 4: alpha 参数调优指南")
    print("=" * 60)

    print("""
    alpha 值的选择取决于查询类型：

    ┌────────────────────────────┬───────┬────────────────────────┐
    │ 查询类型                    │ alpha │ 原因                    │
    ├────────────────────────────┼───────┼────────────────────────┤
    │ 自然语言问题                │ 0.7   │ 语义理解更重要           │
    │ "Python 怎么读文件？"       │       │                        │
    ├────────────────────────────┼───────┼────────────────────────┤
    │ 精确关键词/代码/ID          │ 0.3   │ 精确匹配更重要           │
    │ "ERROR_CODE_4035"          │       │                        │
    ├────────────────────────────┼───────┼────────────────────────┤
    │ 混合查询                    │ 0.5   │ 两者都重要               │
    │ "HbA1c 糖化血红蛋白检测"    │       │                        │
    ├────────────────────────────┼───────┼────────────────────────┤
    │ 不确定查询类型              │ 0.5   │ 等权是最安全的默认值     │
    └────────────────────────────┴───────┴────────────────────────┘
    """)

    print("""
    生产环境的高级策略：

    1. 自适应 alpha
       根据查询特征动态调整 alpha：
       - 查询包含代码/ID/特殊字符 → 降低 alpha（偏向 BM25）
       - 查询是自然语言句子 → 提高 alpha（偏向向量）
       - 查询很短（< 3 词）→ 降低 alpha
       - 查询较长 → 提高 alpha

    2. 查询分类
       先用一个分类器判断查询类型（关键词型/语义型/混合型）
       再选择对应的 alpha

    3. 用 Reranker 代替 alpha 调优
       取 BM25 Top-20 + 向量 Top-20，合并后用 Reranker 精排
       Reranker 会自动学习最佳的融合方式
       这是生产环境最推荐的方案
    """)


# ============================================================
# Part 5: BGE-M3 三合一检索
# ============================================================

def demo_bge_m3_modes():
    """
    BGE-M3 的独特优势：一个模型支持三种检索模式。

    1. Dense（稠密向量）：标准的 Embedding 语义检索
    2. Sparse（稀疏向量）：类似 BM25 的词级匹配
    3. ColBERT（多向量）：Token 级别的细粒度匹配

    不需要分别维护 BM25 索引和向量索引，一个模型搞定。
    """
    print("\n" + "=" * 60)
    print("Part 5: BGE-M3 三合一检索")
    print("=" * 60)

    print("""
    BGE-M3 的三种检索模式：

    ┌────────────┬──────────────┬────────────────────────────┐
    │ 模式        │ 输出          │ 擅长                        │
    ├────────────┼──────────────┼────────────────────────────┤
    │ Dense      │ 1024 维向量   │ 语义相似、同义词替换         │
    │ Sparse     │ 词级权重      │ 精确匹配、关键词命中         │
    │ ColBERT    │ 每 token 一个 │ 细粒度匹配、长文档           │
    │            │ 向量          │                              │
    └────────────┴──────────────┴────────────────────────────┘

    使用方式（FlagEmbedding 库）：

    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    # 三种模式同时输出
    output = model.encode(
        ["Python 编程语言"],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )

    dense_vec = output['dense_vecs']       # (1, 1024)
    sparse_vec = output['lexical_weights'] # 词级权重
    colbert_vecs = output['colbert_vecs']  # 多向量

    # 混合检索
    scores = (
        dense_score * 0.4 +
        sparse_score * 0.3 +
        colbert_score * 0.3
    )
    """)

    print("""
    BGE-M3 vs 分别维护 BM25 + 向量索引：

    方案              优点                      缺点
    ────              ────                      ────
    BM25 + FAISS      各自成熟、可独立优化        需要维护两套系统
    BGE-M3 三合一     一套系统、统一训练          需要 GPU、模型较大

    推荐：
    - 小团队/快速上线 → BM25 + FAISS（用成熟的库）
    - 大团队/追求极致 → BGE-M3 三合一
    """)


# ============================================================
# 主函数
# ============================================================

def main():
    """运行所有演示"""
    demo_vector_search_failures()
    demo_bm25()
    demo_hybrid_search()
    demo_alpha_tuning()
    demo_bge_m3_modes()

    print("\n" + "=" * 60)
    print("📝 本课要点总结")
    print("=" * 60)
    print("""
    1. 纯向量检索在精确匹配、专业术语、否定词场景会"翻车"
    2. BM25 是经典的稀疏检索算法，擅长精确匹配，不需要 GPU
    3. 混合检索 = 向量（语义）+ BM25（关键词），互补短板
    4. 融合策略：RRF（最简单）、加权分数（可调 alpha）、合并+Reranker（最推荐）
    5. alpha 参数：自然语言→0.7，关键词→0.3，不确定→0.5
    6. BGE-M3 一个模型支持 Dense+Sparse+ColBERT 三种模式
    7. 生产环境推荐：混合检索 + Reranker 精排
    """)


if __name__ == "__main__":
    main()

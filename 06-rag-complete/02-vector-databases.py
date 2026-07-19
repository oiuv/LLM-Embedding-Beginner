#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第26课：向量数据库 (Vector Database)
====================================

本课程讲解向量数据库的核心原理、主流方案对比、以及实战使用。
向量数据库是 RAG 系统的"记忆库"，存储和检索 Embedding 向量。

学习目标：
1. 理解为什么需要向量数据库（不能用 pkl 文件存向量）
2. 掌握向量索引的核心算法（暴力搜索 → HNSW → IVF）
3. 对比主流向量数据库（FAISS、Chroma、Milvus、Qdrant）
4. 能用 FAISS 和 Chroma 构建向量检索系统

"""

import numpy as np
import time
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# ============================================================
# Part 1: 为什么需要向量数据库？
# ============================================================

def demo_why_vector_db():
    """
    演示用 pkl 文件存向量的问题，以及向量数据库的优势。
    """
    print("=" * 60)
    print("Part 1: 为什么需要向量数据库？")
    print("=" * 60)

    # 模拟数据
    num_vectors = 10000
    dimension = 1024
    vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
    query = np.random.randn(dimension).astype(np.float32)

    print(f"\n📊 数据规模：{num_vectors} 个向量，每个 {dimension} 维")

    # ---- 方式一：暴力搜索（pkl 文件的方式） ----
    print("\n❌ 方式一：暴力搜索（遍历所有向量）")
    start = time.time()
    # 计算 query 与所有向量的余弦相似度
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
    similarities = np.dot(vectors, query) / (norms + 1e-10)
    top_k_indices = np.argsort(similarities)[-5:][::-1]
    brute_time = time.time() - start
    print(f"   耗时：{brute_time*1000:.1f} ms")
    print(f"   Top-5 索引：{top_k_indices}")
    print(f"   问题：{num_vectors} 条已经要 {brute_time*1000:.0f}ms，")
    print(f"         100 万条就要 ~{brute_time*1000*(1000000/num_vectors):.0f}ms，不可接受")

    # ---- 方式二：向量索引（近似最近邻） ----
    print("\n✅ 方式二：向量索引（HNSW / IVF）")
    print("   原理：预先构建索引结构，搜索时不需要遍历所有向量")
    print("   100 万向量的检索时间：< 10ms")
    print("   代价：建索引需要时间和内存，精度略有损失（99%+）")


# ============================================================
# Part 2: 向量索引核心算法
# ============================================================

class BruteForceIndex:
    """
    暴力搜索索引：遍历所有向量，找到最近的 K 个。

    优点：精度 100%，实现简单
    缺点：数据量大时速度慢（O(n*d)）
    适用：数据量 < 10,000，或作为 baseline
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = None
        self.metadata = []

    def add(self, vectors: np.ndarray, metadata: List[Dict] = None):
        """添加向量"""
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        if metadata:
            self.metadata.extend(metadata)

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """搜索最相似的 K 个向量，返回 (索引, 相似度)"""
        # 余弦相似度
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query)
        similarities = np.dot(self.vectors, query) / (norms + 1e-10)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]


class IVFIndex:
    """
    IVF（Inverted File Index）索引：先聚类，再在最近的簇中搜索。

    原理：
    1. 建索引时：用 K-Means 把所有向量分成 nlist 个簇
    2. 搜索时：先找到最近的 nprobe 个簇，只在这些簇中暴力搜索

    优点：搜索速度快（只搜索部分数据）
    缺点：建索引慢，精度受 nprobe 影响
    适用：数据量 10 万 - 1000 万
    """

    def __init__(self, dimension: int, nlist: int = 10):
        self.dimension = dimension
        self.nlist = nlist  # 簇的数量
        self.centroids = None
        self.clusters = {}  # {cluster_id: [(vector, metadata)]}
        self.all_vectors = None

    def train(self, vectors: np.ndarray):
        """用 K-Means 训练聚类中心"""
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.nlist, random_state=42, n_init=10)
        kmeans.fit(vectors)
        self.centroids = kmeans.cluster_centers_
        self._kmeans = kmeans
        print(f"   训练完成：{self.nlist} 个簇，{len(vectors)} 个向量")

    def add(self, vectors: np.ndarray, metadata: List[Dict] = None):
        """将向量分配到最近的簇"""
        if self.all_vectors is None:
            self.all_vectors = vectors
        else:
            self.all_vectors = np.vstack([self.all_vectors, vectors])

        labels = self._kmeans.predict(vectors)
        for i, (vec, label) in enumerate(zip(vectors, labels)):
            label = int(label)
            if label not in self.clusters:
                self.clusters[label] = []
            meta = metadata[i] if metadata else {}
            self.clusters[label].append((vec, meta))

    def search(self, query: np.ndarray, top_k: int = 5, nprobe: int = 3) -> List[Tuple[int, float]]:
        """
        搜索：先找最近的 nprobe 个簇，再在簇内暴力搜索。
        """
        # 找最近的 nprobe 个簇
        centroid_sims = np.dot(self.centroids, query) / (
            np.linalg.norm(self.centroids, axis=1) * np.linalg.norm(query) + 1e-10
        )
        nearest_clusters = np.argsort(centroid_sims)[-nprobe:][::-1]

        # 在这些簇中搜索
        candidates = []
        for cid in nearest_clusters:
            if cid not in self.clusters:
                continue
            for vec, meta in self.clusters[cid]:
                sim = np.dot(vec, query) / (np.linalg.norm(vec) * np.linalg.norm(query) + 1e-10)
                candidates.append((sim, meta))

        # 取 Top-K
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:top_k]


def demo_index_algorithms():
    """对比暴力搜索和 IVF 索引的性能"""
    print("\n" + "=" * 60)
    print("Part 2: 向量索引算法对比")
    print("=" * 60)

    # 生成测试数据
    num_vectors = 5000
    dimension = 256
    vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
    query = np.random.randn(dimension).astype(np.float32)

    # 暴力搜索
    print(f"\n📊 测试数据：{num_vectors} 个 {dimension} 维向量")
    print("\n🔍 暴力搜索：")
    bf = BruteForceIndex(dimension)
    bf.add(vectors, [{"id": i} for i in range(num_vectors)])
    start = time.time()
    bf_results = bf.search(query, top_k=5)
    bf_time = time.time() - start
    print(f"   耗时：{bf_time*1000:.2f} ms")
    print(f"   Top-5 相似度：{[f'{s:.4f}' for _, s in bf_results]}")

    # IVF 索引
    print("\n🔍 IVF 索引（nlist=10, nprobe=3）：")
    ivf = IVFIndex(dimension, nlist=10)
    start = time.time()
    ivf.train(vectors)
    ivf.add(vectors, [{"id": i} for i in range(num_vectors)])
    build_time = time.time() - start
    print(f"   建索引耗时：{build_time*1000:.2f} ms")

    start = time.time()
    ivf_results = ivf.search(query, top_k=5, nprobe=3)
    ivf_time = time.time() - start
    print(f"   搜索耗时：{ivf_time*1000:.2f} ms")
    print(f"   Top-5 相似度：{[f'{s:.4f}' for s, _ in ivf_results]}")
    print(f"   加速比：{bf_time/ivf_time:.1f}x")


# ============================================================
# Part 3: FAISS 实战
# ============================================================

def demo_faiss():
    """
    FAISS（Facebook AI Similarity Search）实战。

    特点：
    - Meta 开源，性能极强
    - 支持 CPU 和 GPU
    - 支持多种索引类型
    - 是最流行的向量检索库（不是数据库）
    """
    print("\n" + "=" * 60)
    print("Part 3: FAISS 实战")
    print("=" * 60)

    try:
        import faiss
        has_faiss = True
    except ImportError:
        has_faiss = False
        print("\n⚠️ 未安装 faiss-cpu，使用模拟演示")
        print("   安装：pip install faiss-cpu")

    if has_faiss:
        # 真实 FAISS 演示
        dimension = 128
        num_vectors = 10000

        print(f"\n📊 测试数据：{num_vectors} 个 {dimension} 维向量")

        # 生成数据
        vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
        query = np.random.randn(1, dimension).astype(np.float32)

        # ---- Flat 索引（暴力搜索） ----
        print("\n--- FAISS Flat 索引（暴力搜索，精度 100%）---")
        index_flat = faiss.IndexFlatL2(dimension)
        start = time.time()
        index_flat.add(vectors)
        build_time = time.time() - start
        print(f"   建索引：{build_time*1000:.1f} ms，向量数：{index_flat.ntotal}")

        start = time.time()
        distances, indices = index_flat.search(query, 5)
        search_time = time.time() - start
        print(f"   搜索：{search_time*1000:.2f} ms")
        print(f"   Top-5 索引：{indices[0]}")

        # ---- IVF 索引（近似搜索） ----
        print("\n--- FAISS IVF 索引（近似搜索，更快）---")
        nlist = 50  # 簇数量
        quantizer = faiss.IndexFlatL2(dimension)
        index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)

        start = time.time()
        index_ivf.train(vectors)
        index_ivf.add(vectors)
        build_time = time.time() - start
        print(f"   建索引：{build_time*1000:.1f} ms，向量数：{index_ivf.ntotal}")

        index_ivf.nprobe = 5  # 搜索 5 个簇
        start = time.time()
        distances, indices = index_ivf.search(query, 5)
        search_time = time.time() - start
        print(f"   搜索：{search_time*1000:.2f} ms")
        print(f"   Top-5 索引：{indices[0]}")

        # ---- 保存和加载索引 ----
        print("\n--- FAISS 索引保存/加载 ---")
        index_path = "test_index.faiss"
        faiss.write_index(index_flat, index_path)
        loaded_index = faiss.read_index(index_path)
        print(f"   索引已保存到 {index_path}")
        print(f"   加载后向量数：{loaded_index.ntotal}")
        os.remove(index_path)  # 清理

    else:
        # 模拟演示
        print("\n📋 FAISS 核心 API 速查：")
        print("""
        # 创建索引
        import faiss
        index = faiss.IndexFlatL2(dim)          # 暴力搜索
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)  # IVF

        # 添加向量
        index.add(vectors)                      # vectors: np.ndarray, shape (n, dim)

        # 搜索
        distances, indices = index.search(query, k)  # query: (1, dim)

        # 保存/加载
        faiss.write_index(index, "index.faiss")
        index = faiss.read_index("index.faiss")

        # GPU 加速
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        """)


# ============================================================
# Part 4: Chroma 实战
# ============================================================

def demo_chroma():
    """
    Chroma 实战：最简单的向量数据库。

    特点：
    - 纯 Python，pip install chromadb 即可
    - 内置 Embedding 支持（可选）
    - 内置元数据过滤
    - 适合快速原型和小规模应用（< 100 万向量）
    """
    print("\n" + "=" * 60)
    print("Part 4: Chroma 实战")
    print("=" * 60)

    try:
        import chromadb
        has_chroma = True
    except ImportError:
        has_chroma = False
        print("\n⚠️ 未安装 chromadb，使用代码示例演示")
        print("   安装：pip install chromadb")

    if has_chroma:
        # 真实 Chroma 演示
        print("\n📊 使用 Chroma 构建向量数据库：")

        # 创建客户端（内存模式）
        client = chromadb.Client()

        # 创建集合（相当于一个表）
        collection = client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
        print(f"   集合已创建：{collection.name}")

        # 添加文档（Chroma 自动处理 Embedding）
        documents = [
            "Python 是一种高级编程语言",
            "JavaScript 是 Web 前端开发的核心语言",
            "Java 广泛应用于企业级开发",
            "Go 语言以并发性能著称",
            "Rust 注重内存安全和性能",
        ]
        metadatas = [
            {"category": "language", "type": "general"},
            {"category": "language", "type": "web"},
            {"category": "language", "type": "enterprise"},
            {"category": "language", "type": "concurrent"},
            {"category": "language", "type": "system"},
        ]
        ids = [f"doc_{i}" for i in range(len(documents))]

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"   已添加 {len(documents)} 个文档")

        # 查询
        print("\n🔍 查询：'编程语言' 相关文档")
        results = collection.query(
            query_texts=["编程语言"],
            n_results=3
        )
        for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
            print(f"   [{i+1}] 相似度={1-dist:.4f} | {doc}")

        # 带元数据过滤的查询
        print("\n🔍 查询（过滤：type=web）：'开发语言'")
        results = collection.query(
            query_texts=["开发语言"],
            n_results=3,
            where={"type": "web"}
        )
        for i, doc in enumerate(results["documents"][0]):
            print(f"   [{i+1}] {doc}")

        # 清理
        client.delete_collection("documents")

    else:
        print("\n📋 Chroma 核心 API 速查：")
        print("""
        import chromadb

        # 创建客户端
        client = chromadb.PersistentClient(path="./chroma_db")  # 持久化
        client = chromadb.Client()                               # 内存模式

        # 创建集合
        collection = client.create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"}
        )

        # 添加文档
        collection.add(
            documents=["文本1", "文本2"],
            metadatas=[{"source": "a"}, {"source": "b"}],
            ids=["id1", "id2"]
        )

        # 查询
        results = collection.query(
            query_texts=["查询文本"],
            n_results=5,
            where={"source": "a"}       # 元数据过滤
        )

        # 删除
        collection.delete(ids=["id1"])
        client.delete_collection("docs")
        """)


# ============================================================
# Part 5: 主流向量数据库对比
# ============================================================

def demo_comparison():
    """主流向量数据库/库的对比"""
    print("\n" + "=" * 60)
    print("Part 5: 主流向量数据库对比")
    print("=" * 60)

    products = [
        {
            "name": "FAISS",
            "type": "库（不是数据库）",
            "language": "C++/Python",
            "scale": "百万-十亿",
            "pros": "性能最强，GPU 支持，Meta 维护",
            "cons": "没有持久化/元数据过滤，需要自己封装",
            "best_for": "大规模生产环境，对性能要求极高",
        },
        {
            "name": "Chroma",
            "type": "向量数据库",
            "language": "Python",
            "scale": "< 100 万",
            "pros": "最简单，内置 Embedding，元数据过滤",
            "cons": "大规模性能一般，功能相对基础",
            "best_for": "快速原型、小项目、学习",
        },
        {
            "name": "Qdrant",
            "type": "向量数据库",
            "language": "Rust",
            "scale": "百万级",
            "pros": "性能好，功能丰富（过滤、Payload），REST API",
            "cons": "需要独立部署服务",
            "best_for": "中大规模生产环境",
        },
        {
            "name": "Milvus",
            "type": "向量数据库",
            "language": "Go/C++",
            "scale": "十亿级",
            "pros": "分布式架构，超大规模支持，功能最全",
            "cons": "部署复杂（依赖 etcd/MinIO），资源消耗大",
            "best_for": "企业级大规模应用",
        },
        {
            "name": "Weaviate",
            "type": "向量数据库",
            "language": "Go",
            "scale": "千万级",
            "pros": "内置向量化模块，GraphQL API，多模态支持",
            "cons": "学习曲线较陡",
            "best_for": "需要多模态检索的应用",
        },
        {
            "name": "Pinecone",
            "type": "云服务",
            "language": "SaaS",
            "scale": "十亿级",
            "pros": "零运维，开箱即用，自动扩展",
            "cons": "付费，数据在云端，厂商锁定",
            "best_for": "不想运维、快速上线",
        },
    ]

    for p in products:
        print(f"\n{'─' * 55}")
        print(f"📦 {p['name']} ({p['type']})")
        print(f"   语言：{p['language']}  |  规模：{p['scale']}")
        print(f"   ✅ {p['pros']}")
        print(f"   ❌ {p['cons']}")
        print(f"   🎯 最适合：{p['best_for']}")

    print(f"\n{'─' * 55}")
    print("\n📊 选型决策树：")
    print("""
    你的场景是什么？
    ├── 学习/快速原型 → Chroma（最简单）
    ├── 小项目（< 10 万条）→ Chroma 或 FAISS
    ├── 中等规模（10-100 万）→ Qdrant 或 FAISS
    ├── 大规模（> 100 万）→ FAISS / Milvus
    ├── 超大规模（> 1000 万）→ Milvus / Pinecone
    └── 不想运维 → Pinecone（付费 SaaS）
    """)


# ============================================================
# Part 6: 自建简易向量数据库
# ============================================================

class SimpleVectorDB:
    """
    用 FAISS 封装的简易向量数据库。

    功能：
    - 添加向量 + 文本 + 元数据
    - 相似度搜索
    - 元数据过滤
    - 持久化保存/加载

    适合学习理解和小规模应用。
    """

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.texts = []
        self.metadata_list = []
        self.ids = []
        self._index = None
        self._build_index()

    def _build_index(self):
        """构建/重建 FAISS 索引"""
        try:
            import faiss
            self._index = faiss.IndexFlatL2(self.dimension)
            self._has_faiss = True
        except ImportError:
            self._has_faiss = False
            self._vectors = None

    def add(self, texts: List[str], embeddings: np.ndarray,
            metadata: List[Dict] = None, ids: List[str] = None):
        """添加文档"""
        start_idx = len(self.texts)
        self.texts.extend(texts)
        self.metadata_list.extend(metadata or [{} for _ in texts])
        self.ids.extend(ids or [f"doc_{start_idx + i}" for i in range(len(texts))])

        if self._has_faiss:
            self._index.add(embeddings.astype(np.float32))
        else:
            if self._vectors is None:
                self._vectors = embeddings
            else:
                self._vectors = np.vstack([self._vectors, embeddings])

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               filter_fn=None) -> List[Dict]:
        """搜索最相似的文档"""
        query = query_embedding.reshape(1, -1).astype(np.float32)

        if self._has_faiss:
            distances, indices = self._index.search(query, top_k * 3)  # 多取一些用于过滤
            candidates = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.texts):
                    continue
                if filter_fn and not filter_fn(self.metadata_list[idx]):
                    continue
                candidates.append({
                    "id": self.ids[idx],
                    "text": self.texts[idx],
                    "metadata": self.metadata_list[idx],
                    "distance": float(dist),
                    "score": 1.0 / (1.0 + float(dist)),  # 转换为 0-1 的相似度
                })
                if len(candidates) >= top_k:
                    break
        else:
            # 暴力搜索 fallback
            norms = np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(query_embedding)
            similarities = np.dot(self._vectors, query_embedding) / (norms + 1e-10)
            top_indices = np.argsort(similarities)[-top_k*3:][::-1]

            candidates = []
            for idx in top_indices:
                if filter_fn and not filter_fn(self.metadata_list[idx]):
                    continue
                candidates.append({
                    "id": self.ids[idx],
                    "text": self.texts[idx],
                    "metadata": self.metadata_list[idx],
                    "score": float(similarities[idx]),
                })
                if len(candidates) >= top_k:
                    break

        return candidates

    def save(self, path: str):
        """保存到文件"""
        data = {
            "texts": self.texts,
            "metadata": self.metadata_list,
            "ids": self.ids,
            "dimension": self.dimension,
        }
        with open(os.path.join(path, "data.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if self._has_faiss:
            import faiss
            faiss.write_index(self._index, os.path.join(path, "index.faiss"))
        else:
            np.save(os.path.join(path, "vectors.npy"), self._vectors)

        print(f"   已保存到 {path}（{len(self.texts)} 个文档）")

    def load(self, path: str):
        """从文件加载"""
        with open(os.path.join(path, "data.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        self.texts = data["texts"]
        self.metadata_list = data["metadata"]
        self.ids = data["ids"]
        self.dimension = data["dimension"]

        if self._has_faiss:
            import faiss
            self._index = faiss.read_index(os.path.join(path, "index.faiss"))
        else:
            self._vectors = np.load(os.path.join(path, "vectors.npy"))

        print(f"   已加载 {len(self.texts)} 个文档")

    def stats(self) -> Dict:
        """返回统计信息"""
        return {
            "total_documents": len(self.texts),
            "dimension": self.dimension,
            "index_type": "FAISS Flat" if self._has_faiss else "Numpy BruteForce",
        }


def demo_simple_vectordb():
    """演示简易向量数据库的使用"""
    print("\n" + "=" * 60)
    print("Part 6: 自建简易向量数据库")
    print("=" * 60)

    db = SimpleVectorDB(dimension=128)

    # 模拟添加文档
    documents = [
        ("Python 是一种高级编程语言", {"source": "python.md", "type": "text"}),
        ("JavaScript 是 Web 前端核心语言", {"source": "js.md", "type": "text"}),
        ("GET /api/users 返回用户列表", {"source": "api.md", "type": "code"}),
        ("POST /api/users 创建新用户", {"source": "api.md", "type": "code"}),
        ("机器学习需要大量数据训练模型", {"source": "ml.md", "type": "text"}),
    ]

    # 模拟 Embedding（实际项目中用真实的 Embedding API）
    embeddings = np.random.randn(len(documents), 128).astype(np.float32)
    texts = [d[0] for d in documents]
    metadatas = [d[1] for d in documents]

    db.add(texts, embeddings, metadata=metadatas, ids=[f"doc_{i}" for i in range(len(documents))])
    print(f"\n📊 数据库统计：{db.stats()}")

    # 搜索
    query_embedding = np.random.randn(128).astype(np.float32)
    results = db.search(query_embedding, top_k=3)
    print(f"\n🔍 搜索结果（Top-3）：")
    for i, r in enumerate(results):
        print(f"   [{i+1}] score={r['score']:.4f} | {r['text']} | {r['metadata']}")

    # 带过滤的搜索
    print(f"\n🔍 搜索结果（过滤 type=code）：")
    results = db.search(query_embedding, top_k=3, filter_fn=lambda m: m.get("type") == "code")
    for i, r in enumerate(results):
        print(f"   [{i+1}] score={r['score']:.4f} | {r['text']} | {r['metadata']}")


# ============================================================
# 主函数
# ============================================================

def main():
    """运行所有演示"""
    demo_why_vector_db()
    demo_index_algorithms()
    demo_faiss()
    demo_chroma()
    demo_comparison()
    demo_simple_vectordb()

    print("\n" + "=" * 60)
    print("📝 本课要点总结")
    print("=" * 60)
    print("""
    1. pkl 文件存向量只适合原型，生产环境必须用向量数据库
    2. 核心索引算法：暴力搜索（小数据）→ IVF（中等）→ HNSW（大规模）
    3. FAISS 是性能最强的向量检索库，但不是数据库（没有持久化/过滤）
    4. Chroma 是最简单的向量数据库，适合快速原型和学习
    5. 生产选型：小项目 Chroma，中等 Qdrant，大规模 Milvus/FAISS
    6. 元数据过滤是向量数据库的核心功能之一，必须支持
    """)


if __name__ == "__main__":
    main()

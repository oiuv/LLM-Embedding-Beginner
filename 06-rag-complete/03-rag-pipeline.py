#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第27课：RAG Pipeline 完整实现
============================

本课程把 Chunking + Embedding + 向量数据库 + LLM 串起来，
实现一个完整的 RAG（Retrieval-Augmented Generation）系统。

学习目标：
1. 理解 RAG 的完整流程：文档 → 分块 → 向量化 → 存储 → 检索 → 生成
2. 掌握 Prompt 构造技巧（怎么把检索结果喂给 LLM）
3. 能独立实现一个可用的 RAG 问答系统

"""

import numpy as np
import json
import os
import re
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================
# Part 1: RAG 是什么？为什么需要 RAG？
# ============================================================

def demo_what_is_rag():
    """
    RAG = Retrieval-Augmented Generation（检索增强生成）

    核心问题：LLM 的知识有截止日期，且不了解你的私有数据。
    RAG 的解法：先从你的知识库中检索相关内容，再让 LLM 基于检索结果回答。
    """
    print("=" * 60)
    print("Part 1: RAG 是什么？为什么需要 RAG？")
    print("=" * 60)

    print("""
    没有 RAG 的 LLM：

    User: "我们公司的年假政策是什么？"
    LLM:  "抱歉，我没有你们公司的内部信息。"
            ↑ LLM 不知道你的私有数据

    有 RAG 的 LLM：

    User: "我们公司的年假政策是什么？"
    ├── Step 1: 检索 → 从知识库找到《员工手册》相关段落
    ├── Step 2: 构造 Prompt → 把段落塞进 Prompt
    └── Step 3: 生成 → LLM 基于段落回答
    LLM:  "根据公司员工手册，入职满一年的员工享有 10 天年假..."
            ↑ LLM 有了上下文，能准确回答
    """)

    print("\n📊 RAG vs 其他方案对比：")
    print("─" * 55)
    print(f"{'方案':<15} {'优点':<20} {'缺点'}")
    print("─" * 55)
    print(f"{'直接问 LLM':<15} {'简单、免费':<20} {'不知道私有数据，会编造'}")
    print(f"{'微调模型':<15} {'可以学习私有知识':<20} {'成本高、知识会过期'}")
    print(f"{'RAG':<15} {'实时、准确、可追溯':<20} {'依赖检索质量'}")
    print("─" * 55)
    print("结论：RAG 是让 LLM 使用私有知识的最佳方案（成本低、实时更新、可追溯）")


# ============================================================
# Part 2: RAG Pipeline 架构
# ============================================================

def demo_rag_architecture():
    """展示 RAG 的完整架构"""
    print("\n" + "=" * 60)
    print("Part 2: RAG Pipeline 完整架构")
    print("=" * 60)

    print("""
    ┌──────────────────────────────────────────────────────────┐
    │                    RAG 完整流程                            │
    │                                                           │
    │  【离线阶段：索引构建】                                    │
    │                                                           │
    │  原始文档 ──→ 分块 ──→ Embedding ──→ 向量数据库            │
    │  (PDF/MD/..)  (Chunk)   (向量化)     (存储+索引)           │
    │                                                           │
    │  ─────────────────────────────────────────────────────── │
    │                                                           │
    │  【在线阶段：查询应答】                                    │
    │                                                           │
    │  用户问题 ──→ Embedding ──→ 向量检索 ──→ Top-K 相关块      │
    │                                       │                   │
    │                                       ▼                   │
    │                              构造 Prompt                   │
    │                              (问题 + 检索结果)              │
    │                                       │                   │
    │                                       ▼                   │
    │                              LLM 生成回答                  │
    │                                       │                   │
    │                                       ▼                   │
    │                              返回用户 + 引用来源            │
    └──────────────────────────────────────────────────────────┘
    """)


# ============================================================
# Part 3: 核心组件实现
# ============================================================

@dataclass
class Document:
    """文档数据结构"""
    content: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class Chunk:
    """文档块"""
    content: str
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


class TextChunker:
    """
    文本分块器（简化版，核心逻辑来自第25课）
    """

    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """递归分块"""
        separators = ["\n\n", "\n", "。", ".", " "]
        raw_chunks = self._split_recursive(text, separators)

        chunks = []
        for i, content in enumerate(raw_chunks):
            # 添加重叠
            if i > 0 and self.overlap > 0:
                prev = raw_chunks[i - 1]
                overlap_text = prev[-self.overlap:] if len(prev) > self.overlap else prev
                content = overlap_text + content

            chunk_metadata = dict(metadata) if metadata else {}
            chunk_metadata["chunk_index"] = i
            chunks.append(Chunk(content=content.strip(), metadata=chunk_metadata))

        return [c for c in chunks if len(c.content) > 10]  # 过滤太短的块

    def _split_recursive(self, text: str, seps: List[str]) -> List[str]:
        if len(text) <= self.chunk_size or not seps:
            return [text]

        sep = seps[0]
        parts = text.split(sep)
        result, current = [], ""

        for part in parts:
            if len(current) + len(part) + len(sep) <= self.chunk_size:
                current += (sep if current else "") + part
            else:
                if current:
                    result.append(current)
                if len(part) > self.chunk_size:
                    result.extend(self._split_recursive(part, seps[1:]))
                else:
                    current = part
        if current:
            result.append(current)
        return result


class EmbeddingService:
    """
    Embedding 服务封装。

    支持：
    1. 千问 DashScope API
    2. OpenAI API
    3. 本地模拟（用于测试）
    """

    def __init__(self, provider: str = "mock", model: str = "text-embedding-v4",
                 dimension: int = 1024):
        self.provider = provider
        self.model = model
        self.dimension = dimension
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if self.provider == "dashscope":
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.environ.get("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        elif self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def embed(self, texts: List[str]) -> np.ndarray:
        """将文本列表转为向量"""
        if self.provider == "mock":
            # 模拟：用文本哈希生成伪向量（可复现）
            vectors = []
            for text in texts:
                np.random.seed(hash(text) % 2**31)
                vec = np.random.randn(self.dimension).astype(np.float32)
                vec = vec / np.linalg.norm(vec)  # 归一化
                vectors.append(vec)
            return np.array(vectors)

        client = self._get_client()
        response = client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimension
        )
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """单条文本 Embedding"""
        return self.embed([text])[0]


class VectorStore:
    """
    向量存储（使用 FAISS 或 numpy fallback）
    """

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.chunks: List[Chunk] = []
        self._vectors: Optional[np.ndarray] = None

        try:
            import faiss
            self._index = faiss.IndexFlatL2(dimension)
            self._use_faiss = True
        except ImportError:
            self._index = None
            self._use_faiss = False

    def add(self, chunks: List[Chunk]):
        """添加带 embedding 的 chunks"""
        vectors = np.array([c.embedding for c in chunks], dtype=np.float32)

        if self._use_faiss:
            self._index.add(vectors)
        else:
            if self._vectors is None:
                self._vectors = vectors
            else:
                self._vectors = np.vstack([self._vectors, vectors])

        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """搜索最相似的 chunks"""
        query = query_embedding.reshape(1, -1).astype(np.float32)

        if self._use_faiss:
            distances, indices = self._index.search(query, top_k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(self.chunks):
                    score = 1.0 / (1.0 + float(dist))
                    results.append((self.chunks[idx], score))
        else:
            norms = np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(query_embedding)
            similarities = np.dot(self._vectors, query_embedding) / (norms + 1e-10)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            results = [(self.chunks[idx], float(similarities[idx])) for idx in top_indices]

        return results

    @property
    def count(self) -> int:
        return len(self.chunks)


# ============================================================
# Part 4: Prompt 构造（RAG 的关键技巧）
# ============================================================

def demo_prompt_construction():
    """
    RAG 的 Prompt 构造是影响回答质量的关键因素。

    好的 Prompt = 检索结果 + 用户问题 + 指令
    """
    print("\n" + "=" * 60)
    print("Part 4: Prompt 构造技巧")
    print("=" * 60)

    print("""
    RAG Prompt 的三个部分：

    1. System Prompt（角色 + 规则）
    2. Context（检索到的相关文档块）
    3. User Question（用户的问题）
    """)

    # 好的 RAG Prompt 模板
    good_template = """你是一个知识库问答助手。请基于以下参考资料回答用户的问题。

规则：
1. 只基于参考资料回答，不要使用你自己的知识
2. 如果参考资料中没有相关信息，直接说"根据现有资料，我无法回答这个问题"
3. 回答时引用来源（如"根据《员工手册》第3章"）
4. 保持回答简洁准确

参考资料：
{context}

用户问题：{question}"""

    # 差的 RAG Prompt
    bad_template = """根据以下内容回答问题：

{context}

问题：{question}"""

    print("\n✅ 好的 Prompt 模板：")
    print("─" * 50)
    print(good_template)

    print("\n❌ 差的 Prompt 模板：")
    print("─" * 50)
    print(bad_template)

    print("\n📊 差在哪？")
    print("─" * 50)
    print("""
    好的 Prompt：
    ├── 指定了角色（知识库问答助手）
    ├── 明确了规则（只基于资料、不知道就说不知道、引用来源）
    ├── 结构清晰（参考资料和问题分开）
    └── 有边界约束（不要使用自己的知识）

    差的 Prompt：
    ├── 没有角色定义
    ├── 没有规则约束
    ├── LLM 可能忽略检索结果，用自己的知识回答
    └── LLM 可能编造信息
    """)


def build_rag_prompt(context_chunks: List[Tuple[Chunk, float]],
                     question: str) -> str:
    """
    构造 RAG Prompt。

    关键技巧：
    1. 按相关度排序，最相关的放在最前面
    2. 给每个块加上来源标记
    3. 控制总长度，避免超出 LLM 上下文窗口
    """
    # 按相关度排序
    sorted_chunks = sorted(context_chunks, key=lambda x: x[1], reverse=True)

    # 构造参考资料文本
    context_parts = []
    total_length = 0
    max_context_length = 3000  # 控制上下文长度

    for i, (chunk, score) in enumerate(sorted_chunks):
        source = chunk.metadata.get("source", "未知来源")
        section = chunk.metadata.get("section", "")
        source_tag = f"[来源: {source}" + (f", {section}" if section else "") + "]"

        part = f"{source_tag}\n{chunk.content}"
        if total_length + len(part) > max_context_length:
            break
        context_parts.append(part)
        total_length += len(part)

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""你是一个知识库问答助手。请基于以下参考资料回答用户的问题。

规则：
1. 只基于参考资料回答，不要使用你自己的知识
2. 如果参考资料中没有相关信息，直接说"根据现有资料，我无法回答这个问题"
3. 回答时引用来源（如"根据《员工手册》"）
4. 保持回答简洁准确

参考资料：
{context}

用户问题：{question}"""

    return prompt


# ============================================================
# Part 5: 完整 RAG Pipeline
# ============================================================

class RAGPipeline:
    """
    完整的 RAG Pipeline。

    使用方法：
        rag = RAGPipeline()
        rag.ingest(documents)          # 离线：索引文档
        answer = rag.query("问题")     # 在线：问答
    """

    def __init__(self, embedding_provider: str = "mock",
                 embedding_model: str = "text-embedding-v4",
                 embedding_dimension: int = 1024,
                 chunk_size: int = 300, chunk_overlap: int = 50,
                 top_k: int = 5):

        self.chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.embedder = EmbeddingService(
            provider=embedding_provider,
            model=embedding_model,
            dimension=embedding_dimension
        )
        self.vector_store = VectorStore(dimension=embedding_dimension)
        self.top_k = top_k
        self._ingested = False

    def ingest(self, documents: List[Document]) -> Dict:
        """
        离线阶段：索引文档。

        流程：文档 → 分块 → Embedding → 存入向量数据库
        """
        print(f"📥 开始索引 {len(documents)} 个文档...")
        start_time = time.time()

        all_chunks = []

        # Step 1: 分块
        for doc in documents:
            chunks = self.chunker.chunk(doc.content, metadata=doc.metadata)
            all_chunks.extend(chunks)

        print(f"   分块完成：{len(all_chunks)} 个块")

        # Step 2: Embedding
        texts = [c.content for c in all_chunks]
        embeddings = self.embedder.embed(texts)
        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb

        print(f"   Embedding 完成：{embeddings.shape}")

        # Step 3: 存入向量数据库
        self.vector_store.add(all_chunks)
        self._ingested = True

        elapsed = time.time() - start_time
        stats = {
            "documents": len(documents),
            "chunks": len(all_chunks),
            "elapsed_seconds": round(elapsed, 2),
            "total_vectors": self.vector_store.count,
        }
        print(f"   ✅ 索引完成：{stats}")
        return stats

    def query(self, question: str, return_sources: bool = False) -> Dict:
        """
        在线阶段：检索 + 生成。

        流程：问题 → Embedding → 向量检索 → 构造 Prompt → LLM 生成
        """
        if not self._ingested:
            return {"answer": "请先调用 ingest() 索引文档", "sources": []}

        # Step 1: 问题 Embedding
        query_embedding = self.embedder.embed_single(question)

        # Step 2: 向量检索
        search_results = self.vector_store.search(query_embedding, top_k=self.top_k)

        # Step 3: 构造 Prompt
        prompt = build_rag_prompt(search_results, question)

        # Step 4: LLM 生成（这里模拟，实际接 LLM API）
        answer = self._generate(prompt)

        result = {
            "answer": answer,
            "sources": [
                {
                    "content": chunk.content[:100] + "...",
                    "score": round(score, 4),
                    "metadata": chunk.metadata,
                }
                for chunk, score in search_results
            ],
        }

        return result

    def _generate(self, prompt: str) -> str:
        """
        调用 LLM 生成回答。

        这里是模拟实现，实际项目中替换为真实 LLM API 调用。
        """
        # 检查是否有 DashScope API
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
                response = client.chat.completions.create(
                    model="qwen-plus",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"[LLM 调用失败: {e}]\n\n基于检索结果的回答：\n{prompt[-500:]}"

        # 无 API 时的模拟回答
        return "[模拟回答 - 设置 DASHSCOPE_API_KEY 环境变量后可调用真实 LLM]\n\n" + \
               "基于检索到的参考资料，相关内容如下：\n" + \
               prompt[prompt.find("参考资料："):prompt.find("用户问题：")][:300]


# ============================================================
# Part 6: 端到端演示
# ============================================================

def demo_end_to_end():
    """完整的 RAG 端到端演示"""
    print("\n" + "=" * 60)
    print("Part 5: 端到端 RAG 演示")
    print("=" * 60)

    # 模拟知识库文档
    documents = [
        Document(
            content="""Python 入门指南

Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。
Python 的设计哲学强调代码的可读性和简洁性。

安装 Python：
1. 访问 python.org 下载最新版本
2. 运行安装程序，勾选"Add Python to PATH"
3. 打开终端，输入 python --version 验证安装

Python 的基本数据类型包括：
- 整数（int）：如 42, -1, 0
- 浮点数（float）：如 3.14, -0.5
- 字符串（str）：如 "hello", 'world'
- 布尔值（bool）：True 或 False
- 列表（list）：如 [1, 2, 3]
- 字典（dict）：如 {"name": "Python", "version": 3.12}""",
            metadata={"source": "python-guide.md", "section": "入门", "language": "zh"}
        ),
        Document(
            content="""Python 进阶：函数和模块

函数是 Python 中组织代码的基本单位。

定义函数：
def greet(name):
    return f"Hello, {name}!"

函数参数：
- 位置参数：def func(a, b)
- 关键字参数：def func(a=1, b=2)
- 可变参数：def func(*args, **kwargs)

模块是 Python 代码的组织方式。
使用 import 语句导入模块：
import math
from os import path
import numpy as np  # 第三方库

虚拟环境：
python -m venv myenv      # 创建
source myenv/bin/activate  # 激活（Linux/Mac）
myenv\\Scripts\\activate     # 激活（Windows）
pip install requests       # 安装包""",
            metadata={"source": "python-guide.md", "section": "进阶", "language": "zh"}
        ),
        Document(
            content="""Git 版本控制入门

Git 是最流行的分布式版本控制系统。

基本概念：
- 仓库（Repository）：项目的历史记录
- 提交（Commit）：一次保存的快照
- 分支（Branch）：独立的开发线
- 合并（Merge）：将分支合并到主线

常用命令：
git init              # 初始化仓库
git add .             # 暂存所有文件
git commit -m "msg"   # 提交
git push              # 推送到远程
git pull              # 拉取远程更新
git branch feature    # 创建分支
git checkout feature  # 切换分支
git merge feature     # 合并分支""",
            metadata={"source": "git-guide.md", "section": "基础", "language": "zh"}
        ),
        Document(
            content="""机器学习基础

机器学习是人工智能的子领域，让计算机从数据中学习规律。

三种主要类型：
1. 监督学习：有标签数据，学习输入到输出的映射
   - 分类：预测类别（如垃圾邮件检测）
   - 回归：预测数值（如房价预测）

2. 无监督学习：无标签数据，发现数据中的模式
   - 聚类：将相似数据分组
   - 降维：减少特征数量

3. 强化学习：通过与环境交互学习最优策略
   - 应用：游戏 AI、机器人控制

常用 Python 库：
- scikit-learn：传统机器学习算法
- PyTorch：深度学习框架
- TensorFlow：另一个深度学习框架
- Pandas：数据处理
- NumPy：数值计算""",
            metadata={"source": "ml-basics.md", "section": "概述", "language": "zh"}
        ),
    ]

    # 创建 RAG Pipeline
    rag = RAGPipeline(
        embedding_provider="mock",  # 用模拟 Embedding，无需 API
        embedding_dimension=256,    # 模拟用 256 维
        chunk_size=200,
        chunk_overlap=40,
        top_k=3,
    )

    # 索引文档
    stats = rag.ingest(documents)

    # 查询
    questions = [
        "Python 怎么安装？",
        "Git 怎么创建分支？",
        "机器学习有哪些类型？",
        "Python 的虚拟环境怎么用？",
    ]

    for q in questions:
        print(f"\n{'─' * 50}")
        print(f"❓ 问题：{q}")
        print(f"{'─' * 50}")
        result = rag.query(q)
        print(f"💡 回答：{result['answer'][:200]}...")
        print(f"📚 参考来源：")
        for src in result["sources"][:2]:
            print(f"   - [{src['score']}] {src['metadata'].get('source', '?')} | {src['content'][:50]}...")


# ============================================================
# Part 7: RAG 常见问题和优化
# ============================================================

def demo_rag_pitfalls():
    """RAG 系统的常见问题和优化方向"""
    print("\n" + "=" * 60)
    print("Part 6: RAG 常见问题和优化方向")
    print("=" * 60)

    pitfalls = [
        {
            "problem": "检索到了但 LLM 没用",
            "cause": "检索结果中混入了噪声，LLM 被干扰",
            "solution": "提高 top_k 的精度（用 Reranker 二次排序）、减少 top_k 数量",
        },
        {
            "problem": "LLM 编造了检索结果中没有的信息",
            "cause": "Prompt 没有约束'只基于资料回答'",
            "solution": "在 System Prompt 中明确要求'只基于参考资料回答，不确定就说不知道'",
        },
        {
            "problem": "回答不够详细",
            "cause": "检索到的块太小，信息不完整",
            "solution": "增大 chunk_size，或用 Parent-Child 策略（小块检索、大块送 LLM）",
        },
        {
            "problem": "检索不到相关内容",
            "cause": "分块切断了关键信息，或 Embedding 质量差",
            "solution": "调整分块策略（增加 overlap）、换更好的 Embedding 模型",
        },
        {
            "problem": "多轮对话效果差",
            "cause": "每轮独立检索，没有对话历史上下文",
            "solution": "将对话历史合并后重新构造检索 query（Query Rewriting）",
        },
        {
            "problem": "响应速度慢",
            "cause": "Embedding API 延迟 + 向量检索慢 + LLM 生成慢",
            "solution": "缓存 Embedding、用更快的索引（HNSW）、流式输出",
        },
    ]

    for p in pitfalls:
        print(f"\n🔴 问题：{p['problem']}")
        print(f"   原因：{p['cause']}")
        print(f"   解决：{p['solution']}")


# ============================================================
# 主函数
# ============================================================

def main():
    """运行所有演示"""
    demo_what_is_rag()
    demo_rag_architecture()
    demo_prompt_construction()
    demo_end_to_end()
    demo_rag_pitfalls()

    print("\n" + "=" * 60)
    print("📝 本课要点总结")
    print("=" * 60)
    print("""
    1. RAG = 检索 + 生成，让 LLM 能使用私有知识
    2. 离线阶段：文档 → 分块 → Embedding → 向量数据库
    3. 在线阶段：问题 → Embedding → 检索 → 构造 Prompt → LLM 生成
    4. Prompt 构造是关键：角色 + 规则 + 检索结果 + 问题
    5. 约束 LLM "只基于资料回答"，避免编造
    6. 常见优化：Reranker、Query Rewriting、Parent-Child 策略
    """)


if __name__ == "__main__":
    main()

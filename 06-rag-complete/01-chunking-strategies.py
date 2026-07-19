#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第25课：文本分块策略 (Chunking Strategies)
==========================================

本课程讲解 RAG 系统中最关键的预处理环节——如何把长文档切成合适的块。
分块质量直接决定检索质量，进而决定 LLM 回答的质量。

学习目标：
1. 理解为什么分块是 RAG 质量的关键
2. 掌握五种主流分块策略及其适用场景
3. 理解分块大小和重叠对检索效果的影响
4. 能根据文档类型选择合适的分块策略

"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================
# Part 1: 为什么分块重要？
# ============================================================

@dataclass
class Chunk:
    """文档块的数据结构"""
    content: str           # 块的文本内容
    metadata: Dict         # 元数据（来源、页码、章节等）
    index: int = 0         # 块在原文中的序号
    token_count: int = 0   # token 数量（估算）


def demo_why_chunking_matters():
    """
    演示分块质量对检索的影响。

    核心问题：Embedding 模型有 token 上限（通常 512-8192），
    一篇长文档不可能直接塞进去。必须切成块。
    """
    print("=" * 60)
    print("Part 1: 为什么分块重要？")
    print("=" * 60)

    # 模拟一篇长文档
    long_document = """
    Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。
    Python 的设计哲学强调代码的可读性和简洁性。
    Python 支持多种编程范式，包括面向对象、函数式和过程式编程。

    Python 的主要特点包括：
    1. 简洁的语法，使用缩进表示代码块
    2. 动态类型系统
    3. 自动内存管理（垃圾回收）
    4. 丰富的标准库
    5. 跨平台兼容性

    Python 在以下领域广泛应用：
    - Web 开发（Django, Flask）
    - 数据科学（NumPy, Pandas）
    - 机器学习（PyTorch, TensorFlow）
    - 自动化脚本
    - 科学计算

    Python 的包管理器 pip 是安装第三方库的主要工具。
    虚拟环境（venv）用于隔离不同项目的依赖。
    """

    print("\n📄 原始文档长度:", len(long_document), "字符")

    # 差的分块：按固定字符数切割，不考虑语义
    bad_chunks = [long_document[i:i+80] for i in range(0, len(long_document), 80)]
    print(f"\n❌ 差的分块（按 80 字符硬切）：共 {len(bad_chunks)} 块")
    for i, chunk in enumerate(bad_chunks[:3]):
        preview = chunk.replace("\n", " ").strip()[:60]
        print(f"   块 {i+1}: \"{preview}...\"")
    print("   问题：一句话被切成两半，语义不完整，Embedding 质量差")

    # 好的分块：按段落切分
    good_chunks = [p.strip() for p in long_document.split("\n\n") if p.strip()]
    print(f"\n✅ 好的分块（按段落切分）：共 {len(good_chunks)} 块")
    for i, chunk in enumerate(good_chunks[:3]):
        preview = chunk.replace("\n", " ").strip()[:60]
        print(f"   块 {i+1}: \"{preview}...\"")
    print("   优点：每块是一个完整的语义单元")


# ============================================================
# Part 2: 五种分块策略
# ============================================================

class ChunkingStrategies:
    """五种主流分块策略的实现"""

    # ---- 策略一：固定大小分块 ----
    @staticmethod
    def fixed_size(text: str, chunk_size: int = 200, overlap: int = 50) -> List[Chunk]:
        """
        固定大小分块：按字符数切割，带重叠。

        优点：实现简单，块大小可控
        缺点：可能切断句子或段落
        适用：对语义完整性要求不高的场景（如日志分析）
        """
        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = start + chunk_size
            content = text[start:end]

            # 尝试在句子边界切割（不是必须，但能提高质量）
            if end < len(text):
                # 找最后一个句号/换行作为切割点
                last_period = max(content.rfind("。"), content.rfind("."),
                                  content.rfind("\n"))
                if last_period > chunk_size * 0.5:  # 至少保留一半内容
                    content = content[:last_period + 1]
                    end = start + len(content)

            chunks.append(Chunk(
                content=content.strip(),
                metadata={"strategy": "fixed_size", "start": start},
                index=index
            ))
            index += 1
            start = end - overlap  # 重叠部分

        return chunks

    # ---- 策略二：按句子分块 ----
    @staticmethod
    def sentence_based(text: str, max_sentences: int = 5) -> List[Chunk]:
        """
        按句子分块：每块包含 N 个句子。

        优点：不会切断句子，语义相对完整
        缺点：块大小不均匀（句子有长有短）
        适用：文章、报告等结构化文本
        """
        # 中英文句子切分
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunk_sentences = sentences[i:i + max_sentences]
            chunks.append(Chunk(
                content=" ".join(chunk_sentences),
                metadata={"strategy": "sentence_based", "sentence_range": f"{i}-{i+len(chunk_sentences)}"},
                index=i // max_sentences
            ))

        return chunks

    # ---- 策略三：按段落分块 ----
    @staticmethod
    def paragraph_based(text: str, max_chunk_size: int = 500) -> List[Chunk]:
        """
        按段落分块：每个段落是一个块，过长的段落再细分。

        优点：保留段落语义完整性
        缺点：块大小差异大
        适用：结构清晰的文档（技术文档、论文）
        """
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        for i, para in enumerate(paragraphs):
            if len(para) <= max_chunk_size:
                chunks.append(Chunk(
                    content=para,
                    metadata={"strategy": "paragraph_based", "paragraph_index": i},
                    index=i
                ))
            else:
                # 段落太长，按句子再切
                sub_chunks = ChunkingStrategies.sentence_based(para, max_sentences=3)
                for j, sub in enumerate(sub_chunks):
                    sub.metadata["strategy"] = "paragraph_based"
                    sub.metadata["parent_paragraph"] = i
                    sub.index = i * 100 + j
                    chunks.append(sub)

        return chunks

    # ---- 策略四：递归分块（最常用） ----
    @staticmethod
    def recursive(text: str, chunk_size: int = 300, overlap: int = 50,
                  separators: Optional[List[str]] = None) -> List[Chunk]:
        """
        递归分块：先按大分隔符切，切不动就换小分隔符。

        优点：尽可能保留语义完整性，块大小相对可控
        缺点：实现稍复杂
        适用：通用场景（LangChain 默认策略）

        这是 RAG 系统中最推荐的默认策略。
        """
        if separators is None:
            separators = ["\n\n", "\n", "。", ".", "，", ",", " "]

        def _split_recursive(text: str, seps: List[str]) -> List[str]:
            if len(text) <= chunk_size:
                return [text]

            if not seps:
                return [text]

            sep = seps[0]
            parts = text.split(sep)

            result = []
            current = ""

            for part in parts:
                if len(current) + len(part) + len(sep) <= chunk_size:
                    current += (sep if current else "") + part
                else:
                    if current:
                        result.append(current)
                    # 如果单个 part 还是太长，递归用下一个分隔符
                    if len(part) > chunk_size:
                        result.extend(_split_recursive(part, seps[1:]))
                    else:
                        current = part

            if current:
                result.append(current)

            return result

        raw_chunks = _split_recursive(text, separators)

        # 添加重叠
        chunks = []
        for i, content in enumerate(raw_chunks):
            # 从前一块取 overlap 部分
            if i > 0 and overlap > 0:
                prev = raw_chunks[i - 1]
                overlap_text = prev[-overlap:] if len(prev) > overlap else prev
                content = overlap_text + content

            chunks.append(Chunk(
                content=content.strip(),
                metadata={"strategy": "recursive", "separator_used": separators[0]},
                index=i
            ))

        return chunks

    # ---- 策略五：语义分块 ----
    @staticmethod
    def semantic(text: str, embedding_func=None, threshold: float = 0.5,
                 max_chunk_size: int = 500) -> List[Chunk]:
        """
        语义分块：根据相邻句子的语义相似度决定是否切分。

        优点：切分点在语义转折处，块内语义最连贯
        缺点：需要调用 Embedding API，成本高、速度慢
        适用：对检索质量要求极高的场景

        原理：
        1. 把文本按句子切分
        2. 计算相邻句子的 Embedding 相似度
        3. 相似度低于阈值的地方 = 语义转折点 = 切分点
        """
        # 按句子切分
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        # 如果没有 embedding_func，用模拟的相似度
        if embedding_func is None:
            print("   ⚠️ 未提供 embedding_func，使用模拟相似度演示")
            # 模拟：相邻句子如果主题词不同，相似度低
            similarities = []
            for i in range(len(sentences) - 1):
                # 简单模拟：共同字符比例作为"相似度"
                s1_chars = set(sentences[i])
                s2_chars = set(sentences[i + 1])
                if len(s1_chars | s2_chars) > 0:
                    sim = len(s1_chars & s2_chars) / len(s1_chars | s2_chars)
                else:
                    sim = 0
                similarities.append(sim)
        else:
            # 真实的语义分块
            embeddings = [embedding_func(s) for s in sentences]
            from numpy import dot
            from numpy.linalg import norm
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = dot(embeddings[i], embeddings[i + 1]) / (
                    norm(embeddings[i]) * norm(embeddings[i + 1]) + 1e-10
                )
                similarities.append(sim)

        # 在相似度低于阈值的地方切分
        chunks = []
        current_sentences = [sentences[0]]

        for i, sim in enumerate(similarities):
            current_text = " ".join(current_sentences + [sentences[i + 1]])

            if sim < threshold or len(current_text) > max_chunk_size:
                # 切分点
                chunks.append(Chunk(
                    content=" ".join(current_sentences),
                    metadata={"strategy": "semantic", "split_similarity": sim if i < len(similarities) else None},
                    index=len(chunks)
                ))
                current_sentences = [sentences[i + 1]]
            else:
                current_sentences.append(sentences[i + 1])

        # 最后一块
        if current_sentences:
            chunks.append(Chunk(
                content=" ".join(current_sentences),
                metadata={"strategy": "semantic"},
                index=len(chunks)
            ))

        return chunks


def demo_all_strategies():
    """对比演示五种分块策略"""
    print("\n" + "=" * 60)
    print("Part 2: 五种分块策略对比")
    print("=" * 60)

    sample_text = """
    Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。
    Python 的设计哲学强调代码的可读性和简洁性。
    Python 支持多种编程范式，包括面向对象、函数式和过程式编程。

    Python 的主要特点包括简洁的语法，使用缩进表示代码块。
    它采用动态类型系统和自动内存管理。
    Python 拥有丰富的标准库和跨平台兼容性。

    Python 在 Web 开发领域广泛应用，主流框架包括 Django 和 Flask。
    在数据科学领域，NumPy 和 Pandas 是核心库。
    机器学习领域主要使用 PyTorch 和 TensorFlow。

    Java 是另一种流行的编程语言，由 Sun Microsystems 于 1995 年发布。
    Java 的设计理念是"一次编写，到处运行"。
    Java 广泛应用于企业级应用开发和 Android 应用开发。

    JavaScript 是 Web 前端开发的核心语言。
    Node.js 让 JavaScript 也可以用于后端开发。
    React、Vue、Angular 是主流的前端框架。
    """

    strategies = [
        ("固定大小", lambda t: ChunkingStrategies.fixed_size(t, chunk_size=150, overlap=30)),
        ("按句子", lambda t: ChunkingStrategies.sentence_based(t, max_sentences=3)),
        ("按段落", lambda t: ChunkingStrategies.paragraph_based(t, max_chunk_size=200)),
        ("递归分块", lambda t: ChunkingStrategies.recursive(t, chunk_size=200, overlap=30)),
        ("语义分块", lambda t: ChunkingStrategies.semantic(t, threshold=0.4)),
    ]

    for name, strategy in strategies:
        print(f"\n{'─' * 50}")
        print(f"📋 策略：{name}")
        print(f"{'─' * 50}")

        chunks = strategy(sample_text)
        print(f"   生成 {len(chunks)} 个块：")
        for i, chunk in enumerate(chunks):
            preview = chunk.content.replace("\n", " ").strip()[:50]
            print(f"   [{i+1}] ({len(chunk.content)}字) \"{preview}...\"")


# ============================================================
# Part 3: 分块大小和重叠的影响
# ============================================================

def demo_chunk_size_impact():
    """
    演示分块大小对检索质量的影响。

    太小：语义不完整，Embedding 质量差
    太大：包含太多无关信息，检索精度低
    刚好：语义完整且聚焦

    经验值：
    - 200-500 tokens：通用推荐
    - 100-200 tokens：精确检索（FAQ、代码片段）
    - 500-1000 tokens：长文档理解（论文、报告）
    """
    print("\n" + "=" * 60)
    print("Part 3: 分块大小和重叠的影响")
    print("=" * 60)

    print("\n📊 分块大小选择指南：")
    print("─" * 50)

    guide = [
        ("100-200 tokens", "精确检索", "FAQ、代码片段、短句匹配", "精度高，召回低"),
        ("200-500 tokens", "通用推荐", "大多数 RAG 场景", "精度和召回平衡"),
        ("500-1000 tokens", "长文档理解", "论文、报告、技术文档", "召回高，精度稍低"),
        ("> 1000 tokens", "不推荐", "—", "噪声太多，检索质量下降"),
    ]

    print(f"{'大小':<18} {'适用场景':<12} {'典型用途':<28} {'特点'}")
    print("─" * 75)
    for size, scene, usage, feature in guide:
        print(f"{size:<18} {scene:<12} {usage:<28} {feature}")

    print("\n📊 重叠（Overlap）的作用：")
    print("─" * 50)
    print("""
    无重叠：
    [块1: ...介绍了Python的语法]  [块2: 非常简洁，使用缩进...]
                                      ↑ 上下文丢失，"非常简洁"指什么？

    有重叠（overlap=50字符）：
    [块1: ...介绍了Python的语法]  [块2: Python的语法非常简洁，使用缩进...]
                                      ↑ 有上下文，知道"语法"指的是 Python

    推荐重叠比例：块大小的 10%-20%
    - 块大小 300 tokens → 重叠 30-60 tokens
    """)


# ============================================================
# Part 4: 按文档类型选择策略
# ============================================================

def demo_strategy_by_doc_type():
    """
    不同文档类型适合不同的分块策略。
    """
    print("\n" + "=" * 60)
    print("Part 4: 按文档类型选择分块策略")
    print("=" * 60)

    recommendations = [
        {
            "doc_type": "技术文档（API 文档、README）",
            "best_strategy": "按段落 / 递归",
            "reason": "结构清晰，每个段落通常是一个独立的功能说明",
            "chunk_size": "200-400 tokens",
        },
        {
            "doc_type": "学术论文",
            "best_strategy": "递归 / 按段落",
            "reason": "有明确的章节结构，按段落保留论点完整性",
            "chunk_size": "300-600 tokens",
        },
        {
            "doc_type": "FAQ / 客服问答",
            "best_strategy": "按句子 / 固定大小",
            "reason": "问答对通常较短，每对是一个完整的知识单元",
            "chunk_size": "100-200 tokens",
        },
        {
            "doc_type": "新闻/博客",
            "best_strategy": "语义分块 / 按句子",
            "reason": "话题可能转换，语义分块能捕捉转折点",
            "chunk_size": "200-400 tokens",
        },
        {
            "doc_type": "代码文件",
            "best_strategy": "按函数/类分块",
            "reason": "代码有天然的结构（函数、类、方法），按结构切分",
            "chunk_size": "按结构，不按大小",
        },
        {
            "doc_type": "对话/聊天记录",
            "best_strategy": "按轮次分块",
            "reason": "每轮对话是一个完整的交互单元",
            "chunk_size": "按轮次，不按大小",
        },
    ]

    for rec in recommendations:
        print(f"\n📄 {rec['doc_type']}")
        print(f"   推荐策略：{rec['best_strategy']}")
        print(f"   原因：{rec['reason']}")
        print(f"   建议大小：{rec['chunk_size']}")


# ============================================================
# Part 5: 元数据增强
# ============================================================

def demo_metadata_enrichment():
    """
    分块时附加元数据，能大幅提升检索的可过滤性和可追溯性。

    常见元数据：
    - source: 文档来源（文件名、URL）
    - page: 页码
    - section: 章节标题
    - chunk_type: 块类型（text/table/code/image_caption）
    - created_at: 创建时间
    - author: 作者
    """
    print("\n" + "=" * 60)
    print("Part 5: 元数据增强（Metadata Enrichment）")
    print("=" * 60)

    print("\n附加元数据的块示例：")
    example_chunk = Chunk(
        content="Python 的设计哲学强调代码的可读性和简洁性。",
        metadata={
            "source": "python-intro.md",
            "section": "1. 简介",
            "page": 1,
            "chunk_type": "text",
            "language": "zh",
            "token_count": 20,
        },
        index=3
    )

    print(f"   内容: {example_chunk.content}")
    print(f"   元数据: {example_chunk.metadata}")

    print("\n元数据的价值：")
    print("   1. 来源追溯：回答时可以引用原文出处")
    print("   2. 过滤检索：'只在技术文档中搜索'、'只搜索最近一周的'")
    print("   3. 权重调整：官方文档的权重 > 博客的权重")
    print("   4. 去重：相同内容来自不同来源时，选择更权威的")


# ============================================================
# Part 6: 完整的分块流水线
# ============================================================

class ChunkingPipeline:
    """
    生产级分块流水线：文档 → 清洗 → 分块 → 元数据 → 输出

    这是一个完整的分块流程，可以直接用于 RAG 项目。
    """

    def __init__(self, strategy: str = "recursive", chunk_size: int = 300,
                 overlap: int = 50):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stats = {"total_docs": 0, "total_chunks": 0, "avg_chunk_size": 0}

    def clean_text(self, text: str) -> str:
        """文本清洗：去除多余空白、特殊字符等"""
        text = re.sub(r'\n{3,}', '\n\n', text)      # 多个换行合并
        text = re.sub(r' {2,}', ' ', text)           # 多个空格合并
        text = re.sub(r'[^\S\n]+', ' ', text)        # 保留换行，清理其他空白
        return text.strip()

    def chunk_document(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """完整的分块流程"""
        # 1. 清洗
        cleaned = self.clean_text(text)

        # 2. 分块
        strategy_map = {
            "fixed_size": ChunkingStrategies.fixed_size,
            "sentence": ChunkingStrategies.sentence_based,
            "paragraph": ChunkingStrategies.paragraph_based,
            "recursive": ChunkingStrategies.recursive,
            "semantic": ChunkingStrategies.semantic,
        }

        strategy_func = strategy_map.get(self.strategy, ChunkingStrategies.recursive)

        if self.strategy == "recursive":
            chunks = strategy_func(cleaned, chunk_size=self.chunk_size, overlap=self.overlap)
        elif self.strategy == "fixed_size":
            chunks = strategy_func(cleaned, chunk_size=self.chunk_size, overlap=self.overlap)
        else:
            chunks = strategy_func(cleaned)

        # 3. 附加元数据
        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)

        # 4. 估算 token 数（中文约 1.5 字/token，英文约 4 字符/token）
        for chunk in chunks:
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', chunk.content))
            english_chars = len(chunk.content) - chinese_chars
            chunk.token_count = int(chinese_chars / 1.5 + english_chars / 4)

        # 5. 更新统计
        self.stats["total_docs"] += 1
        self.stats["total_chunks"] += len(chunks)
        if chunks:
            self.stats["avg_chunk_size"] = int(
                sum(c.token_count for c in chunks) / len(chunks)
            )

        return chunks

    def get_stats(self) -> Dict:
        """返回分块统计信息"""
        return self.stats.copy()


def demo_pipeline():
    """演示完整的分块流水线"""
    print("\n" + "=" * 60)
    print("Part 6: 完整的分块流水线")
    print("=" * 60)

    pipeline = ChunkingPipeline(strategy="recursive", chunk_size=200, overlap=40)

    # 模拟多个文档
    documents = [
        ("python-intro.md", "Python 是一种高级编程语言...\n\nPython 的特点包括..."),
        ("api-guide.md", "GET /api/users\n返回用户列表...\n\nPOST /api/users\n创建新用户..."),
    ]

    for filename, content in documents:
        chunks = pipeline.chunk_document(
            content,
            metadata={"source": filename, "language": "zh"}
        )
        print(f"\n📄 {filename}: {len(chunks)} 个块")
        for chunk in chunks:
            print(f"   [{chunk.index}] tokens≈{chunk.token_count} | {chunk.content[:40]}...")

    print(f"\n📊 总统计: {pipeline.get_stats()}")


# ============================================================
# 主函数
# ============================================================

def main():
    """运行所有演示"""
    demo_why_chunking_matters()
    demo_all_strategies()
    demo_chunk_size_impact()
    demo_strategy_by_doc_type()
    demo_metadata_enrichment()
    demo_pipeline()

    print("\n" + "=" * 60)
    print("📝 本课要点总结")
    print("=" * 60)
    print("""
    1. 分块质量直接决定 RAG 检索质量
    2. 五种策略：固定大小 → 按句子 → 按段落 → 递归（推荐） → 语义
    3. 推荐块大小：200-500 tokens，重叠 10%-20%
    4. 按文档类型选择策略（技术文档用递归，FAQ 用句子，代码按结构）
    5. 元数据增强能提升检索的可过滤性和可追溯性
    6. 生产环境用 ChunkingPipeline 统一管理分块流程
    """)


if __name__ == "__main__":
    main()

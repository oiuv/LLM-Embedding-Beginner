#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第30课：检索评估指标
====================

本课程讲解如何用数字衡量检索系统的好坏。
没有指标，你就不知道改了分块策略或换了模型后，效果是变好了还是变差了。

学习目标：
1. 掌握三个核心检索指标：Recall@K、Precision@K、MRR
2. 理解 NDCG（排序质量指标）
3. 能构建一套完整的检索评估流程
4. 知道怎么用评估驱动 RAG 系统优化

"""

import numpy as np
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, field


# ============================================================
# Part 1: 为什么需要评估指标？
# ============================================================

def demo_why_metrics():
    """
    没有指标，优化就是瞎猜。
    """
    print("=" * 60)
    print("Part 1: 为什么需要评估指标？")
    print("=" * 60)

    print("""
    场景：你优化了 RAG 系统的分块策略，想知道效果好不好。

    没有指标时：
    ├── 随便问几个问题，"感觉"回答好了一点
    ├── 但不确定是真的好了，还是这几个问题碰巧好了
    └── 换一批问题又"感觉"差了

    有指标时：
    ├── 用 100 个标准问答对测试
    ├── Recall@5 从 0.72 提升到 0.81 → 确认变好了
    ├── MRR 从 0.58 提升到 0.67 → 排序质量也提升了
    └── 量化结论：新策略确实更好，提升 12.5%
    """)

    print("""
    评估的三个层次：

    ┌──────────────────────────────────────────────┐
    │  层次 1: 检索质量                              │
    │  "检索到的文档对不对？"                        │
    │  指标：Recall@K, Precision@K, MRR, NDCG       │
    ├──────────────────────────────────────────────┤
    │  层次 2: 生成质量                              │
    │  "LLM 基于检索结果的回答好不好？"              │
    │  指标：准确性、完整性、忠实度                   │
    ├──────────────────────────────────────────────┤
    │  层次 3: 端到端质量                            │
    │  "用户的最终体验好不好？"                      │
    │  指标：用户满意度、采纳率                       │
    └──────────────────────────────────────────────┘

    本课聚焦层次 1（检索质量），这是 RAG 系统的基础。
    检索不好，后面的生成再好也没用（垃圾进，垃圾出）。
    """)


# ============================================================
# Part 2: 构建评估数据集
# ============================================================

@dataclass
class EvalQuery:
    """评估用的查询"""
    query: str                          # 用户查询
    relevant_doc_ids: Set[int]          # 相关文档的 ID 集合（人工标注）
    metadata: Dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """检索结果"""
    doc_id: int
    score: float


def build_eval_dataset() -> Tuple[List[str], List[EvalQuery]]:
    """
    构建评估数据集。

    评估数据集 = 文档库 + 标注好的查询-相关文档对

    标注方式：
    1. 人工标注（金标准，但慢）
    2. LLM 辅助标注（用 LLM 判断相关性，再人工校验）
    3. 用户行为日志（用户点击了哪些结果）
    """
    # 文档库
    documents = [
        "Python 是一种高级编程语言，广泛用于数据科学",       # 0
        "JavaScript 是 Web 前端开发的核心语言",              # 1
        "Python 的虚拟环境 venv 用于隔离项目依赖",           # 2
        "Git 是分布式版本控制系统",                          # 3
        "Docker 容器化部署提高应用可移植性",                  # 4
        "机器学习需要大量数据训练模型",                       # 5
        "PyTorch 是主流的深度学习框架",                      # 6
        "RESTful API 使用 HTTP 方法进行资源操作",            # 7
        "SQL 是关系数据库的查询语言",                        # 8
        "NoSQL 数据库适合非结构化数据存储",                  # 9
    ]

    # 标注好的查询和相关文档
    eval_queries = [
        EvalQuery(
            query="Python 怎么创建虚拟环境？",
            relevant_doc_ids={0, 2},          # 文档 0 和 2 相关
            metadata={"difficulty": "easy"}
        ),
        EvalQuery(
            query="深度学习用什么框架？",
            relevant_doc_ids={5, 6},           # 文档 5 和 6 相关
            metadata={"difficulty": "easy"}
        ),
        EvalQuery(
            query="Web 开发技术栈",
            relevant_doc_ids={1, 7},           # 文档 1 和 7 相关
            metadata={"difficulty": "medium"}
        ),
        EvalQuery(
            query="数据库有哪些类型？",
            relevant_doc_ids={8, 9},           # 文档 8 和 9 相关
            metadata={"difficulty": "easy"}
        ),
        EvalQuery(
            query="代码版本管理和部署",
            relevant_doc_ids={3, 4},           # 文档 3 和 4 相关
            metadata={"difficulty": "medium"}
        ),
        EvalQuery(
            query="Python 数据科学生态",
            relevant_doc_ids={0, 5, 6},        # 文档 0、5、6 相关
            metadata={"difficulty": "hard"}
        ),
    ]

    return documents, eval_queries


# ============================================================
# Part 3: 核心指标实现
# ============================================================

class RetrievalMetrics:
    """
    检索评估指标集合。

    包含：
    - Recall@K：查全率
    - Precision@K：查准率
    - MRR：平均倒排名
    - NDCG@K：归一化折损累计增益
    - Hit Rate@K：命中率
    """

    @staticmethod
    def recall_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
        """
        Recall@K（查全率）：前 K 个结果中，命中了多少相关文档？

        公式：|检索到的相关文档| / |所有相关文档|

        含义：在 K 次机会中，找到了多少"正确答案"
        这是 RAG 最核心的指标——如果相关文档没被检索到，LLM 就无法基于它回答。

        范围：0-1，越高越好
        """
        retrieved_at_k = set(retrieved_ids[:k])
        if not relevant_ids:
            return 1.0  # 没有相关文档时，定义为 1
        hits = retrieved_at_k & relevant_ids
        return len(hits) / len(relevant_ids)

    @staticmethod
    def precision_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
        """
        Precision@K（查准率）：前 K 个结果中，有多少是相关的？

        公式：|检索到的相关文档| / K

        含义：检索结果的"纯度"——是不是大部分结果都有用
        范围：0-1，越高越好
        """
        retrieved_at_k = set(retrieved_ids[:k])
        if k == 0:
            return 0.0
        hits = retrieved_at_k & relevant_ids
        return len(hits) / k

    @staticmethod
    def mrr(retrieved_ids: List[int], relevant_ids: Set[int]) -> float:
        """
        MRR（Mean Reciprocal Rank，平均倒排名）

        公式：1 / rank_of_first_relevant_result

        含义：第一个相关结果排在第几？越靠前越好。
        适合：关注"第一个正确答案在哪"的场景（如问答系统）

        范围：0-1，越高越好
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0  # 没有找到任何相关文档

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[int], relevance_scores: Dict[int, float],
                  k: int) -> float:
        """
        NDCG@K（Normalized Discounted Cumulative Gain）

        与 Recall/Precision 的区别：
        - Recall/Precision 只关心"有没有命中"（二元：命中=1，没命中=0）
        - NDCG 还关心"排在第几位"和"相关度有多高"

        公式：
        DCG@K = Σ (2^relevance_i - 1) / log2(i + 1)
        NDCG@K = DCG@K / IDCG@K  （IDCG 是理想排序的 DCG）

        含义：排序质量，考虑了位置衰减（排在后面的价值更低）
        范围：0-1，越高越好

        适合：相关度有等级差异的场景（如搜索结果有"高度相关"、"一般相关"、"不相关"）
        """
        # 计算 DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            rel = relevance_scores.get(doc_id, 0)
            dcg += (2 ** rel - 1) / np.log2(i + 2)  # i+2 因为 log2(1)=0

        # 计算 IDCG（理想排序的 DCG）
        ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels):
            idcg += (2 ** rel - 1) / np.log2(i + 2)

        if idcg == 0:
            return 1.0  # 没有相关文档时，定义为 1
        return dcg / idcg

    @staticmethod
    def hit_rate_at_k(retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
        """
        Hit Rate@K（命中率）：前 K 个结果中是否包含至少一个相关文档？

        与 Recall 的区别：
        - Recall 关心"命中了多少"
        - Hit Rate 只关心"有没有命中"（有就是 1，没有就是 0）

        适合：只需要找到一个正确答案的场景
        """
        retrieved_at_k = set(retrieved_ids[:k])
        return 1.0 if (retrieved_at_k & relevant_ids) else 0.0


# ============================================================
# Part 4: 指标演示和解读
# ============================================================

def demo_metrics():
    """用具体例子演示每个指标"""
    print("\n" + "=" * 60)
    print("Part 2: 核心指标详解")
    print("=" * 60)

    # 场景：用户问"Python 怎么创建虚拟环境？"
    # 相关文档：{0, 2}（Python 语言介绍 + 虚拟环境）
    relevant_ids = {0, 2}

    # 检索结果 A：效果好
    retrieved_a = [2, 0, 5, 1, 8]  # 前两个就是相关的
    # 检索结果 B：效果一般
    retrieved_b = [5, 1, 8, 2, 0]  # 相关的排在最后
    # 检索结果 C：效果差
    retrieved_c = [5, 1, 8, 3, 4]  # 一个都没命中

    metrics = RetrievalMetrics()

    print("\n场景：查询 'Python 怎么创建虚拟环境？'")
    print(f"相关文档 ID：{relevant_ids}")
    print()

    for name, retrieved in [("结果A（好）", retrieved_a), ("结果B（一般）", retrieved_b), ("结果C（差）", retrieved_c)]:
        print(f"📋 {name}：{retrieved}")
        print(f"   Recall@3    = {metrics.recall_at_k(retrieved, relevant_ids, 3):.2f}  "
              f"（前3个命中了 {len(set(retrieved[:3]) & relevant_ids)}/{len(relevant_ids)} 个相关文档）")
        print(f"   Precision@3 = {metrics.precision_at_k(retrieved, relevant_ids, 3):.2f}  "
              f"（前3个中有 {len(set(retrieved[:3]) & relevant_ids)} 个相关）")
        print(f"   MRR         = {metrics.mrr(retrieved, relevant_ids):.2f}  "
              f"（第一个相关文档排在第 {int(1/metrics.mrr(retrieved, relevant_ids)) if metrics.mrr(retrieved, relevant_ids) > 0 else '∞'} 位）")
        print(f"   Hit Rate@3  = {metrics.hit_rate_at_k(retrieved, relevant_ids, 3):.2f}")
        print()


def demo_ndcg():
    """NDCG 的详细演示"""
    print("=" * 60)
    print("Part 3: NDCG 详解")
    print("=" * 60)

    print("""
    NDCG 适合"相关度有等级"的场景。

    例如搜索"Python 教程"：
    - 文档A：Python 完整教程 → 相关度 3（高度相关）
    - 文档B：Python 简介     → 相关度 2（中等相关）
    - 文档C：Java 教程       → 相关度 0（不相关）
    - 文档D：Python 安装指南 → 相关度 1（轻微相关）
    """)

    relevance_scores = {0: 3, 1: 2, 2: 0, 3: 1}  # doc_id → relevance

    metrics = RetrievalMetrics()

    # 理想排序：按相关度降序
    ideal = [0, 1, 3, 2]
    # 实际排序：有些错位
    actual_good = [0, 3, 1, 2]
    actual_bad = [2, 3, 1, 0]

    for name, order in [("理想排序", ideal), ("实际排序A（较好）", actual_good), ("实际排序B（较差）", actual_bad)]:
        ndcg3 = metrics.ndcg_at_k(order, relevance_scores, 3)
        ndcg4 = metrics.ndcg_at_k(order, relevance_scores, 4)
        print(f"\n{name}：{order}")
        print(f"   NDCG@3 = {ndcg3:.4f}")
        print(f"   NDCG@4 = {ndcg4:.4f}")

    print("""
    NDCG 的关键洞察：
    - 高度相关的文档排在第一位，NDCG 就高
    - 高度相关的排在后面，NDCG 会惩罚（log2 衰减）
    - NDCG=1 表示完美排序
    """)


# ============================================================
# Part 5: 完整评估流程
# ============================================================

class RetrievalEvaluator:
    """
    检索系统评估器。

    用法：
        evaluator = RetrievalEvaluator()
        evaluator.evaluate(retrieval_system, eval_dataset)
        evaluator.report()
    """

    def __init__(self):
        self.results = []

    def evaluate_query(self, query: str, retrieved_ids: List[int],
                       relevant_ids: Set[int], k_values: List[int] = None) -> Dict:
        """评估单个查询"""
        if k_values is None:
            k_values = [1, 3, 5, 10]

        metrics = RetrievalMetrics()
        result = {"query": query, "scores": {}}

        for k in k_values:
            result["scores"][f"Recall@{k}"] = metrics.recall_at_k(retrieved_ids, relevant_ids, k)
            result["scores"][f"Precision@{k}"] = metrics.precision_at_k(retrieved_ids, relevant_ids, k)
            result["scores"][f"HitRate@{k}"] = metrics.hit_rate_at_k(retrieved_ids, relevant_ids, k)

        result["scores"]["MRR"] = metrics.mrr(retrieved_ids, relevant_ids)

        self.results.append(result)
        return result

    def evaluate_batch(self, eval_queries: List[EvalQuery],
                       retrieval_fn, k_values: List[int] = None) -> Dict:
        """
        批量评估。

        retrieval_fn: 接受 query 字符串，返回 doc_id 列表（按相关度排序）
        """
        all_results = []

        for eq in eval_queries:
            retrieved_ids = retrieval_fn(eq.query)
            result = self.evaluate_query(eq.query, retrieved_ids, eq.relevant_doc_ids, k_values)
            all_results.append(result)

        return self.aggregate()

    def aggregate(self) -> Dict:
        """汇总所有查询的指标"""
        if not self.results:
            return {}

        # 收集所有指标名
        all_metrics = set()
        for r in self.results:
            all_metrics.update(r["scores"].keys())

        # 计算均值
        summary = {}
        for metric_name in sorted(all_metrics):
            values = [r["scores"].get(metric_name, 0) for r in self.results]
            summary[metric_name] = {
                "mean": np.mean(values),
                "min": np.min(values),
                "max": np.max(values),
                "std": np.std(values),
            }

        return summary

    def report(self):
        """打印评估报告"""
        summary = self.aggregate()

        print("\n" + "=" * 60)
        print("📊 检索评估报告")
        print("=" * 60)
        print(f"\n评估查询数：{len(self.results)}")
        print(f"\n{'指标':<15} {'均值':<10} {'最小':<10} {'最大':<10} {'标准差':<10}")
        print("─" * 55)

        for metric_name, stats in summary.items():
            print(f"{metric_name:<15} {stats['mean']:<10.4f} {stats['min']:<10.4f} "
                  f"{stats['max']:<10.4f} {stats['std']:<10.4f}")

        # 找出最差的查询
        print(f"\n⚠️  最差的查询（Recall@5 最低）：")
        worst = sorted(self.results, key=lambda r: r["scores"].get("Recall@5", 0))
        for r in worst[:3]:
            print(f"   Recall@5={r['scores'].get('Recall@5', 0):.2f} | \"{r['query']}\"")

    def compare(self, other_summary: Dict, label_a: str = "系统A", label_b: str = "系统B"):
        """对比两个系统的指标"""
        summary_a = self.aggregate()

        print(f"\n{'=' * 60}")
        print(f"📊 对比：{label_a} vs {label_b}")
        print(f"{'=' * 60}")
        print(f"\n{'指标':<15} {label_a:<12} {label_b:<12} {'变化':<10}")
        print("─" * 49)

        for metric_name in sorted(summary_a.keys()):
            val_a = summary_a[metric_name]["mean"]
            val_b = other_summary.get(metric_name, {}).get("mean", 0)
            diff = val_b - val_a
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"{metric_name:<15} {val_a:<12.4f} {val_b:<12.4f} {arrow} {abs(diff):.4f}")


def demo_evaluation_pipeline():
    """演示完整的评估流程"""
    print("\n" + "=" * 60)
    print("Part 4: 完整评估流程演示")
    print("=" * 60)

    # 1. 构建评估数据集
    documents, eval_queries = build_eval_dataset()
    print(f"\n📄 文档库：{len(documents)} 篇")
    print(f"📝 评估查询：{len(eval_queries)} 个")

    # 2. 模拟两个检索系统
    np.random.seed(42)

    # 系统 A：向量检索（模拟）
    def system_a_retrieve(query: str) -> List[int]:
        """模拟向量检索：随机但偏向相关文档"""
        ids = list(range(len(documents)))
        np.random.seed(hash(query) % 2**31)
        np.random.shuffle(ids)
        return ids

    # 系统 B：混合检索（模拟，效果稍好）
    def system_b_retrieve(query: str) -> List[int]:
        """模拟混合检索：在系统A基础上，把相关文档提前"""
        # 找到该查询的相关文档
        eq = next((e for e in eval_queries if e.query == query), None)
        ids = list(range(len(documents)))
        np.random.seed(hash(query) % 2**31)
        np.random.shuffle(ids)
        if eq:
            # 把相关文档提到前面（模拟混合检索更准）
            for rel_id in sorted(eq.relevant_doc_ids, reverse=True):
                if rel_id in ids:
                    ids.remove(rel_id)
                    ids.insert(0, rel_id)
        return ids

    # 3. 评估系统 A
    print(f"\n{'─' * 55}")
    print("评估系统 A（纯向量检索）：")
    evaluator_a = RetrievalEvaluator()
    summary_a = evaluator_a.evaluate_batch(eval_queries, system_a_retrieve)
    evaluator_a.report()

    # 4. 评估系统 B
    print(f"\n{'─' * 55}")
    print("评估系统 B（混合检索）：")
    evaluator_b = RetrievalEvaluator()
    summary_b = evaluator_b.evaluate_batch(eval_queries, system_b_retrieve)
    evaluator_b.report()

    # 5. 对比
    evaluator_a.compare(summary_b, "系统A-向量", "系统B-混合")


# ============================================================
# Part 6: 评估驱动优化
# ============================================================

def demo_evaluation_driven_optimization():
    """
    用评估驱动 RAG 系统优化的完整流程。
    """
    print("\n" + "=" * 60)
    print("Part 5: 评估驱动优化流程")
    print("=" * 60)

    print("""
    评估驱动的优化闭环：

    ┌─────────────────────────────────────────────────┐
    │                                                  │
    │  1. 建立基线                                     │
    │     用当前系统跑评估，得到基线指标                 │
    │     例：Recall@5 = 0.72, MRR = 0.58             │
    │                                                  │
    │  2. 分析瓶颈                                     │
    │     哪些查询 Recall 低？为什么？                   │
    │     ├── 分块问题？（关键信息被切断）               │
    │     ├── Embedding 问题？（语义理解不准）           │
    │     ├── 检索策略问题？（该用混合检索）             │
    │     └── Top-K 太小？                             │
    │                                                  │
    │  3. 改进一个变量                                  │
    │     每次只改一个：分块策略 OR 模型 OR 检索方式     │
    │                                                  │
    │  4. 重新评估                                     │
    │     对比新旧指标，确认是否改善                     │
    │                                                  │
    │  5. 保留或回滚                                    │
    │     改善 → 保留，没改善 → 回滚，尝试其他方案       │
    │                                                  │
    │  循环 2-5，直到指标达标                           │
    └─────────────────────────────────────────────────┘
    """)

    print("""
    常见优化手段和预期提升：

    优化手段                          预期 Recall@5 提升
    ──────────────────                ─────────────────
    调整分块大小（200→300 tokens）     +3-5%
    增加重叠（20→50 tokens）          +2-3%
    换更好的 Embedding 模型           +5-10%
    加 BM25 混合检索                  +5-15%（精确查询场景）
    加 Reranker 精排                  +3-8%
    优化 Prompt 模板                  不影响检索，影响生成质量
    """)

    print("""
    目标参考值（通用 RAG 系统）：

    指标          差        及格      良好      优秀
    ────          ──        ──        ──        ──
    Recall@5      < 0.5     0.5-0.7   0.7-0.85  > 0.85
    Precision@5   < 0.3     0.3-0.5   0.5-0.7   > 0.7
    MRR           < 0.4     0.4-0.6   0.6-0.8   > 0.8
    NDCG@5        < 0.5     0.5-0.7   0.7-0.85  > 0.85
    """)


# ============================================================
# 主函数
# ============================================================

def main():
    """运行所有演示"""
    demo_why_metrics()
    demo_metrics()
    demo_ndcg()
    demo_evaluation_pipeline()
    demo_evaluation_driven_optimization()

    print("\n" + "=" * 60)
    print("📝 本课要点总结")
    print("=" * 60)
    print("""
    1. 评估指标是 RAG 优化的基础，没有指标就是瞎猜
    2. Recall@K：最重要，相关文档有没有被检索到
    3. Precision@K：检索结果的纯度
    4. MRR：第一个相关结果排在第几
    5. NDCG：排序质量，考虑位置衰减和相关度等级
    6. 建评估数据集：人工标注 > LLM辅助标注 > 用户行为日志
    7. 优化闭环：基线 → 分析瓶颈 → 改一个变量 → 重新评估 → 保留/回滚
    8. 混合检索 + Reranker 通常是提升检索质量最有效的手段
    """)


if __name__ == "__main__":
    main()

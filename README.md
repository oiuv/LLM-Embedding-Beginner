# 从零开始学习Embedding：全面教程

这是一个系统化的文本嵌入(Embedding)学习项目，从基础概念到高级应用，帮助你全面掌握文本嵌入技术。

## 📚 学习路径

### 第一阶段：基础概念 (01-basics/)
- `01-what-is-embedding.py` - 什么是文本嵌入
- `02-first-embedding.py` - 获取第一个文本向量
- `03-similarity-calculation.py` - 计算文本相似度
- `04-vector-operations.py` - 向量操作基础

### 第二阶段：中级应用 (02-intermediate/)
- `01-semantic-search.py` - 语义搜索
- `02-text-classification.py` - 文本分类
- `03-recommendation-system.py` - 推荐系统
- `04-clustering-analysis.py` - 聚类分析

### 第三阶段：高级功能 (03-advanced/)
- `01-knowledge-base.py` - 知识库构建
- `02-anomaly-detection.py` - 异常检测
- `03-visualization.py` - 可视化技术
- `04-performance-optimization.py` - 性能优化

### 文本排序模型 (03-text-reranking/)
- `01-text-rerank-basics.py` - 排序模型基础
- `02-integration-guide.py` - 系统集成指南
- `03-comparison-demo.py` - 对比演示
- `04-performance-benchmark.py` - 性能基准测试

### 第四阶段：实战项目 (04-projects/)
- `01-smart-qa-system/` - 智能问答系统
- `02-content-recommendation/` - 内容推荐引擎
- `03-semantic-search-engine/` - 语义搜索引擎
- `04-document-analyzer/` - 文档分析工具

### 第五阶段：练习与挑战 (05-exercises/)
- 基础练习：相似度计算
- 中级挑战：多语言支持
- 高级项目：实时推荐系统

### 第六阶段：RAG 完整体系 (06-rag-complete/)
- `01-chunking-strategies.py` - 文本分块策略（五种策略对比）
- `02-vector-databases.py` - 向量数据库（FAISS/Chroma/Milvus 对比）
- `03-rag-pipeline.py` - RAG Pipeline 完整实现
- `04-embedding-models.py` - Embedding 模型对比与选型
- `05-hybrid-search.py` - 混合检索（BM25 + 向量）
- `06-retrieval-evaluation.py` - 检索评估指标（Recall/MRR/NDCG）
- `07-multimodal-embedding.py` - 多模态 Embedding（CLIP）

## 🚀 快速开始

### 🎯 一键启动（推荐）

#### 方式1：交互式体验
```bash
python main.py
```

#### 方式2：快速体验模式
```bash
python main.py quick
```

启动后会看到彩色菜单：
```
🚀 Embedding Learning Lab - 交互式学习平台
============================================================

1. 按阶段学习
2. 按功能体验  
3. 推荐学习路径
4. 一键运行所有基础
5. 检查环境
q. 退出
```

### 🔧 传统方式启动

#### 环境要求
```bash
pip install openai numpy scikit-learn pandas matplotlib seaborn sqlite3 dashscope
```

#### 配置API密钥
```bash
export DASHSCOPE_API_KEY="你的API密钥"
```

#### 运行第一个示例
```bash
python 01-basics/01-what-is-embedding.py
```

#### 体验文本排序模型
```bash
# 基础教程
python 03-text-reranking/01-text-rerank-basics.py

# 系统集成演示
python 03-text-reranking/02-integration-guide.py

# 性能对比分析
python 03-text-reranking/03-comparison-demo.py

# 完整性能基准测试
python 03-text-reranking/04-performance-benchmark.py
```

### 📱 使用示例

#### 零基础入门
1. 运行 `python main.py`
2. 选择 `1. 按阶段学习`
3. 选择 `基础概念 (01-basics)`
4. 按数字选择想学的教程

#### ⚡ 快速体验
1. 运行 `python main.py quick`
2. 自动运行3个精选示例
3. 每一步都有确认提示

#### 🛠️ 环境检查
运行前会自动检查：
- ✅ API密钥配置
- ✅ 依赖库安装
- ✅ 文件完整性

### 🎨 特色功能

#### 🎯 智能路径推荐
- **零基础路径**：基础概念 → 语义搜索
- **业务应用路径**：中级应用 → 高级优化
- **完整精通路径**：全部教程循序渐进

#### 🔄 一键批量运行
- 一键运行所有基础教程
- 按功能分类体验
- 支持中断和继续

#### 📊 彩色友好界面
- 绿色：成功提示
- 蓝色：信息展示
- 红色：错误提示
- 黄色：用户输入

### 🎪 使用小贴士

1. **随时退出**：按 `q` 或 `Ctrl+C` 安全退出
2. **逐步确认**：每个步骤都有确认提示，不会错过
3. **彩色输出**：Windows/Mac/Linux 都支持彩色显示
4. **错误友好**：详细的错误提示和解决建议

### 🚀 常用操作

| 操作 | 命令 | 说明 |
|------|------|------|
| 启动主界面 | `python main.py` | 交互式学习 |
| 快速体验 | `python main.py quick` | 自动运行精选示例 |
| 检查环境 | `python main.py` 选5 | 一键环境检测 |
| 退出 | `q` 或 Ctrl+C | 随时安全退出 |

## 📖 学习进度

- [ ] 基础概念理解
- [ ] 环境配置完成
- [ ] 基础功能掌握
- [ ] 中级应用实践
- [ ] 高级功能探索
- [ ] 实战项目完成

## 🔧 项目结构

```
embedding-learning-lab/
├── 01-basics/           # 基础概念
├── 02-intermediate/     # 中级应用
├── 03-advanced/        # 高级功能
├── 04-projects/        # 实战项目
├── 05-exercises/       # 练习与挑战
├── data/               # 示例数据
├── docs/               # 学习文档
├── tests/              # 测试文件
├── utils/              # 工具函数
└── README.md          # 项目说明
```

## 📊 学习成果

完成本项目后，你将能够：

1. **理解原理**：深入理解文本嵌入的工作原理
2. **实践应用**：掌握多种实际应用场景
3. **问题解决**：能够独立解决相关问题
4. **项目开发**：开发完整的embedding应用
5. **性能优化**：优化embedding系统的性能

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个学习项目！

## 📞 联系方式

如有问题，请通过GitHub Issues联系我们。
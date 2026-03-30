# m-indexflow — LlamaIndex RAG 管线可视化教学

一个可交互的教学工具，以可视化方式展示 LlamaIndex 完整 RAG 管线的执行过程。支持 demo 模式（内存存储）和 production 模式（Milvus 向量 + ES BM25 混合检索）。

## 快速启动

```bash
cd m-indexflow

# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. 配置（编辑项目根目录 .env）
cp .env.example ../.env
# 编辑 .env，填入 LLM API Key

# 3. 启动
python backend/main.py

# 4. 浏览器访问
# http://localhost:8766
```

## 两种运行模式

### Demo 模式（默认）

`.env` 中设置 `MODE=demo` 或不设置。

- 内置 3 类示例文档（规章制度 / 产品手册 / 会议纪要）
- 内存存储，不需要外部数据库
- 适合学习 LlamaIndex 核心概念

**操作**：打开页面 → 点击"构建索引" → 输入问题查询

### Production 模式

`.env` 中设置 `MODE=production` 并配置 Milvus + ES 连接信息。

```env
MODE=production
MILVUS_HOST=your-host
MILVUS_PORT=19530
MILVUS_USER=root
MILVUS_PASSWORD=your-password
MILVUS_COLLECTION=m_indexflow_kb
ES_URL=http://elastic:password@your-host:9200
ES_INDEX=m_indexflow_kb
```

- 支持上传真实文件（PDF / Markdown / TXT）
- 向量写入 Milvus，文本写入 Elasticsearch
- Milvus 向量检索 + ES BM25 关键词检索，RRF 融合排序
- 启动时自动检测连通性，连接失败自动回退 demo 模式

**操作**：打开页面 → 上传文件 → 点击"构建索引" → 输入问题查询

## 怎么提问？

### Demo 模式示例

| 问题 | 路由目标 | 展示的 LlamaIndex 特性 |
|------|---------|----------------------|
| 出差报销标准是多少？ | 规章制度 | AutoMergingRetriever + Refine |
| DataFlow Pro 支持哪些数据库？ | 产品手册 | VectorIndexRetriever + TreeSummarize |
| 上周会议的主要结论？ | 会议纪要 | SummaryIndex + TreeSummarize |
| 对比 DataFlow 和 CloudSync | 跨文档 | SubQuestionQueryEngine |

### Production 模式

上传你自己的文档后，直接用自然语言提问。支持两种查询模式：
- **普通查询 (Router)**：自动路由到合适的 QueryEngine
- **跨文档对比 (SubQuestion)**：复杂问题拆解为子问题分别查询

## 页面说明

```
┌──────────────────┬────────────────┬──────────────────────┐
│ 左栏：Node 树     │ 中栏：对话区    │ 右栏：执行步骤        │
│                  │               │                      │
│ 文档解析后的       │ 文件上传       │ 步骤卡片（倒序）       │
│ Node 层级结构     │ (production)   │ 点击展开详情：         │
│                  │ 查询模式选择    │ · 代码片段            │
│ 📋 规章制度       │ 输入框         │ · 输入/输出数据        │
│  ├─ Node1        │ 消息列表       │ · 教学说明            │
│  ├─ Node2        │               │ · 组件标签(绿色)       │
│  └─ ...          │               │                      │
│ 📊 产品手册       │               │                      │
│ 💬 会议纪要       │               │                      │
├──────────────────┴────────────────┤                      │
│ 底部：步骤时间线                    │ [上一步][下一步][自动] │
└────────────────────────────────────┴──────────────────────┘
```

## 覆盖的 LlamaIndex 特性

### 文档加载与 Node 解析

| 特性 | 说明 |
|------|------|
| `SimpleDirectoryReader` | 自动识别文件格式并解析为 Document |
| `SentenceSplitter` | 按句子边界 + token 数切片 |
| `HierarchicalNodeParser` | 多级层级切片（章 → 节 → 段），Node 自带 parent/child 关系 |
| `IngestionPipeline` | 标准数据摄入管线 |

### 索引构建

| 特性 | 说明 |
|------|------|
| `VectorStoreIndex` | 向量索引（demo: 内存 / production: Milvus） |
| `SummaryIndex` | 摘要索引，遍历所有 Node 生成全文摘要 |
| `StorageContext` | 存储上下文，关联 VectorStore 和 DocStore |

### 查询引擎

| 特性 | 说明 |
|------|------|
| `RouterQueryEngine` | 根据问题自动选择合适的 QueryEngine |
| `SubQuestionQueryEngine` | 复杂问题拆解为子问题分别查询 |
| `AutoMergingRetriever` | 检索叶子 Node 后自动合并回父 Node |
| `QueryFusionRetriever` | Milvus 向量 + ES BM25 双路 RRF 融合（production 模式） |
| `RetrieverQueryEngine` | Retriever + PostProcessor + Synthesizer 三件套 |

### 响应合成

| 特性 | 说明 |
|------|------|
| `ResponseSynthesizer(refine)` | 逐个 Node 精炼回答 |
| `ResponseSynthesizer(tree_summarize)` | 树形递归摘要合并 |
| `SimilarityPostprocessor` | 相似度截断过滤 |

### 存储（Production 模式）

| 特性 | 说明 |
|------|------|
| 自定义 `MilvusStore` | 继承 BasePydanticVectorStore，pymilvus ORM 直连 |
| 自定义 `ESBm25Retriever` | 继承 BaseRetriever，elasticsearch-py 查 ES BM25 |
| `QueryFusionRetriever` | RRF (Reciprocal Rank Fusion) 双路融合排序 |

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 主页面 |
| GET | `/api/mode` | 返回当前模式和存储连接状态 |
| POST | `/api/upload` | 上传文件（multipart/form-data，production 模式） |
| POST | `/api/run` | 构建索引，SSE 返回每步事件 |
| POST | `/api/query` | 执行查询 `{query, mode}`，SSE 返回每步事件 |
| POST | `/api/reset` | 重置所有状态 |
| GET | `/api/steps` | 获取步骤历史 |
| GET | `/api/node_tree` | 获取 Node 树 JSON |

## 项目结构

```
m-indexflow/
├── .env.example              # 配置模板
├── requirements.txt
├── README.md
├── ARCHITECTURE.md           # 架构文档
├── backend/
│   ├── main.py               # FastAPI 入口，API 路由
│   ├── config.py             # 配置管理，LLM/Embedding 初始化
│   ├── step_tracer.py        # 步骤追踪器，SSE 推送
│   ├── sample_docs.py        # 内置示例文档
│   ├── file_handler.py       # 文件上传与解析
│   ├── ingestion.py          # 文档加载 + Node 解析管线
│   ├── indexing.py            # 索引构建 + QueryEngine 组装
│   ├── query_engine.py       # 查询执行 + StepTracer 埋点
│   ├── storage.py            # StorageFactory 工厂（demo/production 切换）
│   ├── milvus_store.py       # 自定义 Milvus 向量存储
│   └── es_retriever.py       # 自定义 ES BM25 检索器
├── static/
│   └── index.html            # 前端页面
└── venv/                     # Python 虚拟环境
```

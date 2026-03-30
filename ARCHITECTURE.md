# m-indexflow 架构文档

## 一、系统架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          浏览器（index.html）                            │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────┐    │
│  │  Node 树      │  │  对话区       │  │  步骤列表（倒序卡片）        │    │
│  │  左栏 24%     │  │  中栏 34%     │  │  右栏 42%                  │    │
│  └──────────────┘  └──────────────┘  └────────────────────────────┘    │
│  └──────── 时间线 ─────────────────────────────────────────────────┘    │
└─────┬──────────────────┬─────────────────┬────────────────────────────┘
      │ GET /api/node_tree│ POST /api/run   │ POST /api/query
      │                  │ POST /api/upload │
      ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        FastAPI 后端（main.py）                           │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ file_handler  │  │  ingestion   │  │   indexing    │  │query_engine│ │
│  │ 文件上传解析   │→│ Node 解析     │→│ 索引构建       │→│ 查询执行    │ │
│  └──────────────┘  └──────────────┘  └──────┬───────┘  └─────┬──────┘ │
│                                             │                │        │
│                    ┌────────────────────────┐│                │        │
│                    │   StorageFactory       ││                │        │
│                    │   存储层工厂            ││                │        │
│                    └───┬───────────┬────────┘│                │        │
│                        │           │         │                │        │
│           ┌────────────┤           │         │                │        │
│           ▼            ▼           ▼         ▼                ▼        │
│  ┌─────────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ ┌──────────────┐ │
│  │MilvusStore  │ │ESBm25    │ │内存     │ │内存    │ │StepTracer    │ │
│  │(production) │ │Retriever │ │Vector  │ │BM25   │ │ SSE 推送      │ │
│  │pymilvus ORM │ │(prod.)   │ │(demo)  │ │(demo) │ │              │ │
│  └──────┬──────┘ └────┬─────┘ └────────┘ └────────┘ └──────┬───────┘ │
└─────────┼──────────────┼─────────────────────────────────────┼─────────┘
          ▼              ▼                                     ▼
   ┌────────────┐  ┌──────────┐                         ┌──────────┐
   │  Milvus    │  │  Elastic │                         │  浏览器   │
   │  向量数据库 │  │  search  │                         │  SSE 流   │
   └────────────┘  └──────────┘                         └──────────┘
```

## 二、模式切换机制

```
.env: MODE=demo | production
         │
         ▼
    config.py 读取 MODE
         │
         ├── MODE=demo
         │     └── StorageFactory 返回：
         │           ├── get_vector_store() → None（VectorStoreIndex 用默认内存存储）
         │           └── get_bm25_retriever(nodes) → BM25Retriever（rank_bm25 内存计算）
         │
         └── MODE=production
               │
               ├── check_production_connectivity()
               │     ├── Milvus ping → 成功/失败
               │     └── ES ping → 成功/失败
               │     └── 任一失败 → MODE 回退为 demo，日志告警
               │
               └── StorageFactory 返回：
                     ├── get_vector_store() → MilvusStore（pymilvus ORM 连接真实 Milvus）
                     ├── get_bm25_retriever() → ESBm25Retriever（elasticsearch-py 查真实 ES）
                     └── index_nodes_to_es(nodes) → ES bulk 写入
```

## 三、数据流详解

### 3.1 Demo 模式完整流程

```
用户点击"构建索引"
│
▼
POST /api/run → SSE 流
│
├── ingestion.py: run_ingestion(tracer)
│   │
│   ├── Step 1: sample_docs.py 加载 3 个内置 Document
│   │   Document(text="规章制度...", metadata={"source": "公司规章制度"})
│   │   Document(text="产品手册...", metadata={"source": "产品手册"})
│   │   Document(text="会议纪要...", metadata={"source": "会议纪要"})
│   │
│   ├── Step 2-3: SentenceSplitter(chunk_size=256, chunk_overlap=32)
│   │   产品手册 Document → 4 个 TextNode
│   │   会议纪要 Document → 4 个 TextNode
│   │
│   ├── Step 4-6: HierarchicalNodeParser(chunk_sizes=[1024, 256, 64])
│   │   规章制度 Document → 29 个 Node（根 + 中间 + 叶子，带 parent/child 关系）
│   │
│   └── Step 7: IngestionPipeline 教学说明
│
├── indexing.py: build_indexes(nodes_data, tracer)
│   │
│   ├── Step 8: VectorStoreIndex(product_nodes, embed_model)
│   │   对每个 Node 调用 Embedding → 存入 SimpleVectorStore（内存）
│   │
│   ├── Step 9: SummaryIndex(meeting_nodes)
│   │   不需要 Embedding，全量遍历式索引
│   │
│   ├── Step 10: VectorStoreIndex(policy_leaf_nodes, storage_context)
│   │   叶子节点建向量索引 + 所有层级 Node 存入 DocStore
│   │   为 AutoMergingRetriever 做准备
│   │
│   ├── Step 11: RetrieverQueryEngine 组装
│   │   AutoMergingRetriever + SimilarityPostprocessor + Refine Synthesizer
│   │
│   ├── Step 12-13: 标准 QueryEngine + SummaryIndex QueryEngine
│   │
│   ├── Step 14: RouterQueryEngine + SubQuestionQueryEngine
│   │   3 个 QueryEngineTool 注册到 Router
│   │   LLMSingleSelector 做路由决策
│   │
│   └── Step 15: 索引构建完成
│
▼
前端收到所有 Step 事件 → 渲染步骤列表 + Node 树 + 时间线
```

### 3.2 Production 模式完整流程

```
用户上传文件
│
▼
POST /api/upload (multipart/form-data)
│
├── file_handler.py: save_and_parse(filename, content)
│   ├── 保存到 /tmp/m_indexflow_uploads/{file_id}_{filename}
│   ├── SimpleDirectoryReader(input_files=[path]).load_data()
│   │   根据扩展名自动选择 Reader：
│   │   .pdf → PDFReader (pypdf)
│   │   .md  → MarkdownReader
│   │   .txt → 直接读取
│   └── 返回 {file_id, filename, doc_count}
│
▼
用户点击"构建索引"
│
▼
POST /api/run → SSE 流
│
├── ingestion.py: run_ingestion_from_files(documents, tracer)
│   ├── Step 1: 加载已上传的 Document 列表
│   ├── Step 2-3: SentenceSplitter 切片 → TextNode 列表
│   └── 构建 Node 树 JSON（前端渲染用）
│
├── indexing.py: build_indexes_production(nodes_data, tracer)
│   │
│   ├── Step 4: MilvusStore 向量写入
│   │   StorageFactory.get_vector_store()
│   │     → MilvusStore(host, port, user, password, collection_name, dim=1024)
│   │       内部：
│   │       1. pymilvus connections.connect(alias=uuid_hex, host=..., password=...)
│   │       2. Collection 创建（id/node_id/text/metadata/embedding 字段）
│   │       3. HNSW 索引（COSINE 度量）
│   │   VectorStoreIndex(nodes, embed_model, storage_context)
│   │     → 每个 Node: embed_model(text) → 1024 维向量 → Milvus insert
│   │
│   ├── Step 5: ES BM25 入库
│   │   StorageFactory.index_nodes_to_es(nodes)
│   │     → ESBm25Retriever.index_nodes(nodes)
│   │       内部：
│   │       1. 检查索引是否存在，不存在则创建
│   │          mapping: node_id(keyword), text(text), doc_source(keyword), metadata(object)
│   │          检测 IK 分词器，不可用回退 standard
│   │       2. elasticsearch.helpers.bulk() 批量写入
│   │          每条: {node_id, text, doc_source, metadata}
│   │
│   ├── Step 6: QueryFusionRetriever 构建
│   │   vector_retriever = vector_index.as_retriever(top_k=5)
│   │   es_retriever = ESBm25Retriever(es_url, index_name, top_k=5)
│   │   fusion = QueryFusionRetriever(
│   │       retrievers=[vector_retriever, es_retriever],
│   │       mode="reciprocal_rerank",  # RRF 融合
│   │       num_queries=1,             # 不生成查询变体
│   │   )
│   │
│   ├── Step 7: RetrieverQueryEngine 组装
│   │   engine = RetrieverQueryEngine(
│   │       retriever=fusion_retriever,
│   │       response_synthesizer=Refine,
│   │   )
│   │
│   └── Step 8: 索引构建完成
│
▼
用户输入查询
│
▼
POST /api/query {query, mode} → SSE 流
│
├── mode="router" → _query_production(query)
│   │
│   ├── Step 1: QueryFusionRetriever 混合检索
│   │   并行执行：
│   │   ├── vector_retriever.retrieve(query)
│   │   │   → embed_model(query) → 1024 维向量
│   │   │   → Milvus search(vector, top_k=5, metric=COSINE)
│   │   │   → List[NodeWithScore] (score: 余弦相似度 0~1)
│   │   │
│   │   └── es_retriever._retrieve(query)
│   │       → ES match 查询 {text: query, minimum_should_match: "30%"}
│   │       → List[NodeWithScore] (score: BM25 分数)
│   │   │
│   │   └── RRF 融合
│   │       对每个 Node 计算: rrf_score = Σ 1/(k + rank_in_list_i)
│   │       k=60（默认），按 rrf_score 降序排列
│   │       → List[NodeWithScore] (score: RRF 分数 ~0.01~0.03)
│   │
│   ├── Step 2: 检索结果
│   │   response.source_nodes → 融合后的 Node 列表
│   │
│   └── Step 3: ResponseSynthesizer(Refine) 生成回答
│       逐个 Node 调用 LLM 精炼答案
│
└── mode="sub_question" → SubQuestionQueryEngine
    ├── LLM 分解问题 → 子问题列表
    ├── 每个子问题 → QueryEngineTool → 混合检索 → 子回答
    └── LLM 合并子回答 → 最终回答
```

## 四、核心模块详解

### 4.1 MilvusStore（milvus_store.py）

自定义 Milvus 向量存储，继承 LlamaIndex 的 `BasePydanticVectorStore`。

**为什么不用 `llama-index-vector-stores-milvus` 官方包？**

官方包 0.4.x 内部混用 `MilvusClient`（新 API）和 `Collection`（ORM API），两套 API 的连接管理互不兼容（alias 不同步），导致 `ConnectionNotExistException`。

**解决方案**：参考 ZFAPT 主项目的做法，直接用 pymilvus ORM API：

```python
class MilvusStore(BasePydanticVectorStore):
    """参考 ZFAPT 的 pymilvus ORM 连接方式"""

    def __init__(self, host, port, user, password, collection_name, dim):
        # 1. connections.connect(alias=uuid_hex)  ← ZFAPT 的标准做法
        # 2. Collection 创建/加载
        # 3. HNSW 索引创建

    def add(self, nodes) -> List[str]:
        # Node 数据转为 list[dict] → collection.insert()

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        # collection.search(embedding, top_k) → List[NodeWithScore]
```

**Collection Schema**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INT64, auto_id | Milvus 自动生成的主键 |
| `node_id` | VARCHAR(128) | LlamaIndex Node ID |
| `text` | VARCHAR(65535) | Node 文本内容 |
| `metadata` | VARCHAR(65535) | JSON 序列化的 metadata |
| `embedding` | FLOAT_VECTOR(1024) | Embedding 向量 |

索引：HNSW，metric_type=COSINE，M=16，efConstruction=256

### 4.2 ESBm25Retriever（es_retriever.py）

自定义 ES BM25 检索器，继承 LlamaIndex 的 `BaseRetriever`。

**为什么不用 LlamaIndex 内置的 `BM25Retriever`？**

内置的 BM25Retriever 是纯内存方案（rank_bm25 库），需要把所有 Node 文本加载到内存中计算。数据量超过 10 万条就不现实。ESBm25Retriever 走 ES 倒排索引，无数据量上限。

```python
class ESBm25Retriever(BaseRetriever):
    """继承 BaseRetriever，对上层完全透明"""

    def _retrieve(self, query_bundle) -> List[NodeWithScore]:
        # ES match 查询 → hits → NodeWithScore 列表

    def index_nodes(self, nodes):
        # elasticsearch.helpers.bulk() 批量写入
```

**ES Index Mapping**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `node_id` | keyword | Node ID，去重用 |
| `text` | text (analyzer: ik_max_word 或 standard) | BM25 检索字段 |
| `doc_source` | keyword | 来源文档名 |
| `metadata` | object (enabled: false) | 元数据，不索引 |

### 4.3 StorageFactory（storage.py）

存储层工厂，根据 MODE 返回不同实现：

```
StorageFactory
├── get_vector_store()
│   ├── demo       → None（VectorStoreIndex 自动用内存）
│   └── production → MilvusStore 单例
│
├── get_bm25_retriever(nodes, top_k)
│   ├── demo       → BM25Retriever(nodes, tokenizer=jieba.lcut)
│   └── production → ESBm25Retriever 单例
│
├── index_nodes_to_es(nodes, tracer)
│   ├── demo       → 跳过
│   └── production → ESBm25Retriever.index_nodes()
│
└── reset()
    ├── demo       → 无操作
    └── production → drop Milvus collection + delete ES index
```

### 4.4 QueryFusionRetriever 混合检索

```
                   用户查询
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
  ┌───────────────┐       ┌───────────────┐
  │ Milvus 向量检索 │       │ ES BM25 检索   │
  │               │       │               │
  │ query → embed │       │ query → ES    │
  │ → HNSW search │       │   match 查询   │
  │ → top 5      │       │ → top 5       │
  │               │       │               │
  │ score: 余弦   │       │ score: BM25    │
  │ 0.70, 0.65.. │       │ 5.57, 3.13..  │
  └───────┬───────┘       └───────┬───────┘
          │                       │
          └───────────┬───────────┘
                      ▼
              ┌───────────────┐
              │  RRF 融合排序   │
              │               │
              │ 对每个 Node:   │
              │ score = Σ     │
              │  1/(60+rank)  │
              │               │
              │ 不需要调权重    │
              │ 两路优势互补    │
              └───────┬───────┘
                      ▼
              排序后的 Node 列表
              score: 0.033, 0.017...
                      │
                      ▼
           ResponseSynthesizer(Refine)
              逐个 Node 精炼答案
                      │
                      ▼
                  最终回答
```

**为什么用 RRF 而不是加权融合？**

| 对比 | 加权融合（ZFAPT 的方式） | RRF（本项目的方式） |
|------|----------------------|-------------------|
| 公式 | `score = w1*vec + w2*bm25` | `score = Σ 1/(k+rank)` |
| 需要调参 | 是（weights 0.95:0.05） | 否（k=60 是固定常数） |
| 分数归一化 | 需要（两路分数量级不同） | 不需要（只看排名不看分数） |
| 鲁棒性 | 权重选不好效果差 | 对分数分布不敏感 |

**注意**：RRF 融合后的分数很小（~0.01~0.03），不能用 `SimilarityPostprocessor(cutoff=0.1)` 过滤，否则会把所有结果全部过滤掉。

### 4.5 StepTracer（step_tracer.py）

每个关键操作记录一个 Step 事件，通过 asyncio.Queue 推送给 SSE：

```python
tracer.trace(
    phase='index',                          # 阶段：ingest/index/query/error
    title='Milvus — VectorStoreIndex 构建',  # 步骤标题
    code='▶ index = VectorStoreIndex(...)',   # 代码片段（▶ 标记当前行）
    input_data={...},                        # 实际输入数据
    output_data={...},                       # 实际输出数据
    explanation='这一步做了什么...',           # 教学说明
    component='LlamaIndex:VectorStoreIndex', # 组件标识
)
```

SSE 事件格式：
```json
data: {"step": 4, "phase": "index", "title": "...", "code": "...", "component": "LlamaIndex:MilvusVectorStore", ...}
```

前端通过 `EventSource` 或 `fetch` + `ReadableStream` 接收。

### 4.6 LLM 和 Embedding 适配

**问题**：通义千问的模型名（`qwen-max`、`text-embedding-v3`）不在 LlamaIndex 的白名单中。

**LLM 解决方案**：使用 `llama-index-llms-openai-like` 官方包，专为兼容 OpenAI 接口的非 OpenAI 模型设计，不做模型名校验。

```python
from llama_index.llms.openai_like import OpenAILike
llm = OpenAILike(api_key=..., api_base=..., model="qwen-max", is_chat_model=True, context_window=32000)
```

**Embedding 解决方案**：`llama_index.embeddings.openai.OpenAIEmbedding` 有模型名枚举校验，无法传自定义名称。通过继承 `BaseEmbedding` 直接调用 OpenAI SDK：

```python
class CompatEmbedding(BaseEmbedding):
    def _get_text_embedding(self, text):
        return openai_client.embeddings.create(input=[text], model="text-embedding-v3").data[0].embedding
```

## 五、前后端交互流程

```
前端                                          后端
───                                          ───

1. 页面加载
   GET /api/mode ──────────────────────────→ 返回 {mode, connectivity}
   ← {mode: "production", connectivity: {milvus: true, es: true}}
   根据 mode 显示/隐藏上传区域

2. 文件上传（production 模式）
   POST /api/upload ───────────────────────→ file_handler.save_and_parse()
   (multipart/form-data)                     SimpleDirectoryReader 解析
   ← {file_id, filename, doc_count}          缓存 Document 到 uploaded_documents

3. 构建索引
   POST /api/run ──────────────────────────→ _build_all() 在线程池执行
   ← SSE 流                                  ingestion + indexing
      data: {step:1, title:"加载文档", ...}    每步 tracer.trace() → Queue
      data: {step:2, title:"切片", ...}       Queue → SSE yield
      ...
      data: {"done": true}                    tracer.finish()

   GET /api/node_tree ─────────────────────→ 返回 Node 树 JSON
   渲染左栏 Node 树

4. 查询
   POST /api/query ────────────────────────→ _query() 在线程池执行
   {query: "...", mode: "router"}             QueryEngine.query()
   ← SSE 流                                  每步 tracer.trace()
      data: {step:1, title:"混合检索", ...}
      data: {step:2, title:"检索完成", output_data: {nodes: [...]}}
      data: {step:3, title:"回答生成", output_data: {answer: "..."}}
      data: {"done": true}
   对话区显示回答

5. 重置
   POST /api/reset ────────────────────────→ 清空 state + StorageFactory.reset()
                                              drop Milvus collection + delete ES index
```

## 六、依赖版本说明

| 依赖 | 版本 | 说明 |
|------|------|------|
| `llama-index-core` | >=0.12,<0.13 | LlamaIndex 核心 |
| `llama-index-llms-openai-like` | >=0.4,<0.5 | 兼容非 OpenAI 模型 |
| `llama-index-question-gen-openai` | >=0.3,<0.4 | SubQuestionQueryEngine 依赖 |
| `llama-index-readers-file` | >=0.4,<0.5 | PDF/MD/DOCX 文件解析 |
| `elasticsearch` | >=7,<8 | ES 8.x 服务端用 v7 客户端（v8/v9 客户端有 Accept header 兼容问题） |
| `pymilvus` | 随 llama-index 安装 | Milvus ORM 客户端 |
| `pypdf` | 最新 | PDF 文本提取 |
| `jieba` | 最新 | 中文分词（demo 模式 BM25 用） |

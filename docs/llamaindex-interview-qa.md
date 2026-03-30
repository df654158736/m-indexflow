# LlamaIndex 面试问答 — 基于 m-indexflow 项目实战

## 一、LlamaIndex 基础

### Q1：LlamaIndex 是什么？和 LangChain 有什么区别？

LlamaIndex 是一个专注于 **RAG（检索增强生成）** 的框架，核心是把私有数据和 LLM 连接起来。

```
LlamaIndex 的强项                    LangChain 的强项
──────────────                      ──────────────
数据摄入（Document → Node）           模型调用抽象（ChatModel / Messages）
多种索引类型（Vector / Summary / Tree） Agent + 工具链
查询引擎（Router / SubQuestion）       Callback 回调体系
Node 层级关系 + AutoMerging           LangGraph 工作流编排
```

一句话总结：**LlamaIndex 更擅长"数据怎么进来、怎么检索"，LangChain 更擅长"模型怎么调用、流程怎么编排"。** 实际项目中经常一起用。

### Q2：LlamaIndex 的核心概念有哪些？

```
Document → Node → Index → Retriever → QueryEngine → Response

Document  : 原始文档，一个 PDF/MD/TXT 文件解析后的文本
Node      : 文档切片后的最小单元，LlamaIndex 的一等公民
Index     : Node 的索引结构（向量索引、摘要索引等）
Retriever : 从 Index 中检索相关 Node 的组件
QueryEngine: Retriever + Synthesizer 的组合，完成"检索+生成"
Response  : 最终回答，包含 source_nodes（溯源）
```

### Q3：Node 和普通的 chunk 有什么区别？

**Node 是 LlamaIndex 的一等公民**，比普通 chunk 多了几个关键能力：

```python
class TextNode:
    node_id: str           # 唯一 ID
    text: str              # 文本内容
    metadata: dict         # 元数据（来源、页码等）
    relationships: dict    # 与其他 Node 的关系 ← 这是关键区别
        PARENT → 父节点 ID
        CHILD  → 子节点 ID 列表
        PREV   → 前一个 Node
        NEXT   → 后一个 Node
    embedding: list[float] # 向量（可选）
```

普通 chunk 就是一段文本，Node 之间有关系。这个关系是 AutoMergingRetriever 能工作的基础。

---

## 二、文档解析与切片

### Q4：项目中怎么处理 PDF 文件？

```
PDF 上传
  │
  ▼
SimpleDirectoryReader 解析 → 6 个 Document（每页一个）
  │
  ▼
合并同一来源的 Document → 1 个完整 Document（关键步骤！）
  │
  ▼
HierarchicalNodeParser 三级切片 → 87 个 Node（Root + Mid + Leaf）
```

**为什么要合并？** PDF 按页解析会把一个完整段落切到两个 Document 里。如果不合并，HierarchicalNodeParser 无法跨 Document 建立层级关系，导致四大满贯列表被切断。

### Q5：SentenceSplitter 和 HierarchicalNodeParser 有什么区别？

```
SentenceSplitter:
  输入: Document
  输出: N 个平级 TextNode（互相没有关系）
  切法: 按句子边界 + token 数，固定大小
  适合: 简单 QA

HierarchicalNodeParser:
  输入: Document
  输出: N 个层级 TextNode（有 parent/child 关系）
  切法: 三级层级（如 2048 → 512 → 256）
  适合: 列表型、层级型文档

项目中的配置:
  chunk_sizes=[2048, 512, 256]
  Level 1 (2048 token): 章级别，上下文完整
  Level 2 (512 token):  节级别
  Level 3 (256 token):  段级别，检索精准
```

### Q6：chunk_size 怎么选？有什么 trade-off？

```
chunk_size 小（64-256）:
  ✅ 向量检索精准（小块更容易匹配特定问题）
  ❌ 上下文不完整（列表/表格被切断）
  ❌ LLM 看到碎片化信息

chunk_size 大（1024-2048）:
  ✅ 上下文完整
  ❌ 向量检索噪音大（大块包含无关内容影响相似度）
  ❌ token 浪费

正确答案:
  不是选大或小，而是用 HierarchicalNodeParser：
  用小块检索（精准） + 用大块回答（完整）
  AutoMergingRetriever 自动合并
```

**项目中的实际案例**：用 SentenceSplitter(256) 切"四大满贯"列表（500 字），被切成 5 个小 Node，检索只命中奥运会和世锦赛，回答不完整。换 HierarchicalNodeParser + AutoMerging 后，自动合并回完整列表，回答完整。

---

## 三、索引与存储

### Q7：项目中用了哪些索引类型？

```
Demo 模式：
  ├── VectorStoreIndex   → 产品手册（向量语义检索）
  ├── SummaryIndex        → 会议纪要（全文摘要）
  └── VectorStoreIndex + AutoMerging → 规章制度（层级检索）

Production 模式：
  └── VectorStoreIndex + AutoMerging + ES BM25
      向量存 Milvus，关键词存 ES，层级关系存 DocStore
```

### Q8：VectorStoreIndex 和 SummaryIndex 有什么区别？

```
VectorStoreIndex:
  构建时: 每个 Node 调 Embedding 模型生成向量
  查询时: Query 也生成向量，余弦相似度 top-k 检索
  特点: 语义匹配，适合"找最相关的几条"

SummaryIndex:
  构建时: 不需要 Embedding，只存 Node 列表
  查询时: 遍历所有 Node，逐个让 LLM 判断是否相关并摘要
  特点: 全量遍历，适合"总结全文"

项目中:
  "DataFlow Pro 支持哪些数据库？" → VectorStoreIndex（精确检索）
  "上周会议的主要结论？"         → SummaryIndex（全文总结）
```

### Q9：Production 模式的存储架构是什么？

```
┌──────────────────────────────────────────────────────┐
│                 三份数据各司其职                        │
│                                                      │
│  Milvus（向量库）      ES（搜索引擎）    DocStore（KV）│
│  ├ 存 Leaf 节点       ├ 存 Leaf 节点    ├ 存全部节点  │
│  ├ embedding 向量     ├ text 分词索引    ├ parent 关系 │
│  ├ 职责: 语义检索     ├ 职责: BM25 检索  ├ 职责: 合并  │
│  └ 只存 62 个 Leaf    └ 只存 62 个 Leaf  └ 存 87 个   │
│                                                      │
│  构建时同时写入三份，重置时同时清空                      │
└──────────────────────────────────────────────────────┘
```

### Q10：为什么不用 LlamaIndex 官方的 MilvusVectorStore 包？

官方包 `llama-index-vector-stores-milvus` 0.4.x 内部混用 `MilvusClient`（新 API）和 `Collection`（ORM API），两套 API 的连接管理不兼容（alias 不同步），导致 `ConnectionNotExistException`。

项目参考 ZFAPT 主项目的做法，用 pymilvus ORM 直接封装了 `MilvusStore`，继承 `BasePydanticVectorStore`：

```python
class MilvusStore(BasePydanticVectorStore):
    def __init__(self, host, port, user, password, ...):
        connections.connect(alias=uuid_hex, ...)  # ZFAPT 的标准做法
        self._collection = Collection(...)

    def add(self, nodes): ...   # 写入
    def query(self, query): ... # 检索
```

---

## 四、检索策略

### Q11：什么是 AutoMergingRetriever？为什么需要它？

**核心思想：用小块检索，用大块回答。**

```
问题: "四大满贯有哪几个？"

不用 AutoMerging:
  检索到: [奥运会](98字) [世锦赛](87字) [巡回赛](91字)
  回答: "主要有奥运会和世锦赛" ← 不完整

用 AutoMerging:
  ① 检索叶子: [奥运会] [世锦赛] [苏迪曼] [汤姆斯]  ← 4 个小块
  ② 发现它们的 parent 是同一个 Mid 节点
  ③ 命中比例 4/5 = 80% > 阈值 25%
  ④ 合并: 用 Mid 节点(512字)替代 4 个叶子
  ⑤ LLM 看到完整列表
  回答: "奥运会、世锦赛、苏迪曼杯、汤姆斯杯、尤伯杯" ← 完整
```

### Q12：AutoMerging 在接外部向量库（Milvus）时有什么坑？

**Milvus 不存 Node 的 relationships。** AutoMerging 判断合并依赖 `node.parent_node`，但 Milvus 检索返回的 Node 只有 text 和 metadata，`relationships = {}`。

解决方案：在 Milvus Retriever 和 AutoMerging 之间加一层 `DocStoreEnrichedRetriever`，用 node_id 从 DocStore 补回 relationships：

```python
class DocStoreEnrichedRetriever(BaseRetriever):
    def _retrieve(self, query_bundle):
        results = self._base_retriever.retrieve(query_bundle)
        for nws in results:
            full_node = self._docstore.get_document(nws.node.node_id)
            nws.node.relationships = full_node.relationships  # 补回 parent/child
        return results
```

这是使用任何外部向量库 + AutoMerging 的通用方案。

### Q13：什么是 QueryFusionRetriever？RRF 融合是什么？

项目用 `QueryFusionRetriever` 融合两路检索结果：

```
路线 1: AutoMergingRetriever（Milvus 向量 + 自动合并）
路线 2: ESBm25Retriever（ES 关键词 BM25）

两路结果通过 RRF（Reciprocal Rank Fusion）合并:
  rrf_score = Σ 1/(k + rank_in_list_i)
  k = 60（常数）

RRF vs 加权融合:
  加权: score = 0.95*vec + 0.05*bm25  ← 需要调权重，分数量级不同要归一化
  RRF:  只看排名不看分数              ← 不需要调参，更鲁棒
```

### Q14：RouterQueryEngine 是怎么工作的？

```python
# 每个 QueryEngine 封装为 Tool，带名称和描述
tools = [
    QueryEngineTool(engine=policy_engine,
        metadata=ToolMetadata(name="policy", description="规章制度...")),
    QueryEngineTool(engine=product_engine,
        metadata=ToolMetadata(name="product", description="产品信息...")),
]

# Router 用 LLM 判断该用哪个 Tool
router = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(llm=llm),
    query_engine_tools=tools,
)
```

用户提问时，LLM 根据 Tool 的 description 选择最合适的 QueryEngine。等于用 LLM 做意图识别 + 路由。

### Q15：SubQuestionQueryEngine 是什么？什么场景用？

```
用户问: "对比 DataFlow 和 CloudSync 的区别"

普通 QueryEngine: 向量检索可能只命中一个产品，回答不全

SubQuestionQueryEngine:
  ① LLM 拆解: "DataFlow 的特点？" + "CloudSync 的特点？"
  ② 子问题 1 → product_tool → 检索 DataFlow 信息 → 子回答 1
  ③ 子问题 2 → product_tool → 检索 CloudSync 信息 → 子回答 2
  ④ LLM 合并两个子回答 → 完整的对比回答
```

适合跨文档、跨知识库的复杂对比问题。

---

## 五、响应合成

### Q16：ResponseSynthesizer 的几种模式有什么区别？

```
Refine（逐个精炼）:
  Node1 → LLM → 初始答案
  Node2 + 初始答案 → LLM → 精炼答案
  Node3 + 精炼答案 → LLM → 最终答案
  调用次数: N 次（N = Node 数量）
  耗时: 最慢（项目实测 7 个 Node 耗时 34 秒）
  质量: 最高

Compact（压缩一次）:
  所有 Node 文本压缩拼接 → 一次 LLM 调用 → 答案
  调用次数: 1 次
  耗时: 最快（项目实测 3.6 秒）
  质量: 较好

TreeSummarize（树形合并）:
  Node1+Node2 → LLM → 摘要1
  Node3+Node4 → LLM → 摘要2
  摘要1+摘要2 → LLM → 最终答案
  调用次数: ~N/2 次（树形递归）
  耗时: 中等（项目实测 5.5 秒）
  质量: 较好
```

**项目中默认用 Compact**，因为速度快 10 倍，质量接近 Refine。

### Q17：为什么 RRF 融合后不能用 SimilarityPostprocessor？

项目踩过的坑：

```
RRF 分数量级: ~0.01 ~ 0.03（很小）
SimilarityPostprocessor(similarity_cutoff=0.1)

结果: 所有 Node 的 RRF 分数 < 0.1，全部被过滤掉，返回 0 个 Node

原因: RRF 公式 score = 1/(60+rank)，第 1 名的分数 = 1/61 ≈ 0.016
      和余弦相似度（0~1）的量级完全不同

解决: cutoff 设为 0.0，用 top_k 自然控制数量
```

---

## 六、工程实践

### Q18：项目的整体架构是什么？

```
前端（index.html）
  │  SSE 事件流
  ▼
FastAPI 后端
  ├── file_handler    文件上传 + SimpleDirectoryReader 解析
  ├── ingestion       Document 合并 + HierarchicalNodeParser 切片
  ├── indexing         Milvus/ES/DocStore 三路入库 + QueryEngine 组装
  ├── query_engine    检索 + 合成 + StepTracer 埋点
  ├── storage         StorageFactory 工厂（demo/production 切换）
  ├── milvus_store    自定义 MilvusStore（pymilvus ORM）
  └── es_retriever    自定义 ESBm25Retriever（BaseRetriever）
```

### Q19：怎么适配通义千问（非 OpenAI 模型）？

```
问题 1: LLM
  llama-index-llms-openai 的 OpenAI 类有模型名白名单，不认 qwen-max
  解决: 用 llama-index-llms-openai-like（官方包，不校验模型名）

问题 2: Embedding
  OpenAIEmbedding 有模型名枚举校验，不认 text-embedding-v3
  解决: 继承 BaseEmbedding，直接调 OpenAI SDK

问题 3: ES 客户端版本
  ES 8.x 服务端 + elasticsearch-py v9 → Accept header 不兼容
  解决: 用 elasticsearch-py v7 + http_auth 参数
```

### Q20：这个项目能直接用在生产环境吗？还差什么？

```
已具备:
  ✅ Milvus + ES + DocStore 三路存储
  ✅ HierarchicalNodeParser + AutoMerging
  ✅ RRF 混合检索
  ✅ 多种合成模式

还需要:
  ❌ DocStore 持久化（当前内存，重启丢失）→ 改用 Redis DocStore
  ❌ 多文件管理（当前只支持单文件）→ 加文件管理和知识库概念
  ❌ 用户认证和权限
  ❌ 并发处理（当前单线程）
  ❌ 错误重试和降级
  ❌ 文档解析增强（当前 PDFReader 只能提取纯文本，没有版面分析/OCR）
     → 可以接 DeepDoc / Unstructured 等专业解析引擎
```

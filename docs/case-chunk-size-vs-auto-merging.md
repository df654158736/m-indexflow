# 案例：切片粒度 vs 检索完整性 — 从 SentenceSplitter 到 AutoMergingRetriever

## 一、问题现象

**上传文档**：羽毛球知识图谱.pdf

**用户提问**："国际顶级赛事（四大满贯级）有哪几个？"

**期望回答**：奥运会、世锦赛、苏迪曼杯、汤姆斯杯（+尤伯杯）

**实际回答**：只提到了奥运会和世锦赛，缺少汤姆斯杯和苏迪曼杯

## 二、根因分析

### 2.1 原始文档结构

```
（二）重要赛事
1. 国际顶级赛事（"四大满贯" 级）
  • 奥运会羽毛球比赛：每 4 年举办一次，是全球最高水平的……     ← 约 120 字
  • 世界羽毛球锦标赛：由世界羽毛球联合会主办，每年举办……       ← 约 100 字
  • 苏迪曼杯：世界羽毛球混合团体锦标赛，每 2 年举办……         ← 约 100 字
  • 汤姆斯杯：世界男子羽毛球团体锦标赛，每 2 年举办……         ← 约 90 字
  • 尤伯杯：世界女子羽毛球团体锦标赛，与汤姆斯杯同年……        ← 约 95 字
```

整段"四大满贯"内容约 **500+ 字符**。

### 2.2 SentenceSplitter 切片结果

使用 `SentenceSplitter(chunk_size=256, chunk_overlap=32)` 切片后：

```
Node #28 (133字): "（二）重要赛事 1. 国际顶级赛事…… • 奥运会……"   ← 标题 + 奥运会
Node #29 (100字): "• 世界羽毛球锦标赛：……"                        ← 世锦赛（单独）
Node #30 (101字): "• 苏迪曼杯：……"                                ← 苏迪曼杯（单独）
Node #31 ( 89字): "• 汤姆斯杯：……"                                ← 汤姆斯杯（单独）
Node #32 ( 95字): "• 尤伯杯：……"                                  ← 尤伯杯（单独）
```

**一个完整的列表被切成了 5 个独立的 Node。**

### 2.3 检索结果

RRF 融合检索返回 8 个 Node，其中：

| 排名 | RRF Score | 内容 |
|------|-----------|------|
| #1 | 0.0333 | 奥运会（含标题"四大满贯"） |
| #2 | 0.0328 | 其他重要赛事（世界羽联巡回赛） |
| #3 | 0.0161 | 世锦赛 |
| #4 | 0.0161 | 全英公开赛 |
| #5 | 0.0159 | 各洲锦标赛 |
| #6 | 0.0159 | "双圈大满贯" 运动员介绍 |
| #7 | 0.0156 | 李宗伟 |
| #8 | 0.0156 | 发展历程 |

**问题**：
- 汤姆斯杯（Node #31）和苏迪曼杯（Node #30）没有进入 top-8
- 即使进入了，它们是独立的 Node，LLM 合成时也不知道它们和"四大满贯"是一组的
- Node #1 只有奥运会 + 标题，没有其他赛事

### 2.4 核心矛盾

```
chunk_size 小 → 检索精准度高（小 Node 更容易匹配特定问题）
              → 但上下文完整性差（完整列表被切断）

chunk_size 大 → 上下文完整（一个 Node 包含完整列表）
              → 但检索噪音大（大 Node 包含无关内容，影响向量相似度）
```

**这是 RAG 领域最经典的 trade-off：检索精度 vs 上下文完整性。**

## 三、解决方案：HierarchicalNodeParser + AutoMergingRetriever

LlamaIndex 提供了一个优雅的解决方案，**同时兼顾两者**：

### 3.1 核心思想

```
不是"选大块还是小块"，而是"用小块检索，用大块回答"

         文档原文
            │
            ▼
  ┌─────────────────────┐
  │ HierarchicalNodeParser │
  │ 三级切片：              │
  │  Level 1: 1024 token   │  ← 章级别（大块，上下文完整）
  │  Level 2: 256 token    │  ← 节级别（中块）
  │  Level 3: 64 token     │  ← 段级别（小块，检索精准）
  └─────────┬─────────────┘
            │
            ▼
  三级 Node 树，自动建立 parent/child 关系：

  [Level 1] 重要赛事（完整）   ← 包含全部四大满贯 + 其他赛事
       │
       ├── [Level 2] 四大满贯列表  ← 包含奥运会 + 世锦赛 + 苏迪曼 + 汤尤杯
       │       │
       │       ├── [Level 3] 奥运会   ← 精确匹配"顶级赛事"
       │       ├── [Level 3] 世锦赛
       │       ├── [Level 3] 苏迪曼杯
       │       └── [Level 3] 汤姆斯杯
       │
       └── [Level 2] 其他赛事
               │
               ├── [Level 3] 世界羽联巡回赛
               └── [Level 3] 各洲锦标赛
```

### 3.2 AutoMergingRetriever 的工作原理

```
用户问："四大满贯有哪几个？"
    │
    ▼
① 用 Level 3 叶子节点做向量检索（最精准）
    命中：[奥运会] [世锦赛] [苏迪曼杯] [汤姆斯杯]
    │
    ▼
② AutoMerging 检查：同一个父节点下命中了多少子节点？
    父节点 = "四大满贯列表"
    子节点命中率 = 4/5 = 80%  ← 超过阈值（默认 50%）
    │
    ▼
③ 触发合并！用父节点替代子节点
    返回：[四大满贯列表]（完整的，包含全部赛事）
    │
    ▼
④ LLM 基于完整的父节点生成回答
    → "四大满贯级赛事包括：奥运会、世锦赛、苏迪曼杯、汤姆斯杯、尤伯杯"
```

### 3.3 对比效果

```
方案 A：SentenceSplitter(256) + VectorRetriever
─────────────────────────────────────────────
  检索到：[奥运会] [世锦赛] [各洲锦标赛] [发展历程]...
  回答："主要有奥运会和世锦赛"  ← 不完整❌

方案 B：SentenceSplitter(1024) + VectorRetriever
─────────────────────────────────────────────
  检索到：[整段赛事内容（含无关的巡回赛）]
  回答："有奥运会、世锦赛、苏迪曼杯、汤姆斯杯等" ← 完整但噪音大⚠️

方案 C：HierarchicalNodeParser + AutoMergingRetriever
─────────────────────────────────────────────
  检索到：[四大满贯列表]（自动从小块合并成大块）
  回答："四大满贯级赛事包括奥运会、世锦赛、苏迪曼杯、汤姆斯杯、尤伯杯" ← 完整且精准✅
```

## 四、技术实现

### 4.1 切片阶段

```python
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext

# 三级层级切片
hier_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[1024, 256, 64]  # 章 → 节 → 段
)
all_nodes = hier_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(all_nodes)  # 最细粒度的叶子节点

# 关键：所有层级的 Node 都要存入 docstore
docstore = SimpleDocumentStore()
docstore.add_documents(all_nodes)  # parent + child 全部存进去
```

### 4.2 索引阶段

```python
# 只用叶子节点建向量索引（检索入口）
# 但 storage_context 关联了包含所有层级 Node 的 docstore
storage_context = StorageContext.from_defaults(
    docstore=docstore,
    vector_store=milvus_store,  # production 模式用 Milvus
)
vector_index = VectorStoreIndex(
    leaf_nodes,                    # 只索引叶子节点
    embed_model=embed_model,
    storage_context=storage_context,
)
```

### 4.3 检索阶段

```python
from llama_index.core.retrievers import AutoMergingRetriever

# 基础检索器：从叶子节点检索
base_retriever = vector_index.as_retriever(similarity_top_k=6)

# AutoMerging 检索器：自动合并
auto_merging_retriever = AutoMergingRetriever(
    base_retriever,
    storage_context=storage_context,  # 通过 docstore 找到父节点
    simple_ratio_thresh=0.4,          # 40% 子节点命中就合并
)

# 查询
nodes = auto_merging_retriever.retrieve("四大满贯有哪几个？")
# → 返回合并后的父节点（包含完整的四大满贯列表）
```

### 4.4 与 ES BM25 融合

AutoMergingRetriever 可以作为 QueryFusionRetriever 的一路：

```python
from llama_index.core.retrievers import QueryFusionRetriever

fusion_retriever = QueryFusionRetriever(
    retrievers=[
        auto_merging_retriever,  # Milvus 向量 + 自动合并
        es_bm25_retriever,       # ES 关键词
    ],
    mode="reciprocal_rerank",
)
```

## 五、关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `chunk_sizes` | [1024, 256, 64] | 三级层级大小 |
| `similarity_top_k` | 6 | 叶子节点检索数量 |
| `simple_ratio_thresh` | 0.4 | 子节点命中比例超过 40% 就合并到父节点 |

### 调参建议

- `chunk_sizes` 第一级要足够大（1024-2048），确保一个完整的段落/列表能装下
- `simple_ratio_thresh` 越小越容易触发合并（0.3-0.5 比较合适）
- `similarity_top_k` 适当调大（6-10），增加子节点命中概率

## 六、总结

| 维度 | SentenceSplitter | HierarchicalNodeParser + AutoMerging |
|------|-------------------|--------------------------------------|
| 切片方式 | 固定大小，平级 | 多级层级，parent/child 关系 |
| 检索精度 | 高（小块匹配精准） | 高（用叶子节点检索） |
| 上下文完整性 | 差（列表被切断） | 好（自动合并回大块） |
| 实现复杂度 | 简单 | 中等（需要 docstore） |
| 适用场景 | 简单 QA | 列表、层级文档、长段落 |
| LlamaIndex 独有 | ❌ LangChain 也有类似 | ✅ Node Relationships 是杀手级特性 |

**核心观点**：不是"切大块好还是切小块好"，而是**用小块检索、用大块回答**。HierarchicalNodeParser + AutoMergingRetriever 同时解决了精度和完整性的矛盾。

## 七、实际落地踩过的坑

### 7.1 PDF 跨页切断

**问题**：PDF 按页解析生成多个 Document（每页一个），如果一个完整段落跨页，HierarchicalNodeParser 无法跨 Document 建立层级关系。

**解决**：切片前先合并同一来源的 Document：
```python
# 按来源分组并合并
groups = defaultdict(list)
for doc in documents:
    groups[doc.metadata['source']].append(doc)
merged = [Document(text='\n'.join(d.text for d in dl), metadata=dl[0].metadata)
          for dl in groups.values()]
# 合并后再做层级切片
all_nodes = hier_parser.get_nodes_from_documents(merged)
```

### 7.2 Milvus 不存 relationships

**问题**：AutoMergingRetriever 依赖 `node.parent_node` 判断是否合并，但 Milvus 只存 text/embedding/metadata，不存 Node 的 relationships（parent/child 关系）。导致检索返回的 Node 的 `parent_node` 为 None，永远不会触发合并。

**解决**：在 Milvus retriever 和 AutoMergingRetriever 之间加一个包装层，用 node_id 从 DocStore 补回 relationships：
```python
class DocStoreEnrichedRetriever(BaseRetriever):
    def _retrieve(self, query_bundle):
        results = self._base_retriever.retrieve(query_bundle)
        for nws in results:
            # 从 DocStore 取回完整 Node（含 relationships）
            full_node = self._docstore.get_document(nws.node.node_id)
            if full_node:
                nws.node.relationships = full_node.relationships
        return results
```

这是使用外部向量库（Milvus/Qdrant/Weaviate）+ AutoMerging 的通用方案。内存模式不需要，因为 SimpleVectorStore 的 Node 天然保留 relationships。

### 7.3 metadata 超过 chunk_size

**问题**：Level 3 的 chunk_size=64 时，metadata 长度（PDF 路径等信息约 70-110 字符）超过 chunk_size，报 `ValueError: Metadata length is longer than chunk size`。

**解决**：Level 3 至少设 256，确保 chunk_size 大于 metadata 长度。

### 7.4 合并阈值调优

**问题**：默认 `simple_ratio_thresh=0.5`（50% 子节点命中才合并）在文档内容分散时难以触发。

**解决**：降低到 0.25，并配合增大 `similarity_top_k`（8-10）来增加子节点命中数量。

## 八、最终效果对比

### 修复前（SentenceSplitter 256）
```
问：国际顶级赛事四大满贯有哪几个？
答：主要有奥运会和世锦赛 ← 不完整❌
```

### 修复后（HierarchicalNodeParser + AutoMerging + Document 合并）
```
问：国际顶级赛事四大满贯有哪几个？
答：包括奥运会、世锦赛、苏迪曼杯、汤姆斯杯与尤伯杯 ← 完整✅
```

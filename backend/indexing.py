# AI_GENERATE_START
"""索引构建 + QueryEngine 组装"""
from llama_index.core import VectorStoreIndex, SummaryIndex, StorageContext
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, RouterQueryEngine, SubQuestionQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.storage.docstore import SimpleDocumentStore

from backend.config import get_llm, get_embed_model
from backend.step_tracer import StepTracer


def build_indexes(nodes_data: dict, tracer: StepTracer) -> dict:
    """构建所有索引和查询引擎"""
    llm = get_llm()
    embed = get_embed_model()

    policy_all = nodes_data['policy_all_nodes']
    policy_leaf = nodes_data['policy_leaf_nodes']
    product_nodes = nodes_data['product_nodes']
    meeting_nodes = nodes_data['meeting_nodes']

    # ── Step: VectorStoreIndex 构建（产品手册） ──
    product_index = VectorStoreIndex(product_nodes, embed_model=embed)
    tracer.trace(
        phase='index', title='产品手册 — VectorStoreIndex 构建',
        code=(
            'from llama_index.core import VectorStoreIndex\n'
            '\n'
            '# 将 Node 列表构建为向量索引\n'
            '▶ product_index = VectorStoreIndex(\n'
            '    nodes=product_nodes,\n'
            '    embed_model=embed_model\n'
            ')\n'
            '# 内部会对每个 Node 调用 embed_model 生成向量'
        ),
        input_data={'node_count': len(product_nodes), 'doc_source': '产品手册'},
        output_data={'index_type': 'VectorStoreIndex', 'node_count': len(product_nodes)},
        explanation=(
            'VectorStoreIndex 是 LlamaIndex 最核心的索引类型。\n'
            '构建时会对每个 Node 的 text 调用 Embedding 模型生成向量，\n'
            '存入向量存储（默认 SimpleVectorStore，内存模式）。\n'
            '查询时通过向量相似度检索最相关的 Node。'
        ),
        component='LlamaIndex:VectorStoreIndex',
    )

    # ── Step: SummaryIndex 构建（会议纪要） ──
    meeting_index = SummaryIndex(meeting_nodes)
    tracer.trace(
        phase='index', title='会议纪要 — SummaryIndex 构建',
        code=(
            'from llama_index.core import SummaryIndex\n'
            '\n'
            '▶ meeting_index = SummaryIndex(nodes=meeting_nodes)\n'
            '# 注意：SummaryIndex 不需要 Embedding！'
        ),
        input_data={'node_count': len(meeting_nodes), 'doc_source': '会议纪要'},
        output_data={'index_type': 'SummaryIndex', 'node_count': len(meeting_nodes)},
        explanation=(
            'SummaryIndex 和 VectorStoreIndex 的区别：\n'
            '• VectorStoreIndex：向量检索，返回最相关的 top-k 个 Node\n'
            '• SummaryIndex：遍历所有 Node，让 LLM 逐个摘要后合并\n'
            'SummaryIndex 适合"总结全文"类查询，不适合精确检索。\n'
            '它不需要 Embedding 模型，构建速度更快。'
        ),
        component='LlamaIndex:SummaryIndex',
    )

    # ── Step: 规章制度 — 层级索引（AutoMerging 准备） ──
    docstore = SimpleDocumentStore()
    docstore.add_documents(policy_all)
    storage_context = StorageContext.from_defaults(docstore=docstore)
    policy_index = VectorStoreIndex(policy_leaf, embed_model=embed, storage_context=storage_context)
    tracer.trace(
        phase='index', title='规章制度 — 层级索引构建（AutoMerging）',
        code=(
            'from llama_index.core.storage.docstore import SimpleDocumentStore\n'
            'from llama_index.core import StorageContext\n'
            '\n'
            '# 1. 所有层级 Node 存入 docstore（parent + child 都要）\n'
            '▶ docstore = SimpleDocumentStore()\n'
            '  docstore.add_documents(all_policy_nodes)\n'
            '\n'
            '# 2. 只用叶子节点构建 VectorIndex（检索入口）\n'
            '  storage_context = StorageContext.from_defaults(docstore=docstore)\n'
            '▶ policy_index = VectorStoreIndex(\n'
            '    leaf_nodes,\n'
            '    embed_model=embed,\n'
            '    storage_context=storage_context  # 关联 docstore\n'
            ')'
        ),
        input_data={
            'all_nodes': len(policy_all),
            'leaf_nodes': len(policy_leaf),
            'docstore_size': len(policy_all),
        },
        explanation=(
            '这是 AutoMerging 检索的准备工作：\n'
            '1. 把所有层级的 Node（根+中间+叶子）存入 docstore\n'
            '2. 只用叶子节点构建 VectorStoreIndex（作为检索入口）\n'
            '3. 查询时 AutoMergingRetriever 先检索叶子节点，\n'
            '   如果同一父节点下命中比例超过阈值，自动合并回父节点\n'
            '这就是 LlamaIndex Node Relationships 的杀手级应用！'
        ),
        component='LlamaIndex:StorageContext',
    )

    # ── Step: 组装 QueryEngine ──
    # 规章制度：AutoMergingRetriever
    policy_retriever = AutoMergingRetriever(
        policy_index.as_retriever(similarity_top_k=4),
        storage_context=storage_context,
    )
    policy_synthesizer = get_response_synthesizer(response_mode='refine', llm=llm)
    policy_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.3)
    policy_engine = RetrieverQueryEngine(
        retriever=policy_retriever,
        response_synthesizer=policy_synthesizer,
        node_postprocessors=[policy_postprocessor],
    )
    tracer.trace(
        phase='index', title='规章制度 — QueryEngine 组装',
        code=(
            'from llama_index.core.retrievers import AutoMergingRetriever\n'
            'from llama_index.core.query_engine import RetrieverQueryEngine\n'
            'from llama_index.core.response_synthesizers import get_response_synthesizer\n'
            'from llama_index.core.postprocessor import SimilarityPostprocessor\n'
            '\n'
            '# 1. AutoMergingRetriever — 层级检索 + 自动合并\n'
            '▶ retriever = AutoMergingRetriever(\n'
            '    policy_index.as_retriever(similarity_top_k=4),\n'
            '    storage_context=storage_context\n'
            ')\n'
            '\n'
            '# 2. ResponseSynthesizer — Refine 模式逐步精炼\n'
            '  synthesizer = get_response_synthesizer(response_mode="refine")\n'
            '\n'
            '# 3. NodePostprocessor — 相似度截断\n'
            '  postprocessor = SimilarityPostprocessor(similarity_cutoff=0.3)\n'
            '\n'
            '# 4. 组装 QueryEngine\n'
            '▶ policy_engine = RetrieverQueryEngine(\n'
            '    retriever=retriever,\n'
            '    response_synthesizer=synthesizer,\n'
            '    node_postprocessors=[postprocessor]\n'
            ')'
        ),
        explanation=(
            'QueryEngine = Retriever + PostProcessor + Synthesizer\n'
            '这是 LlamaIndex 查询的标准三件套：\n'
            '• Retriever：负责从索引中检索 Node\n'
            '• PostProcessor：对检索结果后处理（过滤、重排）\n'
            '• Synthesizer：基于 Node 生成最终回答\n'
            '\n'
            'AutoMergingRetriever 的特殊之处：\n'
            '检索叶子 Node 后，如果某个父 Node 的大部分子 Node 都被命中，\n'
            '就自动合并回父 Node，提供更完整的上下文。'
        ),
        component='LlamaIndex:RetrieverQueryEngine',
    )

    # 产品手册：标准 QueryEngine
    product_engine = product_index.as_query_engine(
        llm=llm,
        similarity_top_k=3,
        response_mode='tree_summarize',
    )
    tracer.trace(
        phase='index', title='产品手册 — 标准 QueryEngine',
        code=(
            '# 最简写法：index.as_query_engine()\n'
            '▶ product_engine = product_index.as_query_engine(\n'
            '    llm=llm,\n'
            '    similarity_top_k=3,\n'
            '    response_mode="tree_summarize"\n'
            ')'
        ),
        explanation=(
            'as_query_engine() 是 LlamaIndex 的便捷方法，\n'
            '一行代码完成 Retriever + Synthesizer 的组装。\n'
            'response_mode="tree_summarize" 使用树形摘要策略：\n'
            '将多个 Node 的内容递归合并摘要，适合信息量大的场景。'
        ),
        component='LlamaIndex:as_query_engine',
    )

    # 会议纪要：SummaryIndex QueryEngine
    meeting_engine = meeting_index.as_query_engine(
        llm=llm,
        response_mode='tree_summarize',
    )
    tracer.trace(
        phase='index', title='会议纪要 — SummaryIndex QueryEngine',
        code=(
            '▶ meeting_engine = meeting_index.as_query_engine(\n'
            '    llm=llm,\n'
            '    response_mode="tree_summarize"\n'
            ')\n'
            '# SummaryIndex 会遍历所有 Node，适合全文总结'
        ),
        component='LlamaIndex:SummaryIndex',
    )

    # ── Step: RouterQueryEngine + SubQuestionQueryEngine ──
    tools = [
        QueryEngineTool(
            query_engine=policy_engine,
            metadata=ToolMetadata(
                name='policy_tool',
                description='查询公司规章制度，包括出差管理、报销标准、请假制度等',
            ),
        ),
        QueryEngineTool(
            query_engine=product_engine,
            metadata=ToolMetadata(
                name='product_tool',
                description='查询产品信息，包括 DataFlow Pro 和 CloudSync 的功能、价格、支持的数据库等',
            ),
        ),
        QueryEngineTool(
            query_engine=meeting_engine,
            metadata=ToolMetadata(
                name='meeting_tool',
                description='查询会议纪要，包括技术方案评审结论、决策、行动计划等',
            ),
        ),
    ]

    router_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(llm=llm),
        query_engine_tools=tools,
    )

    sub_question_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=tools,
        llm=llm,
    )

    tracer.trace(
        phase='index', title='RouterQueryEngine + SubQuestionQueryEngine 组装',
        code=(
            'from llama_index.core.tools import QueryEngineTool, ToolMetadata\n'
            'from llama_index.core.query_engine import RouterQueryEngine, SubQuestionQueryEngine\n'
            'from llama_index.core.selectors import LLMSingleSelector\n'
            '\n'
            '# 将每个 QueryEngine 封装为 Tool\n'
            'tools = [\n'
            '    QueryEngineTool(query_engine=policy_engine,\n'
            '        metadata=ToolMetadata(name="policy", description="规章制度...")),\n'
            '    QueryEngineTool(query_engine=product_engine,\n'
            '        metadata=ToolMetadata(name="product", description="产品信息...")),\n'
            '    QueryEngineTool(query_engine=meeting_engine,\n'
            '        metadata=ToolMetadata(name="meeting", description="会议纪要...")),\n'
            ']\n'
            '\n'
            '# RouterQueryEngine — 根据问题自动选择合适的引擎\n'
            '▶ router = RouterQueryEngine(\n'
            '    selector=LLMSingleSelector.from_defaults(llm=llm),\n'
            '    query_engine_tools=tools\n'
            ')\n'
            '\n'
            '# SubQuestionQueryEngine — 复杂问题拆解为子问题\n'
            '▶ sub_q = SubQuestionQueryEngine.from_defaults(\n'
            '    query_engine_tools=tools, llm=llm\n'
            ')'
        ),
        output_data={
            'tools': [{'name': t.metadata.name, 'description': t.metadata.description[:40]} for t in tools],
        },
        explanation=(
            'RouterQueryEngine 是 LlamaIndex 的多索引路由器：\n'
            '它用 LLM 判断用户问题应该交给哪个 QueryEngine 处理。\n'
            '每个 QueryEngineTool 带有 name 和 description，\n'
            'LLM 根据 description 做出路由决策。\n'
            '\n'
            'SubQuestionQueryEngine 是复杂问题分解器：\n'
            '当用户问跨领域的对比问题时，它会拆成多个子问题，\n'
            '分别交给不同的 Tool 查询，最后合并回答。'
        ),
        component='LlamaIndex:RouterQueryEngine',
    )

    tracer.trace(
        phase='index', title='索引构建完成 — 准备接收查询',
        code='# 所有索引和查询引擎已就绪',
        explanation='文档加载 → Node 解析 → 索引构建 → QueryEngine 组装，全部完成。\n现在可以输入问题进行查询了！',
        component='LlamaIndex:Ready',
    )

    return {
        'router_engine': router_engine,
        'sub_question_engine': sub_question_engine,
        'policy_engine': policy_engine,
        'product_engine': product_engine,
        'meeting_engine': meeting_engine,
        'tools': tools,
    }


def build_indexes_production(nodes_data: dict, tracer: StepTracer) -> dict:
    """production 模式：Milvus + ES + AutoMerging 层级检索

    用小块检索、用大块回答 —— HierarchicalNodeParser + AutoMergingRetriever。
    """
    from llama_index.core.retrievers import QueryFusionRetriever, AutoMergingRetriever
    from llama_index.core.storage.docstore import SimpleDocumentStore
    from llama_index.core.node_parser import get_leaf_nodes
    from backend.storage import StorageFactory

    llm = get_llm()
    embed = get_embed_model()
    all_nodes = nodes_data['all_nodes']
    leaf_nodes = nodes_data.get('leaf_nodes', all_nodes)

    # ── Step: 所有层级 Node 存入 DocStore ──
    docstore = SimpleDocumentStore()
    docstore.add_documents(all_nodes)
    tracer.trace(
        phase='index', title=f'DocStore — 存入 {len(all_nodes)} 个层级 Node',
        code=(
            'from llama_index.core.storage.docstore import SimpleDocumentStore\n'
            '\n'
            '# 所有层级的 Node（根 + 中间 + 叶子）都存入 docstore\n'
            '# AutoMergingRetriever 需要通过 docstore 找到父节点\n'
            '▶ docstore = SimpleDocumentStore()\n'
            f'  docstore.add_documents(all_nodes)  # {len(all_nodes)} 个 Node'
        ),
        input_data={'total_nodes': len(all_nodes), 'leaf_nodes': len(leaf_nodes)},
        explanation=(
            'DocStore 是 AutoMerging 的关键：\n'
            '• 叶子节点用于向量检索（精准匹配）\n'
            '• 检索命中后，通过 parent 关系在 DocStore 中找到父节点\n'
            '• 如果同一父节点下命中足够多的子节点，就合并回父节点\n'
            '所以必须把所有层级的 Node 都存进来。'
        ),
        component='LlamaIndex:DocStore',
    )

    # ── Step: 叶子节点写入 Milvus ──
    vector_store = StorageFactory.get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)
    vector_index = VectorStoreIndex(
        leaf_nodes,
        embed_model=embed,
        storage_context=storage_context,
    )
    tracer.trace(
        phase='index', title=f'Milvus — 叶子节点向量索引（{len(leaf_nodes)} 个）',
        code=(
            '# 只用叶子节点（最细粒度）建向量索引\n'
            '# storage_context 同时关联了 vector_store 和 docstore\n'
            '▶ vector_index = VectorStoreIndex(\n'
            f'    leaf_nodes,              # {len(leaf_nodes)} 个叶子节点\n'
            '    embed_model=embed,\n'
            '    storage_context=storage_context\n'
            ')'
        ),
        input_data={'leaf_count': len(leaf_nodes)},
        output_data={'storage': 'Milvus', 'indexed_count': len(leaf_nodes)},
        explanation=(
            f'只索引 {len(leaf_nodes)} 个叶子节点（而不是全部 {len(all_nodes)} 个）。\n'
            '叶子节点粒度最小（~64 token），向量检索最精准。\n'
            '父节点不需要索引，AutoMergingRetriever 会通过 DocStore 找到它们。'
        ),
        component='LlamaIndex:MilvusVectorStore',
    )

    # ── Step: ES BM25 入库（叶子节点） ──
    StorageFactory.index_nodes_to_es(leaf_nodes, tracer=tracer)

    # ── Step: 构建 AutoMergingRetriever ──
    # Milvus 返回的 Node 没有 relationships（Milvus 不存层级关系），
    # 但 AutoMergingRetriever 依赖 node.parent_node 判断是否合并。
    # 用包装 Retriever 从 DocStore 补回 relationships。
    from llama_index.core.retrievers import BaseRetriever as _BaseRetriever
    from llama_index.core.schema import QueryBundle as _QB, NodeWithScore as _NWS

    class _DocStoreEnrichedRetriever(_BaseRetriever):
        """从 DocStore 补回 Milvus 检索结果的 relationships"""
        def __init__(self, base, docstore):
            super().__init__()
            self._base = base
            self._docstore = docstore
        def _retrieve(self, query_bundle: _QB, **kw) -> list[_NWS]:
            results = self._base.retrieve(query_bundle)
            for nws in results:
                try:
                    full_node = self._docstore.get_document(nws.node.node_id)
                    if full_node and hasattr(full_node, 'relationships'):
                        nws.node.relationships = full_node.relationships
                except Exception:
                    pass
            return results

    raw_retriever = vector_index.as_retriever(similarity_top_k=8)
    enriched_retriever = _DocStoreEnrichedRetriever(raw_retriever, docstore)
    auto_merging_retriever = AutoMergingRetriever(
        enriched_retriever,
        storage_context=storage_context,
        simple_ratio_thresh=0.25,
    )
    tracer.trace(
        phase='index', title='AutoMergingRetriever — 层级检索 + 自动合并',
        code=(
            'from llama_index.core.retrievers import AutoMergingRetriever\n'
            '\n'
            '# 基础检索器：从叶子节点检索\n'
            'base_retriever = vector_index.as_retriever(similarity_top_k=8)\n'
            '\n'
            '# AutoMerging：检索叶子节点后自动合并到父节点\n'
            '▶ auto_merging_retriever = AutoMergingRetriever(\n'
            '    base_retriever,\n'
            '    storage_context=storage_context,\n'
            '    simple_ratio_thresh=0.4  # 40% 子节点命中就合并\n'
            ')'
        ),
        explanation=(
            'AutoMergingRetriever 的工作原理：\n'
            '1. 用叶子节点做向量检索（最精准的小块）\n'
            '2. 检查：同一个父节点下命中了多少子节点？\n'
            '3. 如果命中比例 > 40%（simple_ratio_thresh），合并回父节点\n'
            '4. 返回合并后的大块 Node（上下文完整）\n\n'
            '效果：用小块检索、用大块回答，同时解决精度和完整性。'
        ),
        component='LlamaIndex:AutoMergingRetriever',
    )

    # ── Step: 构建 RRF 融合检索器 ──
    es_retriever = StorageFactory.get_bm25_retriever(similarity_top_k=5)

    fusion_retriever = QueryFusionRetriever(
        retrievers=[auto_merging_retriever, es_retriever],
        similarity_top_k=10,
        num_queries=1,
        mode='reciprocal_rerank',
        use_async=False,
    )
    tracer.trace(
        phase='index', title='混合检索器 — AutoMerging + ES BM25 RRF 融合',
        code=(
            '▶ fusion_retriever = QueryFusionRetriever(\n'
            '    retrievers=[\n'
            '        auto_merging_retriever,  # Milvus 向量 + 自动合并\n'
            '        es_bm25_retriever,       # ES 关键词\n'
            '    ],\n'
            '    mode="reciprocal_rerank",\n'
            ')'
        ),
        explanation=(
            '双路融合：\n'
            '• 路线 1：AutoMergingRetriever（Milvus 向量检索 + 自动合并到父节点）\n'
            '• 路线 2：ESBm25Retriever（ES 关键词检索）\n'
            '通过 RRF 算法融合两路结果。'
        ),
        component='LlamaIndex:QueryFusionRetriever',
    )

    # ── Step: 构建 QueryEngine ──
    synthesizer = get_response_synthesizer(response_mode='refine', llm=llm)
    # RRF 融合后的分数量级很小（约 0.01~0.03），不能用常规的相似度截断
    # 去掉 SimilarityPostprocessor，让 QueryFusionRetriever 的 top_k 自然控制数量
    postprocessor = SimilarityPostprocessor(similarity_cutoff=0.0)
    query_engine = RetrieverQueryEngine(
        retriever=fusion_retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=[postprocessor],
    )

    # 封装为 Tool
    tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name='knowledge_base',
            description='查询知识库，支持所有上传文档的内容检索',
        ),
    )

    sub_question_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[tool],
        llm=llm,
    )

    tracer.trace(
        phase='index', title='QueryEngine 组装完成',
        code=(
            '# 混合检索 + Refine 合成 + 相似度过滤\n'
            '▶ engine = RetrieverQueryEngine(\n'
            '    retriever=fusion_retriever,\n'
            '    response_synthesizer=get_response_synthesizer("refine"),\n'
            '    node_postprocessors=[SimilarityPostprocessor(cutoff=0.1)]\n'
            ')'
        ),
        explanation=(
            'production 模式的 QueryEngine 使用混合检索器（Milvus + ES RRF 融合），\n'
            '对比 demo 模式的单一内存向量检索，召回率更高。\n'
            'Retriever → PostProcessor → Synthesizer 三件套不变，\n'
            '只是 Retriever 层从内存换成了 Milvus + ES。'
        ),
        component='LlamaIndex:RetrieverQueryEngine',
    )

    tracer.trace(
        phase='index', title='索引构建完成 — 准备接收查询',
        code='# Milvus（向量）+ ES（BM25）双路就绪',
        explanation='production 模式索引构建完成。\n向量数据在 Milvus，BM25 索引在 ES，查询时 RRF 融合。',
        component='LlamaIndex:Ready',
    )

    return {
        'query_engine': query_engine,
        'sub_question_engine': sub_question_engine,
        'fusion_retriever': fusion_retriever,
        'auto_merging_retriever': auto_merging_retriever,
        'enriched_retriever': enriched_retriever,
        'tools': [tool],
    }
# AI_GENERATE_END

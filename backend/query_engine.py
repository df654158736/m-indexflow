# AI_GENERATE_START
"""查询执行 — 带 StepTracer 埋点"""
from backend.step_tracer import StepTracer


def execute_query(query: str, mode: str, engines: dict, tracer: StepTracer) -> str:
    """执行查询并记录每一步"""

    if mode == 'sub_question':
        return _run_sub_question(query, engines, tracer)
    else:
        return _run_router(query, engines, tracer)


def _run_router(query: str, engines: dict, tracer: StepTracer) -> str:
    """RouterQueryEngine 查询"""
    router = engines['router_engine']

    tracer.trace(
        phase='query', title='RouterQueryEngine — 开始路由',
        code=(
            '# Router 用 LLM 判断该使用哪个 QueryEngine\n'
            '▶ response = router_engine.query(query)\n'
            '\n'
            '# 内部流程：\n'
            '# 1. LLMSingleSelector 分析问题 → 选择 Tool\n'
            '# 2. 调用选中 Tool 的 QueryEngine\n'
            '# 3. QueryEngine 内部：Retrieve → PostProcess → Synthesize'
        ),
        input_data={'query': query, 'mode': 'router'},
        explanation=(
            'RouterQueryEngine 的执行流程：\n'
            '1. 把所有 Tool 的 name + description 发给 LLM\n'
            '2. LLM 返回最匹配的 Tool\n'
            '3. 调用该 Tool 对应的 QueryEngine.query()\n'
            '接下来看 LLM 选择了哪个引擎...'
        ),
        component='LlamaIndex:RouterQueryEngine',
    )

    response = router.query(query)

    # 提取路由信息和检索结果
    source_nodes = response.source_nodes if response.source_nodes else []
    node_info = []
    for sn in source_nodes:
        node_info.append({
            'id': sn.node_id[:8] if sn.node_id else 'N/A',
            'score': round(sn.score, 4) if sn.score else None,
            'text': sn.text[:80] + '...' if len(sn.text) > 80 else sn.text,
            'metadata': dict(sn.metadata) if sn.metadata else {},
        })

    tracer.trace(
        phase='query', title='检索完成 — 返回相关 Node',
        code=(
            '# 检索到的 Node（source_nodes）\n'
            'source_nodes = response.source_nodes\n'
            '▶ for node in source_nodes:\n'
            '    print(node.score, node.text[:50])'
        ),
        output_data={
            'retrieved_count': len(source_nodes),
            'nodes': node_info[:5],
        },
        explanation=(
            f'检索返回了 {len(source_nodes)} 个相关 Node。\n'
            '每个 Node 带有相似度分数 (score)，分数越高越相关。\n'
            'source_nodes 是 LlamaIndex 返回结果的标准字段，\n'
            '可用于做引文溯源和结果展示。'
        ),
        component='LlamaIndex:Retriever',
    )

    answer = str(response)

    tracer.trace(
        phase='query', title='ResponseSynthesizer — 生成回答',
        code=(
            '# Synthesizer 基于检索到的 Node 生成回答\n'
            '▶ answer = str(response)\n'
            '\n'
            '# Refine 模式：逐个 Node 精炼答案\n'
            '# TreeSummarize 模式：树形递归摘要合并'
        ),
        output_data={'answer': answer[:300] + ('...' if len(answer) > 300 else '')},
        explanation=(
            'ResponseSynthesizer 是 RAG 的最后一步：\n'
            '• Refine：先用第 1 个 Node 生成初始答案，再用后续 Node 逐步精炼\n'
            '• TreeSummarize：将所有 Node 两两合并摘要，递归直到得到最终答案\n'
            '• Compact：先压缩 Node 文本，再一次性生成答案'
        ),
        component='LlamaIndex:ResponseSynthesizer',
    )

    return answer


def _run_sub_question(query: str, engines: dict, tracer: StepTracer) -> str:
    """SubQuestionQueryEngine 查询"""
    sub_q_engine = engines['sub_question_engine']

    tracer.trace(
        phase='query', title='SubQuestionQueryEngine — 问题分解',
        code=(
            '# SubQuestion 先让 LLM 分解复杂问题\n'
            '▶ response = sub_question_engine.query(query)\n'
            '\n'
            '# 内部流程：\n'
            '# 1. LLM 分析问题 → 拆成多个子问题\n'
            '# 2. 每个子问题分配给对应的 Tool\n'
            '# 3. 收集所有子回答 → LLM 合并为最终回答'
        ),
        input_data={'query': query, 'mode': 'sub_question'},
        explanation=(
            'SubQuestionQueryEngine 的执行流程：\n'
            '1. LLM 分析原始问题，拆成多个子问题\n'
            '   如 "对比产品X和Y" → "产品X的特点？" + "产品Y的特点？"\n'
            '2. 每个子问题根据内容分配给最合适的 QueryEngineTool\n'
            '3. 各 Tool 独立查询返回子回答\n'
            '4. LLM 合并所有子回答生成最终回答\n'
            '这是 LlamaIndex 处理跨文档复杂问题的招牌能力。'
        ),
        component='LlamaIndex:SubQuestionQueryEngine',
    )

    response = sub_q_engine.query(query)
    answer = str(response)

    # 提取子问题信息
    source_nodes = response.source_nodes if response.source_nodes else []

    tracer.trace(
        phase='query', title='子问题查询完成 — 合并回答',
        code=(
            '# 所有子问题查询完毕，合并回答\n'
            '▶ answer = str(response)\n'
            '  source_nodes = response.source_nodes'
        ),
        output_data={
            'answer': answer[:300] + ('...' if len(answer) > 300 else ''),
            'source_count': len(source_nodes),
        },
        explanation=(
            f'SubQuestionQueryEngine 完成了所有子问题的查询和合并。\n'
            f'共使用了 {len(source_nodes)} 个 source node 来生成最终回答。'
        ),
        component='LlamaIndex:SubQuestionQueryEngine',
    )

    return answer
# AI_GENERATE_END

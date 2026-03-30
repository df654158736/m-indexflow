# AI_GENERATE_START
"""FastAPI 入口 — 支持 demo / production 双模式"""
import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Body, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config
from backend.step_tracer import StepTracer
from backend.ingestion import run_ingestion, run_ingestion_from_files
from backend.indexing import build_indexes, build_indexes_production
from backend.query_engine import execute_query
from backend.storage import StorageFactory

logger = logging.getLogger(__name__)


class AppState:
    """全局状态"""
    def __init__(self):
        self.tracer = StepTracer()
        self.nodes_data = None
        self.engines = None
        self.node_tree = None
        self.initialized = False
        self.connectivity = {}


state = AppState()


@asynccontextmanager
async def lifespan(app):
    # 启动时检测连通性
    state.connectivity = config.check_production_connectivity()
    logger.info(f'启动模式: {config.MODE}, 连通性: {state.connectivity}')
    yield


app = FastAPI(title='LlamaIndex RAG 管线可视化 — m-indexflow', lifespan=lifespan)

static_dir = Path(__file__).resolve().parent.parent / 'static'
app.mount('/static', StaticFiles(directory=str(static_dir)), name='static')


@app.get('/')
async def index():
    return FileResponse(str(static_dir / 'index.html'))


@app.get('/api/mode')
async def get_mode():
    """返回当前模式和连接状态"""
    return JSONResponse({
        'mode': config.MODE,
        'connectivity': state.connectivity,
    })


@app.post('/api/upload')
async def upload_file(file: UploadFile = File(...)):
    """上传文件（production 模式）"""
    if not file.filename:
        raise HTTPException(400, '未选择文件')

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(413, '文件大小不能超过 50MB')

    try:
        from backend.file_handler import save_and_parse, clear_uploads
        # 每次上传清空之前的文件，只保留当前这一个
        clear_uploads()
        result = save_and_parse(file.filename, content)
        return JSONResponse(result)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception('文件上传处理失败')
        raise HTTPException(500, f'文件处理失败: {str(e)[:100]}')


@app.post('/api/run')
async def run_ingestion_and_indexing():
    """执行文档加载 + Node 解析 + 索引构建，SSE 返回步骤"""
    state.tracer.reset()
    state.initialized = False

    async def sse():
        loop = asyncio.get_event_loop()
        asyncio.ensure_future(loop.run_in_executor(None, _build_all))
        async for event in state.tracer.get_events():
            yield event

    return StreamingResponse(sse(), media_type='text/event-stream')


def _force_clean_storage():
    """强制清空 Milvus collection 和 ES index，不依赖单例状态"""
    from backend.config import MILVUS_HOST, MILVUS_PORT, MILVUS_USER, MILVUS_PASSWORD, MILVUS_COLLECTION, ES_URL, ES_INDEX
    col_name = f'm_indexflow_{MILVUS_COLLECTION}'
    idx_name = f'm_indexflow_{ES_INDEX}'

    # 清 Milvus
    try:
        from pymilvus import connections, utility
        alias = 'clean_' + __import__('uuid').uuid4().hex[:8]
        connections.connect(alias=alias, host=MILVUS_HOST, port=str(MILVUS_PORT), user=MILVUS_USER, password=MILVUS_PASSWORD)
        if utility.has_collection(col_name, using=alias):
            utility.drop_collection(col_name, using=alias)
            logger.info(f'Milvus collection 已清空: {col_name}')
        connections.disconnect(alias)
    except Exception as e:
        logger.error(f'Milvus 清理失败: {e}')

    # 清 ES
    try:
        from backend.config import _create_es_client
        es = _create_es_client()
        if es.indices.exists(index=idx_name):
            es.indices.delete(index=idx_name)
            logger.info(f'ES index 已清空: {idx_name}')
    except Exception as e:
        logger.error(f'ES 清理失败: {e}')


def _build_all():
    """同步执行全部构建"""
    try:
        if StorageFactory.is_production():
            # 构建前强制清空 Milvus + ES + 上传缓存
            _force_clean_storage()
            StorageFactory.reset()

            from backend.file_handler import get_all_documents
            documents = get_all_documents()
            if not documents:
                state.tracer.trace(
                    phase='error', title='未上传文件',
                    code='# 请先上传文件再构建索引',
                    component='Error',
                )
                state.tracer.finish()
                return
            state.nodes_data = run_ingestion_from_files(documents, state.tracer)
            state.node_tree = state.nodes_data.get('node_tree', [])
            state.engines = build_indexes_production(state.nodes_data, state.tracer)
        else:
            # demo 模式：使用内置文档
            state.nodes_data = run_ingestion(state.tracer)
            state.node_tree = state.nodes_data.get('node_tree', [])
            state.engines = build_indexes(state.nodes_data, state.tracer)

        state.initialized = True
        state.tracer.finish()
    except Exception as e:
        logger.exception('构建失败')
        state.tracer.trace(
            phase='error', title=f'构建失败: {str(e)[:100]}',
            code=str(e),
            component='Error',
        )
        state.tracer.finish()


@app.post('/api/query')
async def run_query(query: str = Body(...), mode: str = Body('router'), synth_mode: str = Body('refine')):
    """执行查询，SSE 返回步骤

    Args:
        query: 查询内容
        mode: 查询模式 router | sub_question
        synth_mode: 合成模式 refine | compact | tree_summarize
    """
    if not state.initialized or not state.engines:
        return JSONResponse({'error': '请先构建索引'}, status_code=400)

    state.tracer.reset()

    async def sse():
        loop = asyncio.get_event_loop()
        asyncio.ensure_future(loop.run_in_executor(None, _query, query, mode, synth_mode))
        async for event in state.tracer.get_events():
            yield event

    return StreamingResponse(sse(), media_type='text/event-stream')


def _query(query: str, mode: str, synth_mode: str = 'refine'):
    """同步执行查询"""
    try:
        if StorageFactory.is_production():
            _query_production(query, mode, synth_mode)
        else:
            execute_query(query, mode, state.engines, state.tracer)
        state.tracer.finish()
    except Exception as e:
        logger.exception('查询失败')
        state.tracer.trace(
            phase='error', title=f'查询失败: {str(e)[:100]}',
            code=str(e),
            component='Error',
        )
        state.tracer.finish()


def _query_production(query: str, mode: str, synth_mode: str = 'refine'):
    """production 模式查询 — 拆分每个子步骤并计时"""
    import time
    tracer = state.tracer

    if mode == 'sub_question':
        # SubQuestion 模式仍然用 engine.query 整体调用
        engine = state.engines['sub_question_engine']
        tracer.trace(
            phase='query', title='SubQuestionQueryEngine — 问题分解',
            code='▶ response = sub_question_engine.query(query)',
            input_data={'query': query, 'mode': 'sub_question'},
            explanation='SubQuestionQueryEngine 拆解复杂问题，\n每个子问题通过混合检索（Milvus + ES RRF 融合）查询。',
            component='LlamaIndex:SubQuestionQueryEngine',
        )
        t0 = time.time()
        response = engine.query(query)
        elapsed = int((time.time() - t0) * 1000)
        source_nodes = response.source_nodes or []
        answer = str(response)
        tracer.trace(
            phase='query', title=f'子问题查询完成 — {len(source_nodes)} 个 Node',
            code='answer = str(response)',
            output_data={'answer': answer[:300], 'source_count': len(source_nodes)},
            elapsed_ms=elapsed,
            component='LlamaIndex:SubQuestionQueryEngine',
        )
        return

    # ── Router/Hybrid 模式：拆分为 5 个细粒度步骤 ──
    fusion_retriever = state.engines.get('fusion_retriever')
    if not fusion_retriever:
        # demo 模式回退
        engine = state.engines.get('query_engine') or state.engines.get('router_engine')
        t0 = time.time()
        response = engine.query(query)
        elapsed = int((time.time() - t0) * 1000)
        answer = str(response)
        tracer.trace(phase='query', title='查询完成', code='response = engine.query(query)',
                     output_data={'answer': answer[:300]}, elapsed_ms=elapsed, component='LlamaIndex:QueryEngine')
        return

    # ── Step 1: Milvus 叶子节点检索 ──
    auto_merging_retriever = state.engines.get('auto_merging_retriever')
    enriched_retriever = state.engines.get('enriched_retriever')

    t0 = time.time()
    if enriched_retriever:
        leaf_results = enriched_retriever.retrieve(query)
    else:
        leaf_results = []
    elapsed_leaf = int((time.time() - t0) * 1000)

    leaf_info = [{'rank': i+1, 'node_id': n.node_id[:12], 'text_length': len(n.text),
                  'has_parent': hasattr(n.node, 'parent_node') and n.node.parent_node is not None,
                  'parent_id': n.node.parent_node.node_id[:12] if (hasattr(n.node, 'parent_node') and n.node.parent_node) else None,
                  'text': n.text[:120] + '...'} for i, n in enumerate(leaf_results)]

    tracer.trace(
        phase='query', title=f'Milvus 叶子节点检索 — {len(leaf_results)} 个',
        code=(
            '# 从 Milvus 检索叶子节点（最细粒度 ~256 token）\n'
            '▶ leaf_results = enriched_retriever.retrieve(query)\n'
            '# enriched_retriever 会从 DocStore 补回 parent/child 关系'
        ),
        input_data={'query': query},
        output_data={'count': len(leaf_results), 'nodes': leaf_info},
        elapsed_ms=elapsed_leaf,
        explanation=(
            f'向量检索返回 {len(leaf_results)} 个叶子节点。\n'
            '每个节点已从 DocStore 补回了 parent 关系（has_parent / parent_id 字段）。\n'
            '接下来 AutoMergingRetriever 会检查这些叶子的 parent 关系，决定是否合并。'
        ),
        component='LlamaIndex:MilvusRetriever',
    )

    # ── Step 2: AutoMerging 合并判断 ──
    t0 = time.time()
    if auto_merging_retriever:
        merged_results = auto_merging_retriever.retrieve(query)
    else:
        merged_results = leaf_results
    elapsed_merge = int((time.time() - t0) * 1000)

    # 对比叶子和合并后的差异
    leaf_ids = {n.node_id for n in leaf_results}
    merged_ids = {n.node_id for n in merged_results}
    removed_leaves = leaf_ids - merged_ids  # 被合并掉的叶子
    new_parents = merged_ids - leaf_ids      # 新增的 parent 节点

    merge_detail = {
        'before_count': len(leaf_results),
        'after_count': len(merged_results),
        'merged_leaf_count': len(removed_leaves),
        'new_parent_count': len(new_parents),
        'merge_triggered': len(new_parents) > 0,
    }
    # 记录每个新增 parent 吃掉了哪些叶子
    if new_parents:
        parent_details = []
        for pid in new_parents:
            parent_node = next((n for n in merged_results if n.node_id == pid), None)
            if parent_node:
                # 找出这个 parent 下有哪些原始叶子被命中
                eaten_leaves = [lid[:12] for lid in removed_leaves
                                if any(n.node_id == lid and hasattr(n.node, 'parent_node') and n.node.parent_node and n.node.parent_node.node_id == pid for n in leaf_results)]
                parent_details.append({
                    'parent_id': pid[:12],
                    'parent_text_length': len(parent_node.text),
                    'parent_text_preview': parent_node.text[:200] + '...',
                    'absorbed_leaf_count': len(eaten_leaves),
                    'absorbed_leaf_ids': eaten_leaves,
                })
        merge_detail['parent_details'] = parent_details

    tracer.trace(
        phase='query', title=f'AutoMerging — {"触发合并 ✅" if new_parents else "未触发合并"}',
        code=(
            '# AutoMergingRetriever 内部逻辑:\n'
            '# 1. 按 parent_id 分组叶子节点\n'
            '# 2. 计算每个 parent 下命中比例 = 命中数 / 总子节点数\n'
            '# 3. 比例 > 0.25（阈值）→ 用 parent 替代子节点\n'
            '▶ merged_results = auto_merging_retriever.retrieve(query)'
        ),
        output_data=merge_detail,
        elapsed_ms=elapsed_merge,
        explanation=(
            f'合并前: {len(leaf_results)} 个叶子节点\n'
            f'合并后: {len(merged_results)} 个节点\n'
            f'{"🔀 " + str(len(removed_leaves)) + " 个叶子被合并成 " + str(len(new_parents)) + " 个 parent 节点" if new_parents else "ℹ️ 没有叶子满足合并条件（同一 parent 下命中比例 < 25%）"}\n\n'
            + ('合并意味着 LLM 会看到更完整的上下文，而不是碎片化的小块。' if new_parents else '未合并时 LLM 看到的是原始叶子节点。')
        ),
        component='LlamaIndex:AutoMergingRetriever',
    )

    # ── Step 3: RRF 融合（AutoMerging 结果 + ES BM25） ──
    t0 = time.time()
    fused_results = fusion_retriever.retrieve(query)
    elapsed_rrf = int((time.time() - t0) * 1000)

    fused_info = []
    for i, n in enumerate(fused_results):
        text_len = len(n.text) if n.text else 0
        is_merged = n.node_id in new_parents or text_len > 300
        node_detail = {
            'rank': i + 1,
            'node_id': n.node_id[:12] if n.node_id else 'N/A',
            'score': round(n.score, 6) if n.score else None,
            'text_length': text_len,
            'is_merged': is_merged,
            'text': n.text,
            'metadata': {},
        }
        if hasattr(n, 'metadata') and n.metadata:
            node_detail['metadata'] = {k: str(v)[:50] for k, v in n.metadata.items() if k in ('source', 'file_name', 'doc_type', 'file_id')}
        fused_info.append(node_detail)

    tracer.trace(
        phase='query', title=f'RRF 融合完成 — {len(fused_results)} 个 Node',
        code=(
            '# AutoMerging 后的结果 + ES BM25 结果 → RRF 融合\n'
            '▶ fused = fusion_retriever.retrieve(query)'
        ),
        input_data={'query': query},
        output_data={'count': len(fused_results), 'nodes': fused_info},
        elapsed_ms=elapsed_rrf,
        explanation=f'RRF 融合耗时 {elapsed_rrf}ms，最终 {len(fused_results)} 个 Node 送入 LLM。',
        component='LlamaIndex:QueryFusionRetriever',
    )

    # ── Step 2: LLM 生成回答（ResponseSynthesizer） ──
    synth_labels = {
        'refine': ('Refine', '逐个 Node 调用 LLM 精炼答案', f'约 {len(fused_results)} 次 LLM 调用'),
        'compact': ('Compact', '先压缩 Node 文本到一个 prompt，再一次调用 LLM', '约 1 次 LLM 调用'),
        'tree_summarize': ('TreeSummarize', '树形递归合并摘要', f'约 {max(1, len(fused_results)//2)} 次 LLM 调用'),
    }
    label, desc, call_est = synth_labels.get(synth_mode, ('Refine', '逐个精炼', ''))

    # 构造实际送给 LLM 的 context（模拟 synthesizer 内部拼接）
    context_texts = []
    for i, n in enumerate(fused_results):
        context_texts.append(f'[Node {i+1}] {n.text}')
    llm_context = '\n\n'.join(context_texts)

    tracer.trace(
        phase='query', title='LLM 实际输入 — Prompt 构造',
        code=(
            f'# synthesizer 内部将 {len(fused_results)} 个 Node 拼成 context\n'
            f'# 然后构造 prompt: "根据以下内容回答问题: {{context}}\\n问题: {{query}}"'
        ),
        input_data={
            'query': query,
            'synth_mode': synth_mode,
            'node_count': len(fused_results),
            'context_total_chars': len(llm_context),
            'llm_request': {
                'system': f'You are an expert Q&A system. Answer based on the context below.',
                'context': llm_context,
                'question': query,
            }
        },
        explanation=(
            f'{synth_mode} 模式下，synthesizer 将 {len(fused_results)} 个 Node 的文本拼接为 context（共 {len(llm_context)} 字），\n'
            '连同用户问题一起构造 prompt 发送给 LLM。\n'
            '展开"输入数据"中的 llm_request 可以看到完整的 prompt 内容。'
        ),
        component='LlamaIndex:LLM_Request',
    )

    t0 = time.time()
    from llama_index.core.response_synthesizers import get_response_synthesizer
    from backend.config import get_llm
    synthesizer = get_response_synthesizer(response_mode=synth_mode, llm=get_llm())
    response = synthesizer.synthesize(query, fused_results)
    elapsed_llm = int((time.time() - t0) * 1000)
    answer = str(response)

    tracer.trace(
        phase='query', title=f'LLM 回答生成完成 ({label})',
        code=(
            f'# 合成模式: {synth_mode}\n'
            f'# {desc}\n'
            f'▶ synthesizer = get_response_synthesizer(response_mode="{synth_mode}")\n'
            f'  response = synthesizer.synthesize(query, nodes)\n'
            f'# {len(fused_results)} 个 Node → {call_est}'
        ),
        input_data={'query': query, 'node_count': len(fused_results), 'synth_mode': synth_mode},
        output_data={'answer': answer[:300] + ('...' if len(answer) > 300 else '')},
        elapsed_ms=elapsed_llm,
        explanation=(
            f'合成模式: {label} — {desc}\n'
            f'耗时 {elapsed_llm}ms，{call_est}。\n\n'
            '各模式对比：\n'
            '• refine: 逐个 Node 精炼，质量最高但最慢（N 次 LLM 调用）\n'
            '• compact: 压缩后一次调用，速度快但可能丢信息\n'
            '• compact_and_refine: 折中方案\n'
            '• tree_summarize: 树形合并，适合大量 Node'
        ),
        component='LlamaIndex:ResponseSynthesizer',
    )


@app.post('/api/reset')
async def reset():
    """重置状态"""
    state.tracer.reset()
    state.nodes_data = None
    state.engines = None
    state.node_tree = None
    state.initialized = False

    if StorageFactory.is_production():
        StorageFactory.reset()
        from backend.file_handler import clear_uploads
        clear_uploads()

    return JSONResponse({'status': 'ok'})


@app.get('/api/steps')
async def get_steps():
    return JSONResponse(state.tracer.get_history())


@app.get('/api/node_tree')
async def get_node_tree():
    return JSONResponse(state.node_tree or [])


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8766)
# AI_GENERATE_END

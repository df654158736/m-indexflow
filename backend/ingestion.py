# AI_GENERATE_START
"""文档加载与 Node 解析管线"""
from llama_index.core.schema import Document, TextNode
from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes

from backend.step_tracer import StepTracer
from backend.sample_docs import load_documents


def run_ingestion(tracer: StepTracer):
    """执行完整的文档加载 + Node 解析流程，返回分类后的 Node"""

    # ── Step: 加载文档 ──
    docs = load_documents()
    tracer.trace(
        phase='ingest', title='加载文档 — Document 对象创建',
        code=(
            'from llama_index.core.schema import Document\n'
            '\n'
            '# LlamaIndex 的数据入口：Document 对象\n'
            '▶ documents = [\n'
            '    Document(text="规章制度...", metadata={"source": "..."}),\n'
            '    Document(text="产品手册...", metadata={"source": "..."}),\n'
            '    Document(text="会议纪要...", metadata={"source": "..."}),\n'
            ']\n'
            '# 等价于 SimpleDirectoryReader("data/").load_data()'
        ),
        input_data={'doc_count': len(docs), 'docs': [{'source': d.metadata['source'], 'length': len(d.text)} for d in docs]},
        explanation=(
            'Document 是 LlamaIndex 的数据入口。\n'
            '每个 Document 包含 text（原文）和 metadata（元数据）。\n'
            '实际项目中通常用 SimpleDirectoryReader 自动加载文件夹下的文档。'
        ),
        component='LlamaIndex:Document',
    )

    # 按类型分组
    policy_doc = [d for d in docs if d.metadata['doc_type'] == 'policy']
    product_doc = [d for d in docs if d.metadata['doc_type'] == 'product']
    meeting_doc = [d for d in docs if d.metadata['doc_type'] == 'meeting']

    # ── Step: SentenceSplitter 分块（产品手册） ──
    splitter = SentenceSplitter(chunk_size=256, chunk_overlap=32)
    tracer.trace(
        phase='ingest', title='创建 SentenceSplitter 分块器',
        code=(
            'from llama_index.core.node_parser import SentenceSplitter\n'
            '\n'
            '▶ splitter = SentenceSplitter(\n'
            '    chunk_size=256,      # 每个 Node 最大 256 token\n'
            '    chunk_overlap=32     # 相邻 Node 重叠 32 token\n'
            ')'
        ),
        explanation=(
            'SentenceSplitter 是 LlamaIndex 最常用的分块器。\n'
            '它按句子边界切分文本，确保每个 Node 不超过 chunk_size。\n'
            'chunk_overlap 让相邻 Node 有重叠部分，避免语义断裂。'
        ),
        component='LlamaIndex:SentenceSplitter',
    )

    product_nodes = splitter.get_nodes_from_documents(product_doc)
    tracer.trace(
        phase='ingest', title='产品手册 — SentenceSplitter 分块',
        code=(
            '# Document → TextNode 列表\n'
            '▶ product_nodes = splitter.get_nodes_from_documents(product_docs)\n'
            f'# 产品手册被分为 {len(product_nodes)} 个 Node'
        ),
        input_data={'doc_source': '产品手册', 'doc_length': len(product_doc[0].text)},
        output_data={
            'node_count': len(product_nodes),
            'nodes_preview': [{'id': n.node_id[:8], 'text': n.text[:60] + '...', 'length': len(n.text)} for n in product_nodes[:4]],
        },
        explanation=(
            '分块后每个 TextNode 包含：\n'
            '• node_id：唯一标识\n'
            '• text：文本内容\n'
            '• metadata：继承自 Document 的元数据\n'
            '• relationships：与其他 Node 的关系（SOURCE 指向原始 Document）'
        ),
        component='LlamaIndex:TextNode',
    )

    meeting_nodes = splitter.get_nodes_from_documents(meeting_doc)
    tracer.trace(
        phase='ingest', title='会议纪要 — SentenceSplitter 分块',
        code=f'▶ meeting_nodes = splitter.get_nodes_from_documents(meeting_docs)\n# 会议纪要被分为 {len(meeting_nodes)} 个 Node',
        input_data={'doc_source': '会议纪要', 'doc_length': len(meeting_doc[0].text)},
        output_data={'node_count': len(meeting_nodes)},
        component='LlamaIndex:TextNode',
    )

    # ── Step: HierarchicalNodeParser 层级解析（规章制度） ──
    hier_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 256])
    tracer.trace(
        phase='ingest', title='创建 HierarchicalNodeParser 层级解析器',
        code=(
            'from llama_index.core.node_parser import HierarchicalNodeParser\n'
            '\n'
            '▶ hier_parser = HierarchicalNodeParser.from_defaults(\n'
            '    chunk_sizes=[2048, 512, 256]  # 三级层级：章 → 节 → 段\n'
            ')'
        ),
        explanation=(
            'HierarchicalNodeParser 是 LlamaIndex 的杀手级特性。\n'
            '它将文档解析为多层级的 Node 树：\n'
            '• 第一级（1024 token）：章级别\n'
            '• 第二级（256 token）：节级别\n'
            '• 第三级（64 token）：段级别\n'
            'Node 之间自动建立 parent/child 关系。\n'
            '配合 AutoMergingRetriever 可以实现"先精确检索小块，再合并回大块"。'
        ),
        component='LlamaIndex:HierarchicalNodeParser',
    )

    all_policy_nodes = hier_parser.get_nodes_from_documents(policy_doc)
    leaf_nodes = get_leaf_nodes(all_policy_nodes)
    root_nodes = get_root_nodes(all_policy_nodes)

    tracer.trace(
        phase='ingest', title='规章制度 — 层级解析完成',
        code=(
            '▶ all_policy_nodes = hier_parser.get_nodes_from_documents(policy_docs)\n'
            '\n'
            '# 分离不同层级\n'
            'leaf_nodes = get_leaf_nodes(all_policy_nodes)    # 叶子节点（最细粒度）\n'
            'root_nodes = get_root_nodes(all_policy_nodes)    # 根节点（最粗粒度）'
        ),
        input_data={'doc_source': '规章制度', 'doc_length': len(policy_doc[0].text)},
        output_data={
            'total_nodes': len(all_policy_nodes),
            'leaf_count': len(leaf_nodes),
            'root_count': len(root_nodes),
            'mid_count': len(all_policy_nodes) - len(leaf_nodes) - len(root_nodes),
            'tree_sample': _build_node_tree_preview(all_policy_nodes[:20]),
        },
        explanation=(
            f'规章制度被解析为 {len(all_policy_nodes)} 个 Node：\n'
            f'• {len(root_nodes)} 个根节点（章级别）\n'
            f'• {len(all_policy_nodes) - len(leaf_nodes) - len(root_nodes)} 个中间节点（节级别）\n'
            f'• {len(leaf_nodes)} 个叶子节点（段级别）\n'
            'Node 之间通过 relationships 字段关联：\n'
            '  node.relationships[PARENT] → 父节点 ID\n'
            '  node.relationships[CHILD] → 子节点 ID 列表'
        ),
        component='LlamaIndex:NodeRelationships',
    )

    # ── Step: IngestionPipeline 展示 ──
    tracer.trace(
        phase='ingest', title='IngestionPipeline — 标准数据摄入管线',
        code=(
            'from llama_index.core.ingestion import IngestionPipeline\n'
            '\n'
            '# 实际项目中推荐用 Pipeline 组装处理流程\n'
            '▶ pipeline = IngestionPipeline(\n'
            '    transformations=[\n'
            '        SentenceSplitter(chunk_size=256),\n'
            '        embed_model,  # 可在管线中直接 embedding\n'
            '    ]\n'
            ')\n'
            'nodes = pipeline.run(documents=documents)'
        ),
        explanation=(
            'IngestionPipeline 是 LlamaIndex 推荐的数据摄入方式。\n'
            '它将多个 Transformation（分块、embedding、元数据提取等）串联成管线。\n'
            '支持缓存、增量更新、并行处理。\n'
            '本 Demo 为了逐步展示，手动调用了各组件，实际项目中建议用 Pipeline。'
        ),
        component='LlamaIndex:IngestionPipeline',
    )

    return {
        'policy_all_nodes': all_policy_nodes,
        'policy_leaf_nodes': leaf_nodes,
        'product_nodes': product_nodes,
        'meeting_nodes': meeting_nodes,
        'node_tree': _build_full_node_tree(all_policy_nodes, product_nodes, meeting_nodes),
    }


def _build_node_tree_preview(nodes: list) -> list:
    """构建 Node 树预览"""
    result = []
    for n in nodes[:10]:
        info = {'id': n.node_id[:8], 'text': n.text[:50] + '...'}
        from llama_index.core.schema import NodeRelationship
        if NodeRelationship.PARENT in n.relationships:
            info['parent'] = n.relationships[NodeRelationship.PARENT].node_id[:8]
        if NodeRelationship.CHILD in n.relationships:
            info['children'] = len(n.relationships[NodeRelationship.CHILD]) if isinstance(n.relationships[NodeRelationship.CHILD], list) else 1
        result.append(info)
    return result


def _build_full_node_tree(policy_nodes, product_nodes, meeting_nodes) -> list:
    """构建前端可渲染的完整 Node 树 JSON"""
    tree = []

    # 规章制度 — 层级树
    from llama_index.core.schema import NodeRelationship
    root_ids = set()
    node_map = {n.node_id: n for n in policy_nodes}
    for n in policy_nodes:
        if NodeRelationship.PARENT not in n.relationships:
            root_ids.add(n.node_id)

    def build_subtree(node_id):
        n = node_map.get(node_id)
        if not n:
            return None
        item = {
            'id': n.node_id[:8],
            'text': n.text[:60].replace('\n', ' ').strip(),
            'type': 'policy',
            'children': [],
        }
        # 查找子节点
        for cid, cn in node_map.items():
            if NodeRelationship.PARENT in cn.relationships:
                parent_ref = cn.relationships[NodeRelationship.PARENT]
                if parent_ref.node_id == node_id:
                    child = build_subtree(cid)
                    if child:
                        item['children'].append(child)
        return item

    policy_tree = {'id': 'policy', 'text': '📋 公司规章制度', 'type': 'group', 'children': []}
    for rid in root_ids:
        sub = build_subtree(rid)
        if sub:
            policy_tree['children'].append(sub)
    tree.append(policy_tree)

    # 产品手册 — 扁平列表
    product_tree = {'id': 'product', 'text': '📊 产品手册', 'type': 'group', 'children': []}
    for n in product_nodes:
        product_tree['children'].append({
            'id': n.node_id[:8],
            'text': n.text[:60].replace('\n', ' ').strip(),
            'type': 'product',
            'children': [],
        })
    tree.append(product_tree)

    # 会议纪要 — 扁平列表
    meeting_tree = {'id': 'meeting', 'text': '💬 会议纪要', 'type': 'group', 'children': []}
    for n in meeting_nodes:
        meeting_tree['children'].append({
            'id': n.node_id[:8],
            'text': n.text[:60].replace('\n', ' ').strip(),
            'type': 'meeting',
            'children': [],
        })
    tree.append(meeting_tree)

    return tree


def run_ingestion_from_files(documents: list, tracer: StepTracer):
    """production 模式：HierarchicalNodeParser 层级切片

    用小块检索、用大块回答 —— 解决切片粒度 vs 上下文完整性的矛盾。
    """
    tracer.trace(
        phase='ingest', title=f'加载上传文档 — {len(documents)} 个 Document',
        code=(
            '# production 模式：使用用户上传的真实文件\n'
            '▶ documents = SimpleDirectoryReader(input_files=[...]).load_data()\n'
            f'# 共 {len(documents)} 个 Document'
        ),
        input_data={'doc_count': len(documents), 'sources': list({d.metadata.get("source", "unknown") for d in documents})},
        explanation='production 模式下，文档来自用户上传的 PDF/MD/TXT 文件。',
        component='LlamaIndex:SimpleDirectoryReader',
    )

    # ── 关键步骤：合并同一来源的 Document ──
    # PDF 按页解析会产生多个 Document，导致跨页内容被切断。
    # 合并后让 HierarchicalNodeParser 看到完整文本，才能正确建立层级关系。
    from collections import defaultdict
    groups = defaultdict(list)
    for doc in documents:
        src = doc.metadata.get('source', doc.metadata.get('file_name', 'unknown'))
        groups[src].append(doc)

    merged_docs = []
    for src, doc_list in groups.items():
        merged_text = '\n'.join([d.text for d in doc_list])
        merged_meta = doc_list[0].metadata.copy()
        merged_meta['original_doc_count'] = len(doc_list)
        merged_docs.append(Document(text=merged_text, metadata=merged_meta))

    tracer.trace(
        phase='ingest', title=f'合并 Document — {len(documents)} 页 → {len(merged_docs)} 个完整文档',
        code=(
            '# PDF 按页解析会产生多个 Document，跨页内容会被切断\n'
            '# 合并同一来源的 Document，确保完整段落不会跨页断裂\n'
            '▶ merged_docs = merge_by_source(documents)\n'
            f'# {len(documents)} 页合并为 {len(merged_docs)} 个完整文档'
        ),
        input_data={'before': len(documents), 'after': len(merged_docs)},
        explanation=(
            f'PDF 解析出 {len(documents)} 个 Document（每页一个），\n'
            f'合并为 {len(merged_docs)} 个完整文档后再做层级切片。\n'
            '这是生产级 RAG 的关键步骤：\n'
            '• 不合并 → "四大满贯"列表跨页被切断，检索不完整\n'
            '• 合并后 → HierarchicalNodeParser 能看到完整文本，正确建立 parent/child 关系'
        ),
        component='LlamaIndex:DocumentMerge',
    )
    documents = merged_docs

    # 使用 HierarchicalNodeParser 三级层级切片
    hier_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 256])
    tracer.trace(
        phase='ingest', title='HierarchicalNodeParser 层级切片',
        code=(
            'from llama_index.core.node_parser import HierarchicalNodeParser\n'
            '\n'
            '# 三级层级：大块(1024) → 中块(256) → 小块(64)\n'
            '▶ hier_parser = HierarchicalNodeParser.from_defaults(\n'
            '    chunk_sizes=[2048, 512, 256]\n'
            ')\n'
            '  all_nodes = hier_parser.get_nodes_from_documents(documents)'
        ),
        explanation=(
            'HierarchicalNodeParser 将文档切成三级 Node 树：\n'
            '• Level 1 (1024 token)：章级别，上下文完整\n'
            '• Level 2 (256 token)：节级别\n'
            '• Level 3 (64 token)：段级别，检索精准\n'
            'Node 之间自动建立 parent/child 关系。\n'
            '配合 AutoMergingRetriever 实现"用小块检索、用大块回答"。'
        ),
        component='LlamaIndex:HierarchicalNodeParser',
    )

    all_nodes = hier_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(all_nodes)
    root_nodes = get_root_nodes(all_nodes)
    mid_count = len(all_nodes) - len(leaf_nodes) - len(root_nodes)

    tracer.trace(
        phase='ingest', title=f'层级切片完成 — {len(all_nodes)} 个 Node',
        code=(
            f'all_nodes = hier_parser.get_nodes_from_documents(documents)\n'
            f'leaf_nodes = get_leaf_nodes(all_nodes)   # {len(leaf_nodes)} 个叶子\n'
            f'root_nodes = get_root_nodes(all_nodes)   # {len(root_nodes)} 个根节点\n'
            f'# 中间节点: {mid_count} 个'
        ),
        output_data={
            'total_nodes': len(all_nodes),
            'leaf_count': len(leaf_nodes),
            'root_count': len(root_nodes),
            'mid_count': mid_count,
            'nodes_preview': [{'id': n.node_id[:8], 'text': n.text[:80] + '...', 'level': 'leaf'} for n in leaf_nodes[:5]],
        },
        explanation=(
            f'层级切片结果：\n'
            f'• {len(root_nodes)} 个根节点（Level 1，最粗粒度）\n'
            f'• {mid_count} 个中间节点（Level 2）\n'
            f'• {len(leaf_nodes)} 个叶子节点（Level 3，最细粒度）\n'
            f'共 {len(all_nodes)} 个 Node，通过 parent/child 关系关联。\n'
            '叶子节点用于向量检索（精准），父节点用于合并回答（完整）。'
        ),
        component='LlamaIndex:NodeRelationships',
    )

    # 构建 Node 树 JSON（前端展示用）
    tree = _build_hierarchical_tree(all_nodes, documents)

    return {
        'all_nodes': all_nodes,
        'leaf_nodes': leaf_nodes,
        'node_tree': tree,
    }


def _build_hierarchical_tree(all_nodes, documents):
    """构建前端可渲染的层级 Node 树"""
    from llama_index.core.schema import NodeRelationship

    tree = []
    sources = list({d.metadata.get('source', 'unknown') for d in documents})

    for src in sources:
        group = {'id': src[:8], 'text': f'📄 {src}', 'type': 'uploaded', 'children': []}
        # 只展示根节点和它们的子树摘要
        root_ids = set()
        node_map = {}
        for n in all_nodes:
            if n.metadata.get('source', n.metadata.get('file_name', '')) == src or \
               n.metadata.get('file_name', '').endswith(src):
                node_map[n.node_id] = n
                if NodeRelationship.PARENT not in n.relationships:
                    root_ids.add(n.node_id)

        for rid in list(root_ids)[:20]:  # 限制展示数量
            n = node_map.get(rid)
            if n:
                group['children'].append({
                    'id': n.node_id[:8],
                    'text': n.text[:80].replace('\n', ' ').strip(),
                    'type': 'root',
                    'children': [],
                })
        tree.append(group)

    return tree
# AI_GENERATE_END

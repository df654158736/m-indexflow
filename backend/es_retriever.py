# AI_GENERATE_START
"""
自定义 ES BM25 检索器

继承 LlamaIndex 的 BaseRetriever，内部使用 elasticsearch-py 查询。
返回标准的 List[NodeWithScore]，可无缝接入 QueryFusionRetriever。
"""
import json
import logging
from typing import Any, Optional

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

logger = logging.getLogger(__name__)


class ESBm25Retriever(BaseRetriever):
    """ES BM25 全文检索器

    用真实的 Elasticsearch 倒排索引做 BM25 检索，
    替代 LlamaIndex 内置的内存 BM25Retriever，支持任意数据量。
    """

    def __init__(
        self,
        es_url: str,
        index_name: str,
        similarity_top_k: int = 5,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        from backend.config import _create_es_client
        self._es = _create_es_client()
        self._index_name = index_name
        self._similarity_top_k = similarity_top_k
        self._ensure_index()

    def _ensure_index(self):
        """确保 ES 索引存在，不存在则创建"""
        if self._es.indices.exists(index=self._index_name):
            return

        # 检测 IK 分词器是否可用
        analyzer = 'standard'
        try:
            self._es.indices.analyze(body={'analyzer': 'ik_max_word', 'text': '测试'})
            analyzer = 'ik_max_word'
            logger.info(f'ES IK 分词器可用，使用 ik_max_word')
        except Exception:
            logger.warning('ES IK 分词器不可用，回退到 standard 分词器（中文效果会差一些）')

        mapping = {
            'mappings': {
                'properties': {
                    'node_id': {'type': 'keyword'},
                    'text': {'type': 'text', 'analyzer': analyzer},
                    'doc_source': {'type': 'keyword'},
                    'metadata': {'type': 'object', 'enabled': False},
                }
            },
            'settings': {
                'number_of_shards': 1,
                'number_of_replicas': 0,
            }
        }
        self._es.indices.create(index=self._index_name, body=mapping)
        logger.info(f'ES 索引已创建: {self._index_name} (analyzer={analyzer})')

    def _retrieve(self, query_bundle: QueryBundle, **kwargs) -> list[NodeWithScore]:
        """执行 BM25 检索

        用 ES match 查询搜索 text 字段，
        将 hits 转为 LlamaIndex 标准的 NodeWithScore 列表。
        """
        body = {
            'query': {
                'match': {
                    'text': {
                        'query': query_bundle.query_str,
                        'minimum_should_match': '30%',
                    }
                }
            },
            'size': self._similarity_top_k,
        }

        try:
            resp = self._es.search(index=self._index_name, body=body)
        except Exception as e:
            logger.error(f'ES 检索失败: {e}')
            return []

        nodes = []
        for hit in resp['hits']['hits']:
            source = hit['_source']
            metadata = source.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            node = TextNode(
                id_=source.get('node_id', hit['_id']),
                text=source.get('text', ''),
                metadata=metadata,
            )
            nodes.append(NodeWithScore(node=node, score=hit['_score']))

        return nodes

    def index_nodes(self, nodes: list, tracer=None):
        """将 Node 列表批量写入 ES

        Args:
            nodes: LlamaIndex TextNode 列表
            tracer: StepTracer（可选）
        """
        from elasticsearch.helpers import bulk

        actions = []
        for node in nodes:
            doc_source = ''
            if hasattr(node, 'metadata') and node.metadata:
                doc_source = node.metadata.get('source', node.metadata.get('file_name', ''))

            actions.append({
                '_index': self._index_name,
                '_id': node.node_id,
                '_source': {
                    'node_id': node.node_id,
                    'text': node.text,
                    'doc_source': doc_source,
                    'metadata': json.dumps(node.metadata, ensure_ascii=False, default=str) if node.metadata else '{}',
                },
            })

        if actions:
            success, errors = bulk(self._es, actions, raise_on_error=False)
            logger.info(f'ES 入库完成: {success} 条成功, {len(errors)} 条失败')

            if tracer:
                tracer.trace(
                    phase='index', title='ES BM25 入库',
                    code=(
                        'from elasticsearch.helpers import bulk\n'
                        '\n'
                        '# 将每个 Node 的文本写入 ES 倒排索引\n'
                        '▶ bulk(es_client, [\n'
                        '    {"_index": index_name, "_source": {\n'
                        '        "node_id": node.node_id,\n'
                        '        "text": node.text,      # BM25 检索字段\n'
                        '        "doc_source": "...",\n'
                        '        "metadata": {...}\n'
                        '    }}\n'
                        '    for node in nodes\n'
                        '])'
                    ),
                    input_data={'node_count': len(nodes), 'index_name': self._index_name},
                    output_data={'success': success, 'errors': len(errors)},
                    explanation=(
                        'ES 负责 BM25 全文检索（关键词匹配）。\n'
                        '每个 Node 的 text 字段会被 ES 分词并建立倒排索引。\n'
                        '查询时 ES 通过倒排索引快速找到包含查询词的文档，\n'
                        '按 BM25 算法计算相关性分数排序。\n'
                        '相比内存 BM25Retriever，ES 支持任意数据量且毫秒级响应。'
                    ),
                    component='LlamaIndex:ESBm25Retriever',
                )

    def clear_index(self):
        """清空索引数据"""
        try:
            if self._es.indices.exists(index=self._index_name):
                self._es.indices.delete(index=self._index_name)
                logger.info(f'ES 索引已删除: {self._index_name}')
        except Exception as e:
            logger.error(f'ES 索引删除失败: {e}')
# AI_GENERATE_END

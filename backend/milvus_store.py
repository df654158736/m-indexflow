# AI_GENERATE_START
"""
自定义 Milvus 向量存储

参考 ZFAPT 的 pymilvus ORM 连接方式，
封装为 LlamaIndex 的 BasePydanticVectorStore 接口。
"""
import json
import logging
from typing import Any, List, Optional
from uuid import uuid4

from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

logger = logging.getLogger(__name__)

COLLECTION_PREFIX = 'm_indexflow_'


class MilvusStore(BasePydanticVectorStore):
    """基于 pymilvus ORM 的 Milvus 向量存储

    参考 ZFAPT 的连接方式（connections.connect + Collection ORM），
    实现 LlamaIndex VectorStore 标准接口。
    """

    stores_text: bool = True
    is_embedding_query: bool = True

    _collection: Any = None
    _alias: str = ''
    _collection_name: str = ''
    _dim: int = 0

    class Config:
        arbitrary_types_allowed = True

    @property
    def client(self) -> Any:
        """LlamaIndex 要求的 client 属性"""
        return self._collection

    def __init__(self, host: str, port: int, user: str, password: str,
                 collection_name: str, dim: int = 1024, **kwargs):
        super().__init__(**kwargs)
        from pymilvus import connections

        self._collection_name = f'{COLLECTION_PREFIX}{collection_name}'
        self._dim = dim

        # 参考 ZFAPT 的连接方式：随机 alias + connections.connect
        self._alias = uuid4().hex
        connections.connect(
            alias=self._alias,
            host=host,
            port=str(port),
            user=user,
            password=password,
        )
        logger.info(f'Milvus 已连接: {host}:{port}, alias={self._alias}')

        self._ensure_collection()

    def _ensure_collection(self):
        """确保 Collection 存在"""
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

        if utility.has_collection(self._collection_name, using=self._alias):
            self._collection = Collection(self._collection_name, using=self._alias)
            self._collection.load()
            logger.info(f'Milvus Collection 已存在: {self._collection_name}')
            return

        # 创建 Collection
        fields = [
            FieldSchema('id', DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema('node_id', DataType.VARCHAR, max_length=128),
            FieldSchema('text', DataType.VARCHAR, max_length=65535),
            FieldSchema('metadata', DataType.VARCHAR, max_length=65535),
            FieldSchema('embedding', DataType.FLOAT_VECTOR, dim=self._dim),
        ]
        schema = CollectionSchema(fields, description='m-indexflow vector store')
        self._collection = Collection(
            self._collection_name, schema=schema, using=self._alias
        )

        # 创建索引
        self._collection.create_index(
            'embedding',
            {'index_type': 'HNSW', 'metric_type': 'COSINE', 'params': {'M': 16, 'efConstruction': 256}},
        )
        self._collection.load()
        logger.info(f'Milvus Collection 已创建: {self._collection_name}, dim={self._dim}')

    def add(self, nodes: List[BaseNode], **kwargs) -> List[str]:
        """写入 Node 到 Milvus"""
        if not nodes:
            return []

        ids = []
        data = []
        for node in nodes:
            node_id = node.node_id
            text = node.get_content()
            embedding = node.get_embedding()
            metadata = json.dumps(node.metadata, ensure_ascii=False, default=str) if node.metadata else '{}'

            ids.append(node_id)
            data.append({
                'node_id': node_id,
                'text': text[:65000],
                'metadata': metadata[:65000],
                'embedding': embedding,
            })

        # 批量插入（pymilvus 新版用 list[dict] 格式）
        self._collection.insert(data)
        self._collection.flush()
        logger.info(f'Milvus 写入 {len(ids)} 条数据')
        return ids

    def delete(self, ref_doc_id: str, **kwargs) -> None:
        """删除指定文档"""
        self._collection.delete(f'node_id == "{ref_doc_id}"')

    def query(self, query: VectorStoreQuery, **kwargs) -> VectorStoreQueryResult:
        """向量检索"""
        if query.query_embedding is None:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        results = self._collection.search(
            data=[query.query_embedding],
            anns_field='embedding',
            param={'metric_type': 'COSINE', 'params': {'ef': 100}},
            limit=query.similarity_top_k or 5,
            output_fields=['node_id', 'text', 'metadata'],
        )

        nodes = []
        similarities = []
        ids = []

        for hits in results:
            for hit in hits:
                text = hit.entity.get('text', '')
                metadata_str = hit.entity.get('metadata', '{}')
                try:
                    metadata = json.loads(metadata_str)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

                node = TextNode(
                    id_=hit.entity.get('node_id', str(hit.id)),
                    text=text,
                    metadata=metadata,
                )
                nodes.append(node)
                similarities.append(hit.score)
                ids.append(str(hit.id))

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def drop_collection(self):
        """删除整个 Collection"""
        from pymilvus import utility
        try:
            if utility.has_collection(self._collection_name, using=self._alias):
                self._collection.drop()
                logger.info(f'Milvus Collection 已删除: {self._collection_name}')
        except Exception as e:
            logger.error(f'删除失败: {e}')
# AI_GENERATE_END

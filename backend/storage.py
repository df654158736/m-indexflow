# AI_GENERATE_START
"""
存储层工厂

根据 MODE 配置返回不同的 VectorStore 和 Retriever 实现。
上层代码通过工厂获取存储，不直接依赖具体实现。

demo 模式：SimpleVectorStore（内存）+ BM25Retriever（内存）
production 模式：MilvusVectorStore + ESBm25Retriever
"""
import logging
from typing import Optional

from backend.config import (
    MODE, MILVUS_HOST, MILVUS_PORT, MILVUS_USER, MILVUS_PASSWORD,
    MILVUS_COLLECTION, ES_URL, ES_INDEX, get_embed_model,
)

logger = logging.getLogger(__name__)


class StorageFactory:
    """存储层工厂"""

    _es_retriever = None
    _milvus_store = None

    @classmethod
    def get_vector_store(cls):
        """获取向量存储

        demo → None（VectorStoreIndex 使用默认内存存储）
        production → MilvusVectorStore
        """
        if MODE != 'production':
            return None

        if cls._milvus_store is None:
            from backend.milvus_store import MilvusStore

            cls._milvus_store = MilvusStore(
                host=MILVUS_HOST,
                port=MILVUS_PORT,
                user=MILVUS_USER,
                password=MILVUS_PASSWORD,
                collection_name=MILVUS_COLLECTION,
                dim=1024,
            )
            logger.info(f'MilvusStore 已创建: {MILVUS_HOST}:{MILVUS_PORT}')

        return cls._milvus_store

    @classmethod
    def get_bm25_retriever(cls, nodes: Optional[list] = None, similarity_top_k: int = 5):
        """获取 BM25 检索器

        demo → BM25Retriever（内存，需要传入 nodes）
        production → ESBm25Retriever（真实 ES）
        """
        if MODE != 'production':
            if nodes is None:
                return None
            try:
                from llama_index.retrievers.bm25 import BM25Retriever
                import jieba
                return BM25Retriever.from_defaults(
                    nodes=nodes,
                    similarity_top_k=similarity_top_k,
                    tokenizer=lambda text: jieba.lcut(text),
                )
            except ImportError:
                logger.warning('BM25Retriever 不可用（缺少 llama-index-retrievers-bm25），跳过 BM25 检索')
                return None

        if cls._es_retriever is None:
            from backend.es_retriever import ESBm25Retriever
            cls._es_retriever = ESBm25Retriever(
                es_url=ES_URL,
                index_name=f'm_indexflow_{ES_INDEX}',
                similarity_top_k=similarity_top_k,
            )
            logger.info(f'ESBm25Retriever 已创建: {ES_URL}, index=m_indexflow_{ES_INDEX}')

        return cls._es_retriever

    @classmethod
    def index_nodes_to_es(cls, nodes: list, tracer=None):
        """将 Node 写入 ES（仅 production 模式）"""
        if MODE != 'production':
            return

        retriever = cls.get_bm25_retriever()
        if retriever and hasattr(retriever, 'index_nodes'):
            retriever.index_nodes(nodes, tracer=tracer)

    @classmethod
    def reset(cls):
        """重置所有存储连接"""
        if cls._es_retriever and hasattr(cls._es_retriever, 'clear_index'):
            cls._es_retriever.clear_index()
        cls._es_retriever = None

        if cls._milvus_store:
            try:
                cls._milvus_store.drop_collection()
            except Exception as e:
                logger.error(f'Milvus collection 删除失败: {e}')
        cls._milvus_store = None

    @classmethod
    def is_production(cls) -> bool:
        return MODE == 'production'
# AI_GENERATE_END

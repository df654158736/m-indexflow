# AI_GENERATE_START
"""
配置管理

使用 llama-index-llms-openai-like 适配非 OpenAI 模型（通义千问、DeepSeek 等）。
通过 MODE 配置切换 demo（内存）/ production（Milvus + ES）模式。
"""
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 从项目根目录加载 .env
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / '.env')

# ── LLM 配置 ──
API_KEY = os.getenv('LLM_API_KEY', '')
BASE_URL = os.getenv('LLM_BASE_URL', '')
MODEL = os.getenv('LLM_MODEL', 'qwen-max')
TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.7'))
EMBED_MODEL = os.getenv('LLM_EMBED_MODEL', 'text-embedding-v3')

# ── 模式配置 ──
MODE = os.getenv('MODE', 'demo')  # demo | production

# ── Milvus 配置（production 模式） ──
MILVUS_HOST = os.getenv('MILVUS_HOST', '127.0.0.1')
MILVUS_PORT = int(os.getenv('MILVUS_PORT', '19530'))
MILVUS_USER = os.getenv('MILVUS_USER', 'root')
MILVUS_PASSWORD = os.getenv('MILVUS_PASSWORD', '')
MILVUS_COLLECTION = os.getenv('MILVUS_COLLECTION', 'm_indexflow_kb')

# ── ES 配置（production 模式） ──
ES_URL = os.getenv('ES_URL', 'http://localhost:9200')
ES_INDEX = os.getenv('ES_INDEX', 'm_indexflow_kb')


def get_llm():
    """获取 LLM 实例

    使用 OpenAILike 适配 OpenAI 兼容接口的非 OpenAI 模型。
    """
    from llama_index.llms.openai_like import OpenAILike
    return OpenAILike(
        api_key=API_KEY,
        api_base=BASE_URL,
        model=MODEL,
        temperature=TEMPERATURE,
        is_chat_model=True,
        context_window=32000,
    )


def get_embed_model():
    """获取 Embedding 模型实例

    通过继承 BaseEmbedding 直接调用 OpenAI SDK 适配通义千问。
    """
    from llama_index.core.embeddings import BaseEmbedding
    from openai import OpenAI
    from pydantic import PrivateAttr

    class CompatEmbedding(BaseEmbedding):
        """OpenAI 兼容接口 Embedding 适配器"""
        _client: OpenAI = PrivateAttr()

        def __init__(self):
            super().__init__(model_name=EMBED_MODEL, embed_batch_size=10)
            self._client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

        def _get_text_embedding(self, text: str) -> list[float]:
            resp = self._client.embeddings.create(input=[text], model=EMBED_MODEL)
            return resp.data[0].embedding

        def _get_query_embedding(self, query: str) -> list[float]:
            return self._get_text_embedding(query)

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_text_embedding(query)

    return CompatEmbedding()


def _create_es_client():
    """创建 ES 客户端，从 URL 中提取认证信息"""
    from elasticsearch import Elasticsearch
    from urllib.parse import urlparse
    parsed = urlparse(ES_URL)
    if parsed.username and parsed.password:
        # URL 包含认证信息：http://user:pass@host:port
        host = f'{parsed.scheme}://{parsed.hostname}:{parsed.port}'
        return Elasticsearch(host, http_auth=(parsed.username, parsed.password), verify_certs=False, request_timeout=10)
    return Elasticsearch(ES_URL, verify_certs=False, request_timeout=10)


def check_production_connectivity() -> dict:
    """检测 production 模式下 Milvus + ES 的连通性

    Returns:
        {"milvus": True/False, "es": True/False, "mode": "demo"/"production"}
    """
    global MODE
    result = {'milvus': False, 'es': False, 'mode': MODE}

    if MODE != 'production':
        return result

    # 检测 Milvus
    try:
        from pymilvus import connections
        connections.connect(
            alias='_check',
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            user=MILVUS_USER,
            password=MILVUS_PASSWORD,
        )
        connections.disconnect('_check')
        result['milvus'] = True
        logger.info(f'Milvus 连接成功: {MILVUS_HOST}:{MILVUS_PORT}')
    except Exception as e:
        logger.error(f'Milvus 连接失败: {e}')

    # 检测 ES
    try:
        from elasticsearch import Elasticsearch
        es = _create_es_client()
        if es.ping():
            result['es'] = True
            logger.info(f'ES 连接成功: {ES_URL}')
        else:
            logger.error('ES ping 失败')
    except Exception as e:
        logger.error(f'ES 连接失败: {e}')

    # 任一不可用则回退 demo 模式
    if not result['milvus'] or not result['es']:
        logger.warning('存储连接失败，自动回退到 demo 模式')
        MODE = 'demo'
        result['mode'] = 'demo'

    return result
# AI_GENERATE_END

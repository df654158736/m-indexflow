# AI_GENERATE_START
"""
文件上传与解析

支持 PDF、Markdown、TXT 文件。
使用 LlamaIndex 的 SimpleDirectoryReader 自动识别格式并解析为 Document。
"""
import os
import uuid
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# 支持的文件格式
SUPPORTED_EXTENSIONS = {'.pdf', '.md', '.txt', '.markdown'}

# 临时上传目录
UPLOAD_DIR = Path(tempfile.gettempdir()) / 'm_indexflow_uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 已上传的文档缓存 {file_id: [Document, ...]}
uploaded_documents: dict[str, list] = {}


def save_and_parse(filename: str, content: bytes) -> dict:
    """保存上传文件并解析为 LlamaIndex Document

    Args:
        filename: 原始文件名
        content: 文件二进制内容

    Returns:
        {"file_id": str, "filename": str, "doc_count": int}
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f'不支持的文件格式: {ext}，支持: {", ".join(SUPPORTED_EXTENSIONS)}')

    # 保存到临时目录
    file_id = uuid.uuid4().hex[:12]
    save_path = UPLOAD_DIR / f'{file_id}_{filename}'
    save_path.write_bytes(content)
    logger.info(f'文件已保存: {save_path} ({len(content)} bytes)')

    # 使用 SimpleDirectoryReader 解析
    from llama_index.core import SimpleDirectoryReader
    reader = SimpleDirectoryReader(input_files=[str(save_path)])
    documents = reader.load_data()

    # 覆盖 metadata，用原始文件名替换带前缀的临时文件名
    for doc in documents:
        doc.metadata['file_name'] = filename
        doc.metadata['source'] = filename
        doc.metadata['file_id'] = file_id

    # 缓存
    uploaded_documents[file_id] = documents
    logger.info(f'文件解析完成: {filename} → {len(documents)} 个 Document')

    return {
        'file_id': file_id,
        'filename': filename,
        'doc_count': len(documents),
    }


def get_all_documents() -> list:
    """获取所有已上传的 Document 列表"""
    all_docs = []
    for docs in uploaded_documents.values():
        all_docs.extend(docs)
    return all_docs


def clear_uploads():
    """清空上传缓存"""
    uploaded_documents.clear()
    # 清理临时文件
    for f in UPLOAD_DIR.iterdir():
        try:
            f.unlink()
        except Exception:
            pass
# AI_GENERATE_END

"""评测专用的中文 Embedding Function。

背景：ChromaDB 默认的 `all-MiniLM-L6-v2` 是**纯英文**模型，中文短查询的向量接近随机，
导致线上 `knowledge_base` collection 对中文问句基本检索不到东西（实测 recall@5≈0.17）。

本模块**只在评测侧**注入一个中文 embedding（默认 BAAI/bge-small-zh-v1.5），
用来回答「换上合适的中文 embedding 后，检索链路能到多少」。
**不改动生产代码** —— 线上仍用 ChromaDB 默认英文模型。

因此报告里的数字必须标注为「评测用中文 embedding」，不能当成线上实况。
"""
from __future__ import annotations

import logging
import os

# 本机/国内环境 huggingface.co 不可达，默认走 hf-mirror 下载模型；可用 HF_ENDPOINT 覆盖。
# 必须在 import sentence_transformers 之前设置才会生效。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)

DEFAULT_ZH_MODEL = "BAAI/bge-small-zh-v1.5"


class ChineseEmbeddingFunction(EmbeddingFunction):
    """包装 sentence-transformers 的中文模型，供 ChromaDB 使用（懒加载）。"""

    def __init__(self, model_name: str = DEFAULT_ZH_MODEL):
        self._model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"加载中文 embedding 模型: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - ChromaDB 的签名
        model = self._ensure_model()
        vectors = model.encode(
            list(input),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    # ── ChromaDB 1.x 需要的元信息接口 ────────────────────────────────────────

    @staticmethod
    def name() -> str:
        return "carepilot-chinese-embedding"

    def get_config(self) -> dict:
        return {"model_name": self._model_name}

    @staticmethod
    def build_from_config(config: dict) -> "ChineseEmbeddingFunction":
        return ChineseEmbeddingFunction(
            model_name=config.get("model_name", DEFAULT_ZH_MODEL)
        )


def make_eval_embedding_function() -> ChineseEmbeddingFunction:
    """按环境变量构造评测用 embedding；CAREPILOT_EVAL_EMBED_MODEL 可覆盖模型。"""
    model_name = os.getenv("CAREPILOT_EVAL_EMBED_MODEL", "").strip() or DEFAULT_ZH_MODEL
    return ChineseEmbeddingFunction(model_name=model_name)

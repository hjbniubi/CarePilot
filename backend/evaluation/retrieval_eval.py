"""
Phase 3：检索质量评测（RAG retrieval + answer accuracy）

为什么需要单独一块：
  Phase 1 测意图、Phase 2 测会话质量，都回答不了「检索到底找没找对文档」。
  简历里「检索召回率 X%」「RAG 回答准确率 Y%」正是这一块的产物。

测什么：
  1. 检索指标 —— recall@k / hit_rate@k / precision@k / MRR，对同一批 query 跑三条链路：
       - full            : 查询改写 + 并行召回 + LLM 重排（线上链路）
       - baseline_vector : 纯向量检索（ChromaDB，无改写无重排）
       - baseline_keyword: 纯字面匹配（字符二元组重叠，无向量无语义）
     给出 full 相对两条基线的 delta，支撑「提升 X 个百分点」的说法。
  2. 回答准确率 —— 用检索到的资料生成回答，再由 LLM-as-Judge 对照 gold 文档判准确性。

重要：ground truth 是**反向合成**的——每条 query 由某篇文档推导而来，该文档即 gold。
报告与简历必须注明是合成数据集，不能当成人工标注。
"""
from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content
from evaluation.evaluator import LLMJudge

logger = logging.getLogger(__name__)

DEFAULT_KS: Tuple[int, ...] = (1, 3, 5)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class RetrievalCase:
    query: str
    relevant_docs: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalCase":
        return cls(
            query=str(data.get("query", "")),
            relevant_docs=[str(x) for x in (data.get("relevant_docs") or [])],
        )


@dataclass
class RetrievalCaseResult:
    query: str
    relevant: List[str]
    full: List[str]
    vector: List[str]
    keyword: List[str]


# ── 指标计算（纯 Python，无依赖）─────────────────────────────────────────────

def _metrics(ranking: Sequence[str], relevant: Sequence[str], ks: Sequence[int]) -> Dict[str, float]:
    """对单条 query 的排名序列算 recall@k / hit_rate@k / precision@k / MRR。"""
    rel = set(relevant)
    out: Dict[str, float] = {}
    for k in ks:
        topk = list(ranking[:k])
        if not k:
            continue
        n_hit = sum(1 for t in topk if t in rel)
        out[f"recall@{k}"] = (n_hit / len(rel)) if rel else 0.0
        out[f"hit_rate@{k}"] = 1.0 if n_hit > 0 else 0.0
        out[f"precision@{k}"] = n_hit / len(topk) if topk else 0.0
    mrr = 0.0
    for rank, title in enumerate(ranking, start=1):
        if title in rel:
            mrr = 1.0 / rank
            break
    out["mrr"] = mrr
    return out


def _average(per_case: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_case:
        return {}
    keys = per_case[0].keys()
    return {k: round(statistics.mean(c[k] for c in per_case), 4) for k in keys}


# ── 检索评测器 ────────────────────────────────────────────────────────────────

class RetrievalEvaluator:
    """对同一批 query 跑 full / vector / keyword 三条链路并对比。"""

    def __init__(
        self,
        tool_manager,
        kb,
        corpus: Dict[str, str],
        ks: Sequence[int] = DEFAULT_KS,
    ):
        """
        tool_manager: MCPToolManager（需已注册 knowledge_search 工具）
        kb          : KnowledgeBase（提供纯向量检索作基线）
        corpus      : {title: content}，关键词基线用；也用于统计语料规模
        """
        self._tm = tool_manager
        self._kb = kb
        self._corpus = corpus
        self._ks = tuple(sorted(ks))
        self._max_k = max(self._ks)

    async def evaluate(self, cases: List[RetrievalCase]) -> Dict[str, Any]:
        per_case_results: List[RetrievalCaseResult] = []
        full_m, vec_m, kw_m = [], [], []

        for case in cases:
            if not case.query or not case.relevant_docs:
                continue
            full = await self._full_ranking(case.query)
            vector = await self._vector_ranking(case.query)
            keyword = self._keyword_ranking(case.query)

            per_case_results.append(RetrievalCaseResult(
                query=case.query,
                relevant=case.relevant_docs,
                full=full, vector=vector, keyword=keyword,
            ))
            full_m.append(_metrics(full, case.relevant_docs, self._ks))
            vec_m.append(_metrics(vector, case.relevant_docs, self._ks))
            kw_m.append(_metrics(keyword, case.relevant_docs, self._ks))

        full_avg = _average(full_m)
        vector_avg = _average(vec_m)
        keyword_avg = _average(kw_m)

        return {
            "dataset_size": len(per_case_results),
            "corpus_docs": len(self._corpus),
            "ks": list(self._ks),
            "full": full_avg,
            "baseline_vector": vector_avg,
            "baseline_keyword": keyword_avg,
            "delta_vs_keyword": {k: round(full_avg[k] - keyword_avg[k], 4) for k in full_avg},
            "delta_vs_vector": {k: round(full_avg[k] - vector_avg[k], 4) for k in full_avg},
            "cases": [
                {
                    "query": c.query,
                    "relevant": c.relevant,
                    "full": c.full,
                    "vector": c.vector,
                    "keyword": c.keyword,
                }
                for c in per_case_results
            ],
        }

    # ── 三条链路 ──────────────────────────────────────────────────────────────

    async def _full_ranking(self, query: str) -> List[str]:
        """线上链路：改写 + 并行召回 + 重排。"""
        try:
            result = await self._tm.search_with_rewrite(
                "knowledge_search", query, top_k=self._max_k
            )
        except Exception as ex:
            logger.warning(f"full 检索失败: {ex}")
            return []
        if not getattr(result, "success", False) or not isinstance(result.data, list):
            return []
        return self._titles(result.data)

    async def _vector_ranking(self, query: str) -> List[str]:
        """基线一：纯向量检索（无改写、无重排）。"""
        try:
            items = await self._kb.search_async(query, top_k=self._max_k)
        except Exception as ex:
            logger.warning(f"vector 检索失败: {ex}")
            return []
        return self._titles(items)

    def _keyword_ranking(self, query: str) -> List[str]:
        """基线二：纯字面匹配（字符二元组重叠），无向量、无语义。"""
        q_tokens = self._tokens(query)
        if not q_tokens:
            return []
        scored: List[Tuple[float, str]] = []
        for title, content in self._corpus.items():
            d_tokens = self._tokens(f"{title}{content}")
            overlap = len(q_tokens & d_tokens) / len(q_tokens)
            scored.append((overlap, title))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [title for _, title in scored[: self._max_k]]

    # ── 工具 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _titles(items: Sequence[Any]) -> List[str]:
        """从检索结果里抽 title 并去重，保留首次出现顺序（结果无稳定 doc id，只能按 title 标）。"""
        seen, titles = set(), []
        for item in items:
            if isinstance(item, dict):
                title = str(item.get("title", "")).strip()
            else:
                title = str(getattr(item, "title", "")).strip()
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        return titles

    @staticmethod
    def _tokens(text: str) -> set:
        text = re.sub(r"\s+", "", str(text))
        if len(text) < 2:
            return set(text)
        return {text[i : i + 2] for i in range(len(text) - 1)}


# ── RAG 回答准确率评测 ────────────────────────────────────────────────────────

class RagAnswerEvaluator:
    """用检索到的资料生成回答，再由 LLM-as-Judge 对照 gold 文档判准确性。"""

    GEN_PROMPT = """你是客服知识助手。请**仅依据**下面的资料回答用户问题，不要编造资料中没有的信息。
如果资料不足以回答，请说明资料中没有相关信息。

资料:
{context}

用户问题: {query}

请用简洁的中文回答。"""

    def __init__(
        self,
        tool_manager,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        judge: Optional[LLMJudge] = None,
        corpus: Optional[Dict[str, str]] = None,
        top_k: int = 5,
        threshold: float = 0.75,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model = model
        self._judge = judge or LLMJudge(self._client, model)
        self._tm = tool_manager
        self._corpus = corpus or {}
        self._top_k = top_k
        self._threshold = threshold

    async def evaluate(self, cases: List[RetrievalCase]) -> Dict[str, Any]:
        per_case: List[Dict[str, Any]] = []
        acc_scores: List[float] = []
        correct = 0

        for case in cases:
            if not case.query or not case.relevant_docs:
                continue
            gold_title = case.relevant_docs[0]
            chunks = await self._retrieve(case.query)
            if not chunks:
                per_case.append({"query": case.query, "gold": gold_title,
                                 "answer": "", "score": 0.0, "correct": False,
                                 "note": "无检索结果"})
                acc_scores.append(0.0)
                continue

            answer = await self._generate(case.query, chunks)
            gold_content = self._corpus.get(gold_title, "")
            scores = await self._judge.judge(case.query, answer, context=gold_content or None)
            score = 0.0 if scores.judge_failed else scores.accuracy
            passed = score >= self._threshold
            correct += 1 if passed else 0
            acc_scores.append(score)
            per_case.append({
                "query": case.query,
                "gold": gold_title,
                "retrieved": [c.get("title", "") for c in chunks],
                "answer": answer,
                "score": round(score, 4),
                "correct": passed,
                "judge_failed": scores.judge_failed,
            })

        n = len(per_case)
        return {
            "total": n,
            "answer_accuracy": round(correct / n, 4) if n else 0.0,
            "avg_accuracy_score": round(statistics.mean(acc_scores), 4) if acc_scores else 0.0,
            "threshold": self._threshold,
            "cases": per_case,
        }

    async def _retrieve(self, query: str) -> List[Dict[str, Any]]:
        try:
            result = await self._tm.search_with_rewrite(
                "knowledge_search", query, top_k=self._top_k
            )
        except Exception as ex:
            logger.warning(f"回答评测检索失败: {ex}")
            return []
        if not getattr(result, "success", False) or not isinstance(result.data, list):
            return []
        return [c for c in result.data if isinstance(c, dict)]

    async def _generate(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        context = "\n\n".join(
            f"[{c.get('title', '')}] {c.get('content', '')}" for c in chunks
        )
        prompt = self.GEN_PROMPT.format(context=context, query=query)
        prompt = prompt.encode("utf-8", errors="ignore").decode("utf-8")
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=512, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return extract_text_content(resp.content).strip()
        except Exception as ex:
            logger.warning(f"生成回答失败: {ex}")
            return ""

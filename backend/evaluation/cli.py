"""独立评测 CLI。

用法（在后端根目录执行）：
    set PYTHONIOENCODING=utf-8          # Windows 控制台
    python -m evaluation.cli --mode intent
    python -m evaluation.cli --mode intent --intent-data evaluation/datasets/intent_cases.json --out reports/eval_intent.json

为什么是独立脚本而不是走 HTTP /eval/run：
- 评测产物是"指标报告"，不是请求响应；
- 不需要起 FastAPI 服务，意图评测也不需要 Redis / ChromaDB；
- 报告落到文件，便于和上一次结果对比、写进简历。

本文件只负责"装配 + 跑 + 打印 + 落盘报告"，评测逻辑一律复用 evaluation/evaluator.py。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# 让脚本在任意 cwd 下都能 import 项目模块（core.* / agents.* / evaluation.*）
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv 未安装时退化为只读环境变量
    load_dotenv = None


def _setup_stdout() -> None:
    """Windows 控制台默认 GBK，中文/符号会报错，强制 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def _patch_max_tokens(floor: int = 2048, threshold: int = 600) -> None:
    """只给「结构化输出」调用兜底 max_tokens —— 不改项目源码。

    思考型模型（deepseek-v4-pro / -reasoner / -flash）会先输出 thinking 块，
    而 thinking 与正文共用同一个 max_tokens 预算。项目里 5 处结构化输出
    （判官/意图/RAG 改写/记忆压缩）写死 256，会被 thinking 吃光 → 只返回
    thinking 块、没有 text → json.loads("") 失败、静默降级。

    这里包装 SDK 的 create：max_tokens < threshold 的抬到 floor，
    不动 Agent 生成的 900–1200，避免回答变长变贵。
    """
    from anthropic.resources.messages import AsyncMessages

    orig = AsyncMessages.create
    if getattr(orig, "_carepilot_patched", False):
        return

    async def create(self, *args, **kwargs):
        mt = kwargs.get("max_tokens")
        if isinstance(mt, int) and mt < threshold:
            kwargs["max_tokens"] = floor
        return await orig(self, *args, **kwargs)

    create._carepilot_patched = True  # type: ignore[attr-defined]
    AsyncMessages.create = create  # type: ignore[assignment]


def _anthropic_cfg() -> Dict[str, Any]:
    """读取模型配置。逻辑与 api/main.py:_anthropic_cfg 保持一致，但不 import api.main。"""
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY —— 请在项目 .env 或环境变量里配置")
    cfg: Dict[str, Any] = {
        "api_key": key,
        "model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip(),
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    return cfg


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_report(out: Optional[str], mode: str, dataset: str, payload: Dict[str, Any]) -> str:
    reports_dir = _ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    if out:
        path = pathlib.Path(out)
        if not path.is_absolute():
            path = _ROOT / path
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = reports_dir / f"eval_{mode}_{ts}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(path)


# ── Phase 1：意图评测 ────────────────────────────────────────────────────────

async def run_intent(args, cfg: Dict[str, Any]) -> Dict[str, Any]:
    from core.intent_recognizer import IntentRecognizer
    from evaluation.evaluator import IntentEvaluator, IntentTestCase

    raw = _load_json(args.intent_data)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"意图数据集为空或格式不对：{args.intent_data}")
    if args.limit:
        raw = raw[: args.limit]

    cases = [IntentTestCase(**c) for c in raw]
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"], base_url=cfg.get("base_url"), model=cfg["model"]
    )
    evaluator = IntentEvaluator(recognizer)
    metrics = await evaluator.evaluate(cases)

    _print_intent(metrics, cfg)
    payload = {
        "mode": "intent",
        "timestamp": datetime.now().isoformat(),
        "dataset": str(args.intent_data),
        "model": cfg["model"],
        "total": metrics["total"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class": metrics["per_class"],
        "cases": metrics["cases"],
    }
    return payload


def _print_intent(m: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    print("=" * 64, flush=True)
    print(f"意图识别评测  |  模型: {cfg['model']}", flush=True)
    print("=" * 64, flush=True)
    print(f"用例数   : {m['total']}", flush=True)
    print(f"正确数   : {m['correct']}", flush=True)
    print(f"准确率   : {m['accuracy']:.4f}", flush=True)
    print(f"宏平均F1 : {m['macro_f1']:.4f}", flush=True)
    print("-" * 64, flush=True)
    print(f"{'标签':<20}{'precision':>10}{'recall':>10}{'f1':>10}", flush=True)
    for label, s in sorted(m["per_class"].items()):
        print(f"{label:<20}{s['precision']:>10.4f}{s['recall']:>10.4f}{s['f1']:>10.4f}", flush=True)
    print("-" * 64, flush=True)
    wrong = [c for c in m["cases"] if c["expected"] != c["predicted"]]
    print(f"错判 {len(wrong)} 条：", flush=True)
    for c in wrong:
        print(f"  [{c['expected']} -> {c['predicted']}]  {c['message']}", flush=True)
    print("=" * 64, flush=True)


# ── Phase 2：端到端对话评测 ──────────────────────────────────────────────────

async def run_e2e(args, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """端到端：真实调 Orchestrator 生成回复，再用 LLM-as-Judge 打分。

    默认以「无 RAG」模式装配（rag_tool_manager=None），因为 agents/tools.py
    不 import chromadb，且 build_shared_rag_tools(None) 会返回降级错误字典——
    于是本模式不需要 Docker / ChromaDB 也能跑通判官闭环。
    """
    from agents.agent_orchestrator import AgentOrchestrator
    from core.intent_recognizer import IntentRecognizer
    from evaluation.evaluator import EndToEndEvaluator

    raw = _load_json(args.dialog_data)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"对话数据集为空或格式不对：{args.dialog_data}")
    if args.limit:
        raw = raw[: args.limit]

    # 判官独立模型：JUDGE_MODEL 未设时回落答题模型（消除自评偏差靠它）
    judge_model = os.getenv("JUDGE_MODEL", "").strip() or cfg["model"]

    recognizer = IntentRecognizer(
        api_key=cfg["api_key"], base_url=cfg.get("base_url"), model=cfg["model"]
    )
    orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        rag_tool_manager=None,   # 无 RAG；接了 Chroma 后把真实 tool_manager 传进来即可
    )
    evaluator = EndToEndEvaluator(
        orchestrator=orchestrator,
        recognizer=recognizer,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=judge_model,       # 只喂给 LLMJudge，答题侧不受影响
        baseline_path=None,      # 关键：不覆盖 data/eval/baseline.json
    )
    report = await evaluator.run(dialog_cases=raw)

    _print_e2e(report, cfg, judge_model)
    payload = {
        "mode": "e2e",
        "timestamp": report.timestamp,
        "dataset": str(args.dialog_data),
        "model": cfg["model"],
        "judge_model": judge_model,
        "total": report.total,
        "passed": report.passed,
        "pass_rate": report.pass_rate,
        "avg_scores": report.avg_scores,
        "regressions": report.regressions,
        "recommendations": report.recommendations,
        "cases": [r.__dict__ for r in report.results],
        "sessions": [s.__dict__ for s in getattr(report, "sessions", [])],
    }
    return payload


def _print_e2e(report, cfg: Dict[str, Any], judge_model: str) -> None:
    print("=" * 64, flush=True)
    print(f"端到端对话评测  |  答题: {cfg['model']}  判官: {judge_model}", flush=True)
    print("=" * 64, flush=True)
    print(f"用例数   : {report.total}   通过: {report.passed}   通过率: {report.pass_rate:.4f}", flush=True)
    print("-" * 64, flush=True)
    print("平均分（LLM-as-Judge，0-1）：", flush=True)
    for k, v in report.avg_scores.items():
        print(f"  {k:<16}{v:>8.4f}", flush=True)
    print("-" * 64, flush=True)
    for r in report.results:
        flag = "OK " if r.passed else "!! "
        jf = " [judge_failed]" if r.metadata.get("judge_failed") else ""
        print(f"{flag}{r.test_id:<22} overall={r.scores.get('overall', 0):.3f}{jf}", flush=True)

    # 会话级评测（仅标注了 goal 的多轮用例）
    sessions = getattr(report, "sessions", None) or []
    if sessions:
        print("-" * 64, flush=True)
        print(f"会话级评测（{len(sessions)} 通会话）", flush=True)
        ses_keys = ("session_pass_rate", "session_resolved_rate", "session_coherent_rate",
                    "no_reask_rate", "intent_trajectory_acc", "context_recall_rate")
        for k in ses_keys:
            if k in report.avg_scores:
                print(f"  {k:<24}{report.avg_scores[k]:>8.4f}", flush=True)
        print("-" * 64, flush=True)
        for s in sessions:
            flag = "OK " if s.passed else "!! "
            print(f"{flag}{s.session_id:<22} {s.detail}", flush=True)
    if report.regressions:
        print("-" * 64, flush=True)
        print("回归（对比基线，退化 >5%）：", flush=True)
        for reg in report.regressions:
            print(f"  {reg}", flush=True)
    if report.recommendations:
        print("-" * 64, flush=True)
        print("优化建议：", flush=True)
        for rec in report.recommendations:
            print(f"  - {rec}", flush=True)
    print("=" * 64, flush=True)


# ── Phase 3：检索 + RAG 回答评测 ─────────────────────────────────────────────

def _load_corpus() -> Dict[str, str]:
    """加载评测语料（data/demo_docs/kb_carepilot_*.json），返回 {title: content}。"""
    demo_dir = _ROOT / "data" / "demo_docs"
    corpus: Dict[str, str] = {}
    for path in sorted(demo_dir.glob("kb_carepilot_*.json")):
        for doc in _load_json(str(path)):
            title = str(doc.get("title", "")).strip()
            content = str(doc.get("content", "")).strip()
            if title and content and title not in corpus:
                corpus[title] = content
    return corpus


def _build_eval_kb(corpus: Dict[str, str], embedding_function=None):
    """构建**隔离**的评测知识库：独立目录 + 独立 collection，避免污染线上 knowledge_base。

    强制走嵌入式模式（bad host 让 HttpClient 心跳失败）——评测只依赖本地 ChromaDB，
    不需要 Docker；每次运行前删掉旧 collection 重新导入，保证可复现。

    embedding_function 传入自定义 embedding（评测用中文模型）；None 则用 ChromaDB 默认。
    """
    from mcp.knowledge_base import KnowledgeBase

    eval_dir = os.getenv("CAREPILOT_EVAL_CHROMA", str(_ROOT / "data" / "chroma_eval"))
    kb = KnowledgeBase(chroma_host="127.0.0.1", chroma_port=1, chroma_path=eval_dir)

    name = "kb_eval"
    try:
        kb._client.delete_collection(name)
    except Exception:
        pass
    kwargs: Dict[str, Any] = {"metadata": {"description": "CarePilot RAG 评测语料"}}
    if embedding_function is not None:
        kwargs["embedding_function"] = embedding_function
    kb._collection = kb._client.get_or_create_collection(name, **kwargs)
    kb.add_documents([{"title": t, "content": c} for t, c in corpus.items()])
    return kb


def _make_eval_embedding():
    """按 CAREPILOT_EVAL_EMBED 决定评测用 embedding。返回 (ef, label)。

    默认 zh：用中文模型（bge-small-zh），回答「换上合适的中文 embedding 能到多少」。
    default：用 ChromaDB 默认英文模型（与线上一致），用于对照。
    """
    mode = os.getenv("CAREPILOT_EVAL_EMBED", "zh").strip().lower()
    if mode == "default":
        return None, "all-MiniLM-L6-v2(ChromaDB默认/线上同款)"
    from evaluation.embedding_zh import make_eval_embedding_function

    ef = make_eval_embedding_function()
    return ef, f"{ef.get_config()['model_name']}(评测专用，未改线上)"


async def run_rag(args, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 3：检索指标（recall@k/hit_rate/MRR + 基线对比）+ RAG 回答准确率。"""
    from evaluation.evaluator import LLMJudge
    from evaluation.retrieval_eval import (
        RagAnswerEvaluator,
        RetrievalCase,
        RetrievalEvaluator,
    )
    from mcp.tool_manager import MCPToolManager, Tool

    corpus = _load_corpus()
    if not corpus:
        raise ValueError("评测语料为空 —— 检查 data/demo_docs/kb_carepilot_*.json")

    raw = _load_json(args.rag_data)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"检索数据集为空或格式不对：{args.rag_data}")
    if args.limit:
        raw = raw[: args.limit]
    cases = [RetrievalCase.from_dict(c) for c in raw]

    eval_ef, embed_label = _make_eval_embedding()
    kb = _build_eval_kb(corpus, eval_ef)
    print(f"评测知识库已就绪: {kb.doc_count} 个片段 / {len(corpus)} 篇文档 | embedding: {embed_label}", flush=True)

    # 与 api/main.py 一致的接线；cache_ttl=0 保证每条 query 都真实检索（不被缓存污染）
    tool_manager = MCPToolManager(
        api_key=cfg["api_key"], base_url=cfg.get("base_url"), model=cfg["model"]
    )
    tool_manager.register(Tool(
        name="knowledge_search",
        description="搜索知识库（基于 ChromaDB 向量检索）",
        handler=kb.search_handler,
        schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
        cache_ttl=0.0,
        supports_rerank=True,
    ))

    # 1. 检索指标 + 两条基线对比
    retrieval = await RetrievalEvaluator(tool_manager, kb, corpus).evaluate(cases)
    _print_rag_retrieval(retrieval, cfg, embed_label)

    # 2. RAG 回答准确率（判官用独立模型）
    judge_model = os.getenv("JUDGE_MODEL", "").strip() or cfg["model"]
    answer_cases = cases[: args.answer_limit] if args.answer_limit else cases
    judge = LLMJudge(_make_client(cfg, judge_model), judge_model)
    answer = await RagAnswerEvaluator(
        tool_manager,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        judge=judge,
        corpus=corpus,
        top_k=args.top_k,
    ).evaluate(answer_cases)
    _print_rag_answer(answer, cfg, judge_model)

    return {
        "mode": "rag",
        "timestamp": datetime.now().isoformat(),
        "dataset": str(args.rag_data),
        "model": cfg["model"],
        "judge_model": judge_model,
        "corpus_docs": len(corpus),
        "eval_embedding": embed_label,
        "retrieval": retrieval,
        "answer": answer,
        "note": "ground truth 为反向合成（每篇文档推导出问题，该文档为 gold），非人工标注；"
                "评测用中文 embedding，线上知识库仍为 ChromaDB 默认英文模型",
    }


def _make_client(cfg: Dict[str, Any], model: str):
    from anthropic import AsyncAnthropic

    kwargs: Dict[str, Any] = {"api_key": cfg["api_key"]}
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]
    return AsyncAnthropic(**kwargs)


def _print_rag_retrieval(m: Dict[str, Any], cfg: Dict[str, Any], embed_label: str = "") -> None:
    print("=" * 72, flush=True)
    print(f"Phase 3 检索评测  |  模型: {cfg['model']}", flush=True)
    print(f"用例数: {m['dataset_size']}   语料: {m['corpus_docs']} 篇   embedding: {embed_label}", flush=True)
    print("=" * 72, flush=True)
    ks = m["ks"]
    metrics = [f"recall@{k}" for k in ks] + [f"hit_rate@{k}" for k in ks] + ["mrr"]
    header = f"{'链路':<18}" + "".join(f"{mt:>12}" for mt in metrics)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for label, key in (("full(改写+重排)", "full"),
                       ("baseline_vector", "baseline_vector"),
                       ("baseline_keyword", "baseline_keyword")):
        row = f"{label:<18}"
        for mt in metrics:
            row += f"{m[key].get(mt, 0.0):>12.4f}"
        print(row, flush=True)
    print("-" * len(header), flush=True)
    for label, key in (("Δ vs keyword", "delta_vs_keyword"), ("Δ vs vector", "delta_vs_vector")):
        row = f"{label:<18}"
        for mt in metrics:
            row += f"{m[key].get(mt, 0.0):>+12.4f}"
        print(row, flush=True)
    print("=" * 72, flush=True)


def _print_rag_answer(m: Dict[str, Any], cfg: Dict[str, Any], judge_model: str) -> None:
    print("=" * 72, flush=True)
    print(f"RAG 回答准确率  |  答题: {cfg['model']}  判官: {judge_model}", flush=True)
    print("=" * 72, flush=True)
    print(f"用例数     : {m['total']}", flush=True)
    print(f"回答准确率 : {m['answer_accuracy']:.4f}  （判官 accuracy ≥ {m['threshold']} 记为正确）", flush=True)
    print(f"平均准确性分: {m['avg_accuracy_score']:.4f}", flush=True)
    failed = [c for c in m["cases"] if not c.get("correct")]
    print(f"未通过 {len(failed)} 条：", flush=True)
    for c in failed[:15]:
        print(f"  [{c['score']:.2f}] gold={c['gold']}  Q: {c['query']}", flush=True)
    print("=" * 72, flush=True)


# ── 入口 ─────────────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m evaluation.cli", description="独立评测 CLI")
    p.add_argument("--mode", default="intent", choices=["intent", "e2e", "rag", "all"])
    p.add_argument("--intent-data", default="evaluation/datasets/intent_cases.json")
    p.add_argument("--dialog-data", default="evaluation/datasets/dialog_cases.json")
    p.add_argument("--rag-data", default="evaluation/datasets/rag_cases.json")
    p.add_argument("--out", default=None, help="报告路径；缺省写到 reports/eval_<mode>_<ts>.json")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部），用于冒烟")
    p.add_argument("--answer-limit", type=int, default=0,
                   help="RAG 回答准确率只跑前 N 条（0=全部）；该阶段每条需 2 次 LLM 调用，较慢")
    p.add_argument("--generate", type=int, default=0, help="预留：扩写数据集（未实现）")
    return p.parse_args(argv)


async def _main(args) -> int:
    _setup_stdout()
    if load_dotenv is not None:
        load_dotenv(_ROOT / ".env")
    _patch_max_tokens()  # 思考型模型下给小预算结构化输出兜底

    cfg = _anthropic_cfg()
    modes = ["intent", "e2e", "rag"] if args.mode == "all" else [args.mode]

    for mode in modes:
        if mode == "intent":
            payload = await run_intent(args, cfg)
            path = _write_report(args.out, "intent", args.intent_data, payload)
            print(f"报告已写入: {path}", flush=True)
        elif mode == "e2e":
            payload = await run_e2e(args, cfg)
            path = _write_report(args.out, "e2e", args.dialog_data, payload)
            print(f"报告已写入: {path}", flush=True)
        elif mode == "rag":
            payload = await run_rag(args, cfg)
            path = _write_report(args.out, "rag", args.rag_data, payload)
            print(f"报告已写入: {path}", flush=True)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        return asyncio.run(_main(args))
    except KeyboardInterrupt:
        return 130
    except Exception as ex:
        print(f"评测失败: {ex}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

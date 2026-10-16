"""
亮点：端到端 Agent 评测框架

核心问题：如何评测端到端 Agent？

评测维度：
  1. 意图识别准确率 —— 预测意图 vs 标注意图，计算 Accuracy / F1
  2. 响应质量评分 —— 用 LLM 作为评判者（LLM-as-Judge），
     从相关性、准确性、完整性、有用性四个维度打分
  3. 端到端对话评测 —— 模拟完整多轮对话，评估整体体验
  4. 回归测试 —— 与历史基线对比，防止性能退化

LLM-as-Judge 是评测 Agent 质量的关键技术：
  人工标注成本高、主观性强；用 LLM 评判可以规模化、可重复。
"""
import asyncio
import json
import logging
import pathlib
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content

from core.intent_recognizer import IntentCategory, IntentRecognizer

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class IntentTestCase:
    message:          str
    expected_intent:  str
    context:          Optional[Dict[str, Any]] = None


@dataclass
class QualityScores:
    """LLM-as-Judge 评分结果。"""
    relevance:    float   # 相关性：回答是否针对问题
    accuracy:     float   # 准确性：信息是否正确
    completeness: float   # 完整性：是否完整解决问题
    helpfulness:  float   # 有用性：用户是否能据此行动
    judge_failed: bool = False
    error: Optional[str] = None

    @property
    def overall(self) -> float:
        return statistics.mean([self.relevance, self.accuracy, self.completeness, self.helpfulness])


@dataclass
class SessionScores:
    """会话级 LLM-as-Judge 结果（评估整通会话，而非单轮）。"""
    resolved:     bool          # 会话目标是否达成
    coherent:     bool          # 上下文是否连贯、无自相矛盾/答非所问
    reasked:      bool          # 是否重复索要用户已提供的信息（True = 有问题）
    reason:       str = ""
    judge_failed: bool = False
    error:        Optional[str] = None


@dataclass
class EvalResult:
    test_id:    str
    passed:     bool
    scores:     Dict[str, float]
    detail:     str = ""
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionResult:
    """一通多轮会话的聚合评测结果。"""
    session_id:            str
    goal:                  str
    resolved:              bool
    coherent:              bool
    reasked:               bool
    passed:                bool          # resolved and coherent and not reasked
    intent_trajectory_acc: Optional[float]  # 逐轮意图 vs expected_intents；未标注为 None
    context_recall_rate:   Optional[float]  # must_recall 命中率；未标注为 None
    detail:                str = ""
    transcript:            List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EvalReport:
    """评测报告。"""
    timestamp:        str
    total:            int
    passed:           int
    pass_rate:        float
    avg_scores:       Dict[str, float]
    regressions:      List[str]          # 相比基线退化的指标
    recommendations:  List[str]
    results:          List[EvalResult]
    sessions:         List[SessionResult] = field(default_factory=list)  # 会话级评测明细（无既有用例时为空）


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

class LLMJudge:
    """
    用 LLM 评判 Agent 响应质量。

    为什么用 LLM 而不是人工？
    - 可规模化：数千条测试用例自动评测
    - 可重复：相同输入得到稳定评分
    - 多维度：同时评估相关性、准确性等多个维度

    注意：LLM Judge 本身也有偏差，建议定期用人工标注校准。
    """

    JUDGE_PROMPT = """你是一个客服质量评估专家。请对以下客服响应进行评分。

用户问题: {question}
Agent 响应: {response}
{context_section}

请从以下四个维度评分（0.0-1.0），返回 JSON：
- relevance: 响应是否直接针对用户问题（0=完全无关，1=完全相关）
- accuracy: 信息是否准确无误（0=明显错误，1=完全正确）
- completeness: 是否完整解决了用户需求（0=完全没解决，1=完全解决）
- helpfulness: 用户能否据此采取行动（0=毫无帮助，1=非常有帮助）

只返回 JSON，例如: {{"relevance": 0.9, "accuracy": 0.8, "completeness": 0.7, "helpfulness": 0.85}}"""

    SESSION_PROMPT = """你是一个客服会话质量评估专家。请评估以下**多轮**客服会话是否达成目标。

会话目标: {goal}

完整对话记录:
{transcript}

请返回 JSON：
- resolved: 会话目标是否达成（true/false）
- coherent: 上下文是否连贯，有无自相矛盾、答非所问或遗忘前文（true/false）
- reasked: Agent 是否重复索要了用户已经提供过的信息（true/false）
- reason: 一句话说明判定依据

只返回 JSON，例如: {{"resolved": true, "coherent": true, "reasked": false, "reason": "已按用户给的订单号完成退款并说明到账时间"}}"""

    def __init__(self, client: AsyncAnthropic, model: str):
        self._client = client
        self._model  = model

    async def judge(
        self,
        question: str,
        response: str,
        context: Optional[str] = None,
    ) -> QualityScores:
        ctx_section = f"背景信息: {context}" if context else ""
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            response=response,
            context_section=ctx_section,
        )
        prompt = self._clean_text(prompt)
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            return QualityScores(
                relevance=float(data.get("relevance", 0.5)),
                accuracy=float(data.get("accuracy", 0.5)),
                completeness=float(data.get("completeness", 0.5)),
                helpfulness=float(data.get("helpfulness", 0.5)),
            )
        except Exception as ex:
            logger.warning(f"LLM Judge 失败: {ex}")
            return QualityScores(
                0.5, 0.5, 0.5, 0.5,
                judge_failed=True,
                error=str(ex),
            )

    async def judge_session(
        self,
        goal: str,
        transcript: List[Dict[str, Any]],
    ) -> SessionScores:
        """对整通多轮会话做一次判断：目标是否达成、是否连贯、是否重复问。

        transcript 元素形如 {"role": "user"|"assistant", "content": "..."}；
        意图等额外字段打印时会被忽略。
        """
        goal = self._clean_text(goal) or "(未标注明确目标)"
        lines = []
        for m in transcript:
            role = m.get("role", "?")
            content = self._clean_text(m.get("content", ""))
            lines.append(f"{role}: {content}")
        body = "\n".join(lines)
        prompt = self.SESSION_PROMPT.format(goal=goal, transcript=body)
        prompt = self._clean_text(prompt)
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            return SessionScores(
                resolved=self._as_bool(data.get("resolved"), default=False),
                coherent=self._as_bool(data.get("coherent"), default=True),
                reasked=self._as_bool(data.get("reasked"), default=False),
                reason=self._clean_text(str(data.get("reason", ""))),
            )
        except Exception as ex:
            logger.warning(f"会话级 Judge 失败: {ex}")
            return SessionScores(
                resolved=False, coherent=False, reasked=False,
                judge_failed=True, error=str(ex),
            )

    @staticmethod
    def _as_bool(value: Any, default: bool) -> bool:
        """LLM 可能把布尔写成字符串/中文，统一解析。"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("true", "yes", "1", "是", "有"):
                return True
            if v in ("false", "no", "0", "否", "无", "没有"):
                return False
        return default

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")


# ── 意图识别评测 ──────────────────────────────────────────────────────────────

class IntentEvaluator:
    """评测意图识别的准确率和 F1。"""

    def __init__(self, recognizer: IntentRecognizer):
        self._recognizer = recognizer

    async def evaluate(self, cases: List[IntentTestCase]) -> Dict[str, Any]:
        predictions, ground_truth = [], []
        case_details: List[Dict[str, Any]] = []

        for case in cases:
            result = await self._recognizer.recognize(case.message)
            predicted = result.intent.value
            predictions.append(predicted)
            ground_truth.append(case.expected_intent)
            case_details.append({
                "message": case.message,
                "expected": case.expected_intent,
                "predicted": predicted,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
            })

        # 纯 Python 计算指标
        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        accuracy = correct / len(predictions) if predictions else 0.0

        # 每类 F1
        labels = sorted(set(ground_truth + predictions))
        per_class: Dict[str, Dict[str, float]] = {}
        for label in labels:
            tp = sum(p == label and g == label for p, g in zip(predictions, ground_truth))
            fp = sum(p == label and g != label for p, g in zip(predictions, ground_truth))
            fn = sum(p != label and g == label for p, g in zip(predictions, ground_truth))
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_class[label] = {"precision": prec, "recall": rec, "f1": f1}

        macro_f1 = statistics.mean(v["f1"] for v in per_class.values()) if per_class else 0.0

        return {
            "accuracy":   round(accuracy, 4),
            "macro_f1":   round(macro_f1, 4),
            "per_class":  per_class,
            "total":      len(cases),
            "correct":    correct,
            "cases":      case_details,
        }


# ── 端到端评测器 ──────────────────────────────────────────────────────────────

class EndToEndEvaluator:
    """
    端到端 Agent 评测。

    评测流程：
      1. 运行意图识别评测（准确率/F1）
      2. 运行对话质量评测（LLM-as-Judge）
      3. 与历史基线对比（回归检测）
      4. 生成可操作的优化建议
    """

    # 质量及格线
    PASS_THRESHOLD = 0.75

    def __init__(
        self,
        orchestrator,
        recognizer: IntentRecognizer,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        baseline_path: Optional[str] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncAnthropic(**kwargs)

        self._orchestrator     = orchestrator
        self._judge            = LLMJudge(client, model)
        self._intent_evaluator = IntentEvaluator(recognizer)
        self._history:         List[EvalReport] = []
        self._baseline_path = pathlib.Path(baseline_path) if baseline_path else None
        self._baseline: Optional[EvalReport] = self._load_baseline()

    async def run(
        self,
        intent_cases:    Optional[List[IntentTestCase]] = None,
        dialog_cases:    Optional[List[Dict[str, Any]]] = None,
    ) -> EvalReport:
        """
        运行完整评测。

        intent_cases: 意图识别测试用例
        dialog_cases:
          - 单轮: [{"question": "..."}]
          - 多轮: [{"turns": ["第一轮", "第二轮", ...]}]
        """
        results: List[EvalResult] = []
        all_scores: Dict[str, List[float]] = {
            "relevance": [], "accuracy": [], "completeness": [], "helpfulness": []
        }

        # 1. 意图识别评测
        intent_metrics: Dict[str, Any] = {}
        if intent_cases:
            intent_metrics = await self._intent_evaluator.evaluate(intent_cases)
            passed = intent_metrics["accuracy"] >= self.PASS_THRESHOLD
            results.append(EvalResult(
                test_id="intent_recognition",
                passed=passed,
                scores={"accuracy": intent_metrics["accuracy"], "macro_f1": intent_metrics["macro_f1"]},
                detail=f"准确率 {intent_metrics['accuracy']:.1%}，Macro-F1 {intent_metrics['macro_f1']:.3f}",
                metadata={
                    "total": intent_metrics.get("total", 0),
                    "correct": intent_metrics.get("correct", 0),
                    "cases": intent_metrics.get("cases", []),
                },
            ))

        # 2. 对话质量评测（调用 orchestrator 产出回复，再用 LLM Judge 评分）
        session_results: List[SessionResult] = []
        if dialog_cases:
            for i, case in enumerate(dialog_cases):
                case_results = await self._evaluate_dialog_case(case, i)
                results.extend(case_results)
                for r in case_results:
                    for k in all_scores:
                        if k in r.scores:
                            all_scores[k].append(r.scores[k])
                # 标注了 goal 的用例（且已跑出逐轮结果）额外做会话级评测
                if case.get("goal") and case_results:
                    session_results.append(await self._evaluate_session(case, i, case_results))

        # 3. 汇总
        avg_scores = {
            k: round(statistics.mean(v), 4) for k, v in all_scores.items() if v
        }
        if intent_metrics:
            avg_scores["intent_accuracy"] = intent_metrics["accuracy"]
        # 会话级指标并入 avg_scores（同名 key 不冲突），使其同样参与基线回归检测
        avg_scores.update(self._aggregate_sessions(session_results))

        passed_count = sum(1 for r in results if r.passed)
        pass_rate    = passed_count / len(results) if results else 0.0

        # 4. 回归检测
        regressions = self._detect_regressions(avg_scores)

        # 5. 优化建议
        recommendations = self._recommendations(avg_scores, intent_metrics)

        report = EvalReport(
            timestamp=datetime.now().isoformat(),
            total=len(results),
            passed=passed_count,
            pass_rate=round(pass_rate, 4),
            avg_scores=avg_scores,
            regressions=regressions,
            recommendations=recommendations,
            results=results,
            sessions=session_results,
        )
        self._history.append(report)
        self._save_baseline(report)
        return report

    async def _evaluate_dialog_case(self, case: Dict[str, Any], case_idx: int) -> List[EvalResult]:
        """评测单轮或多轮对话用例。"""
        from agents.agent_orchestrator import Request as OrcReq

        questions = self._dialog_turns(case)
        if not questions:
            return []

        conv_id = str(case.get("conv_id") or f"eval_{case_idx}")
        user_id = str(case.get("user_id") or "eval_user")
        history: List[Dict[str, str]] = []
        results: List[EvalResult] = []

        for turn_idx, question in enumerate(questions):
            context = self._history_context(history)
            orch_req = OrcReq(
                message=question,
                user_id=user_id,
                conv_id=conv_id,
                context=context,
                history=history[-6:] if history else None,
            )
            orch_result = await self._orchestrator.run(orch_req)
            actual_answer = orch_result.response

            scores = await self._judge.judge(question, actual_answer, context=context or None)
            passed = scores.overall >= self.PASS_THRESHOLD

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": actual_answer})

            test_id = f"dialog_{case_idx}" if len(questions) == 1 else f"dialog_{case_idx}_turn_{turn_idx}"
            results.append(EvalResult(
                test_id=test_id,
                passed=passed,
                scores={
                    "relevance": scores.relevance,
                    "accuracy": scores.accuracy,
                    "completeness": scores.completeness,
                    "helpfulness": scores.helpfulness,
                    "overall": scores.overall,
                },
                detail=f"Q: {question[:30]}... → 综合评分 {scores.overall:.3f}",
                metadata={
                    "question": question,
                    "response": actual_answer,
                    "agent_type": orch_result.agent_type.value,
                    "intent": orch_result.intent.value if orch_result.intent else None,
                    "turn": turn_idx,
                    "conv_id": conv_id,
                    "judge_failed": scores.judge_failed,
                    "judge_error": scores.error,
                },
            ))

        return results

    async def _evaluate_session(
        self,
        case: Dict[str, Any],
        case_idx: int,
        turn_results: List[EvalResult],
    ) -> SessionResult:
        """在逐轮结果之上做会话级评测（目标达成 / 连贯性 / 是否重复问）。"""
        conv_id  = str(case.get("conv_id") or f"eval_{case_idx}")
        goal     = str(case.get("goal") or "")
        expected = case.get("expected_intents") or []
        must_recall = case.get("must_recall") or []

        transcript: List[Dict[str, Any]] = []
        for r in turn_results:
            transcript.append({"role": "user", "content": r.metadata.get("question", "")})
            transcript.append({"role": "assistant", "content": r.metadata.get("response", "")})

        scores = await self._judge.judge_session(goal, transcript)

        # 逐轮意图轨迹准确率（未标注不惩罚，记 None）
        pred_intents = [r.metadata.get("intent") for r in turn_results]
        intent_traj: Optional[float] = None
        if expected:
            hits = sum(1 for p, e in zip(pred_intents, expected) if p == e)
            intent_traj = hits / len(expected)

        context_recall = self._context_recall(turn_results, must_recall)

        passed = scores.resolved and scores.coherent and not scores.reasked
        detail = (
            f"目标{'达成' if scores.resolved else '未达成'} / "
            f"上下文{'连贯' if scores.coherent else '断裂'} / "
            f"{'重复问' if scores.reasked else '未重复问'}"
        )
        if scores.reason:
            detail += f" —— {scores.reason}"

        return SessionResult(
            session_id=conv_id,
            goal=goal,
            resolved=scores.resolved,
            coherent=scores.coherent,
            reasked=scores.reasked,
            passed=passed,
            intent_trajectory_acc=round(intent_traj, 4) if intent_traj is not None else None,
            context_recall_rate=round(context_recall, 4) if context_recall is not None else None,
            detail=detail,
            transcript=transcript,
        )

    @staticmethod
    def _context_recall(
        turn_results: List[EvalResult],
        must_recall: List[Dict[str, Any]],
    ) -> Optional[float]:
        """检查第 N 轮及之后，Agent 回复是否提到了应记住的早期实体。"""
        if not must_recall:
            return None
        hits = 0
        for item in must_recall:
            try:
                start = int(item.get("turn", 0))
            except (TypeError, ValueError):
                start = 0
            needles = [str(n) for n in item.get("must_mention", [])]
            if not needles:
                continue
            haystack = "".join(
                str(r.metadata.get("response", ""))
                for r in turn_results
                if int(r.metadata.get("turn", 0)) >= start
            )
            if all(n in haystack for n in needles):
                hits += 1
        return hits / len(must_recall)

    @staticmethod
    def _aggregate_sessions(sessions: List[SessionResult]) -> Dict[str, float]:
        if not sessions:
            return {}
        n = len(sessions)
        metrics = {
            "session_pass_rate":     round(sum(1 for s in sessions if s.passed) / n, 4),
            "session_resolved_rate": round(sum(1 for s in sessions if s.resolved) / n, 4),
            "session_coherent_rate": round(sum(1 for s in sessions if s.coherent) / n, 4),
            "no_reask_rate":         round(sum(1 for s in sessions if not s.reasked) / n, 4),
        }
        traj = [s.intent_trajectory_acc for s in sessions if s.intent_trajectory_acc is not None]
        if traj:
            metrics["intent_trajectory_acc"] = round(statistics.mean(traj), 4)
        recall = [s.context_recall_rate for s in sessions if s.context_recall_rate is not None]
        if recall:
            metrics["context_recall_rate"] = round(statistics.mean(recall), 4)
        return metrics

    @staticmethod
    def _dialog_turns(case: Dict[str, Any]) -> List[str]:
        turns = case.get("turns")
        if isinstance(turns, list):
            return [str(t) for t in turns if str(t).strip()]
        question = case.get("question")
        return [str(question)] if question else []

    @staticmethod
    def _history_context(history: List[Dict[str, str]]) -> str:
        if not history:
            return ""
        lines = [f"{m['role']}: {m['content']}" for m in history[-8:]]
        return "[评测多轮历史]\n" + "\n".join(lines)

    def _detect_regressions(self, current: Dict[str, float]) -> List[str]:
        """与上一次评测对比，找出退化超过 5% 的指标。"""
        prev_report = self._history[-1] if self._history else self._baseline
        if prev_report is None:
            return []
        prev = prev_report.avg_scores
        regressions = []
        for metric, value in current.items():
            if metric in prev and prev[metric] > 0:
                delta = (value - prev[metric]) / prev[metric]
                if delta < -0.05:
                    regressions.append(
                        f"{metric}: {prev[metric]:.3f} → {value:.3f} (退化 {abs(delta):.1%})"
                    )
        return regressions

    def _recommendations(
        self,
        scores: Dict[str, float],
        intent_metrics: Dict[str, Any],
    ) -> List[str]:
        recs = []
        if scores.get("intent_accuracy", 1.0) < 0.90:
            recs.append("意图识别准确率 < 90%：增加 Few-shot 示例，或对低 F1 的意图类别补充训练数据")
        if scores.get("relevance", 1.0) < 0.75:
            recs.append("相关性偏低：检查 Agent system_prompt，确保 Agent 聚焦于用户问题")
        if scores.get("completeness", 1.0) < 0.75:
            recs.append("完整性偏低：Agent 可能过早结束回答，考虑在 prompt 中要求提供完整解决方案")
        if scores.get("helpfulness", 1.0) < 0.75:
            recs.append("有用性偏低：回答可能过于抽象，考虑要求 Agent 提供具体操作步骤")
        if not recs:
            recs.append("所有指标均达标，继续保持")
        return recs

    @property
    def history(self) -> List[EvalReport]:
        return self._history

    def _load_baseline(self) -> Optional[EvalReport]:
        if not self._baseline_path or not self._baseline_path.exists():
            return None
        try:
            data = json.loads(self._baseline_path.read_text(encoding="utf-8"))
            return self._report_from_dict(data)
        except Exception as ex:
            logger.warning(f"读取评测基线失败: {ex}")
            return None

    def _save_baseline(self, report: EvalReport) -> None:
        if not self._baseline_path:
            return
        try:
            self._baseline_path.parent.mkdir(parents=True, exist_ok=True)
            self._baseline_path.write_text(
                json.dumps(asdict(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._baseline = report
        except Exception as ex:
            logger.warning(f"保存评测基线失败: {ex}")

    @staticmethod
    def _report_from_dict(data: Dict[str, Any]) -> EvalReport:
        return EvalReport(
            timestamp=data.get("timestamp", ""),
            total=int(data.get("total", 0)),
            passed=int(data.get("passed", 0)),
            pass_rate=float(data.get("pass_rate", 0.0)),
            avg_scores=dict(data.get("avg_scores", {})),
            regressions=list(data.get("regressions", [])),
            recommendations=list(data.get("recommendations", [])),
            results=[
                EvalResult(
                    test_id=r.get("test_id", ""),
                    passed=bool(r.get("passed", False)),
                    scores=dict(r.get("scores", {})),
                    detail=r.get("detail", ""),
                    metadata=dict(r.get("metadata", {})),
                )
                for r in data.get("results", [])
            ],
            sessions=[
                SessionResult(
                    session_id=s.get("session_id", ""),
                    goal=s.get("goal", ""),
                    resolved=bool(s.get("resolved", False)),
                    coherent=bool(s.get("coherent", False)),
                    reasked=bool(s.get("reasked", False)),
                    passed=bool(s.get("passed", False)),
                    intent_trajectory_acc=s.get("intent_trajectory_acc"),
                    context_recall_rate=s.get("context_recall_rate"),
                    detail=s.get("detail", ""),
                    transcript=list(s.get("transcript", [])),
                )
                for s in data.get("sessions", [])
            ],
        )


# ── 内置测试用例（开箱即用）──────────────────────────────────────────────────

DEFAULT_INTENT_CASES: List[IntentTestCase] = [
    IntentTestCase("我的订单什么时候到？",       "logistics"),
    IntentTestCase("帮我取消订单",               "request"),
    IntentTestCase("你们服务太差了！",            "complaint"),
    IntentTestCase("应用一直报500错误",           "technical_crash"),
    IntentTestCase("为什么扣了两次款？",          "payment_issue"),
    IntentTestCase("我要投诉，转人工！",          "human_handoff"),
    IntentTestCase("你好",                        "greeting"),
    IntentTestCase("修改我的邮箱地址",            "account"),
    IntentTestCase("帮我开发票",                  "invoice"),
    IntentTestCase("退款多久到账？",              "refund"),
    IntentTestCase("登录一直报401",               "technical_login"),
]

DEFAULT_DIALOG_CASES: List[Dict[str, Any]] = [
    {"question": "我的订单 #12345 还没到，已经超时了"},
    {"question": "应用登录一直报错 401"},
    {"question": "为什么这个月多扣了 50 块钱？"},
    {"question": "帮我把收货地址改成北京市朝阳区"},
    {"turns": ["你好，我想退款", "订单号是 #12345", "退款多久能到账？"]},
]

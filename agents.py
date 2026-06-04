from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict
from typing import Any

from .llm_client import LLMClient
from .models import InputDocument, QualityIssue, SceneResult, SummaryResult
from .text_utils import clean_transcript, keyword_hits, semantic_chunks, split_sentences


MEETING_KEYWORDS = [
    "会议",
    "议题",
    "决策",
    "决定",
    "负责人",
    "截止",
    "任务",
    "项目",
    "风险",
    "推进",
    "下周",
    "复盘",
]

CLASSROOM_KEYWORDS = [
    "老师",
    "同学",
    "课程",
    "知识点",
    "定义",
    "定理",
    "公式",
    "例题",
    "考试",
    "重点",
    "难点",
    "复习",
]

ACTION_PATTERNS = [
    r"(?P<owner>[\u4e00-\u9fa5A-Za-z]{2,8}?)(?:负责|牵头|跟进|完成)(?:完成|处理|推进)?(?P<task>[^。！？\n]{4,80})",
    r"(?P<task>[^。！？\n]{4,80})(?:由|让|请)(?P<owner>[\u4e00-\u9fa5A-Za-z]{2,8})(?:负责|跟进|完成)",
]

TIME_PATTERN = r"(\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}月\d{1,2}日|本周|下周|月底|周[一二三四五六日天]|明天|后天|下次会议前)"


class TextCleaningAgent:
    name = "文本清洗智能体"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm

    def run(self, text: str) -> str:
        cleaned = clean_transcript(text)
        if not self.llm:
            return cleaned
        data = self.llm.generate_json(
            system_prompt="你是文本清洗智能体，只返回 JSON。",
            user_prompt=(
                "请清洗下面的语音转写文本，去除低价值口语词、重复表达和明显闲聊，"
                "保留业务或知识信息。返回格式：{\"cleaned_text\":\"...\"}。\n\n"
                f"{cleaned}"
            ),
        )
        return str(data.get("cleaned_text") or cleaned) if isinstance(data, dict) else cleaned


class SceneRecognitionAgent:
    name = "场景识别智能体"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm

    def run(self, text: str, forced_mode: str = "auto") -> SceneResult:
        if forced_mode in {"meeting", "classroom", "mixed", "general"}:
            return SceneResult(forced_mode, 1.0, ["用户手动指定场景"], self._template(forced_mode))

        if self.llm:
            result = self._run_llm(text)
            if result:
                return result

        return self._run_rules(text)

    def _run_llm(self, text: str) -> SceneResult | None:
        data = self.llm.generate_json(
            system_prompt="你是场景识别智能体。只返回合法 JSON，不要输出解释。",
            user_prompt=(
                "判断文本属于 meeting、classroom、mixed、general 之一。"
                "返回格式：{\"scene_type\":\"meeting\",\"confidence\":0.9,"
                "\"reasons\":[\"...\"],\"recommended_template\":\"会议纪要模板\"}。\n\n"
                f"{text[:6000]}"
            ),
        )
        if not isinstance(data, dict):
            return None
        scene_type = data.get("scene_type")
        if scene_type not in {"meeting", "classroom", "mixed", "general"}:
            return None
        return SceneResult(
            scene_type=scene_type,
            confidence=float(data.get("confidence") or 0.75),
            reasons=list(data.get("reasons") or ["大模型识别"]),
            recommended_template=str(data.get("recommended_template") or self._template(scene_type)),
        )

    def _run_rules(self, text: str) -> SceneResult:
        meeting_hits = keyword_hits(text, MEETING_KEYWORDS)
        classroom_hits = keyword_hits(text, CLASSROOM_KEYWORDS)
        meeting_score = len(meeting_hits)
        classroom_score = len(classroom_hits)

        if meeting_score >= 2 and classroom_score >= 2:
            scene_type = "mixed"
        elif meeting_score > classroom_score:
            scene_type = "meeting"
        elif classroom_score > meeting_score:
            scene_type = "classroom"
        else:
            scene_type = "general"

        total = max(meeting_score + classroom_score, 1)
        confidence = min(0.95, 0.45 + abs(meeting_score - classroom_score) / total * 0.5)
        reasons = []
        if meeting_hits:
            reasons.append("会议特征：" + "、".join(meeting_hits[:8]))
        if classroom_hits:
            reasons.append("课堂特征：" + "、".join(classroom_hits[:8]))
        if not reasons:
            reasons.append("未检测到强场景关键词，按普通讨论处理")
        return SceneResult(scene_type, round(confidence, 2), reasons, self._template(scene_type))

    @staticmethod
    def _template(scene_type: str) -> str:
        return {
            "meeting": "会议纪要模板",
            "classroom": "课堂知识总结模板",
            "mixed": "混合分段模板",
            "general": "通用结构化摘要模板",
        }[scene_type]


class LongTextPlannerAgent:
    name = "长文本规划智能体"

    def run(self, text: str) -> list[str]:
        return semantic_chunks(text)


class MeetingSummaryAgent:
    name = "会议总结智能体"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm

    def run(self, text: str, chunks: list[str]) -> dict[str, Any]:
        if self.llm:
            result = self._run_llm(text, chunks)
            if result:
                return result
        return self._run_rules(text)

    def _run_llm(self, text: str, chunks: list[str]) -> dict[str, Any] | None:
        data = self.llm.generate_json(
            system_prompt="你是会议总结智能体。请严格输出 JSON，不要输出 Markdown。",
            user_prompt=(
                "请根据会议转写文本生成结构化会议纪要。必须包含这些顶层字段："
                "会议基本信息、会议背景、核心议题、讨论内容、决策结果、工作部署、待办事项、风险与问题、下一步计划。"
                "其中待办事项为数组，每项包含：任务、责任人、截止时间、优先级、备注。"
                "无法确认的信息写“待人工确认”。\n\n"
                f"分块数量：{len(chunks)}\n\n{text[:12000]}"
            ),
        )
        return data if isinstance(data, dict) and "待办事项" in data else None

    def _run_rules(self, text: str) -> dict[str, Any]:
        sentences = split_sentences(text)
        decisions = [s for s in sentences if any(k in s for k in ["决定", "通过", "确认", "同意", "结论"])]
        risks = [s for s in sentences if any(k in s for k in ["风险", "问题", "延期", "阻塞", "困难", "待确认"])]
        topics = self._topics(sentences, MEETING_KEYWORDS)
        tasks = self._tasks(sentences)
        return {
            "会议基本信息": {
                "会议主题": topics[0] if topics else "待人工确认",
                "会议时间": self._first_time(text) or "待人工确认",
                "会议地点": "待人工补充",
                "参会人员": self._speakers(text),
                "记录人": "待人工补充",
            },
            "会议背景": self._brief(sentences, ["背景", "目标", "原因", "目的"]),
            "核心议题": topics or ["待人工确认核心议题"],
            "讨论内容": self._discussion(sentences),
            "决策结果": decisions[:8] or ["未识别到明确决策，请人工复核"],
            "工作部署": [task["任务"] for task in tasks] or ["未识别到明确工作部署"],
            "待办事项": tasks,
            "风险与问题": risks[:8] or ["未识别到明确风险"],
            "下一步计划": [s for s in sentences if any(k in s for k in ["下一步", "后续", "下周", "下次"])][:8]
            or ["待补充后续计划"],
        }

    def _topics(self, sentences: list[str], stopwords: list[str]) -> list[str]:
        candidates = [s for s in sentences if any(k in s for k in ["议题", "讨论", "项目", "方案", "需求", "目标"])]
        if candidates:
            return [self._shorten(s) for s in candidates[:6]]
        words = re.findall(r"[\u4e00-\u9fa5A-Za-z]{2,8}", " ".join(sentences))
        common = [w for w, _ in Counter(words).most_common(8) if w not in stopwords]
        return common[:5]

    def _tasks(self, sentences: list[str]) -> list[dict[str, str]]:
        tasks: list[dict[str, str]] = []
        for sentence in sentences:
            if not any(k in sentence for k in ["负责", "跟进", "完成", "截止", "推进", "安排"]):
                continue
            owner = "待确认"
            task = self._shorten(sentence, 90)
            for pattern in ACTION_PATTERNS:
                match = re.search(pattern, sentence)
                if match:
                    owner = match.groupdict().get("owner") or owner
                    task = self._shorten(match.groupdict().get("task") or task, 90)
                    break
            time_match = re.search(TIME_PATTERN, sentence)
            tasks.append(
                {
                    "任务": task,
                    "责任人": owner,
                    "截止时间": time_match.group(1) if time_match else "待确认",
                    "优先级": "高" if any(k in sentence for k in ["紧急", "必须", "优先"]) else "中",
                    "备注": sentence,
                }
            )
        return tasks[:12]

    @staticmethod
    def _speakers(text: str) -> list[str]:
        speakers = re.findall(r"(^|\n)([\u4e00-\u9fa5A-Za-z]{2,8})[:：]", text)
        return sorted({name for _, name in speakers}) or ["待人工确认"]

    @staticmethod
    def _first_time(text: str) -> str | None:
        match = re.search(TIME_PATTERN, text)
        return match.group(1) if match else None

    @staticmethod
    def _brief(sentences: list[str], markers: list[str]) -> str:
        for sentence in sentences:
            if any(marker in sentence for marker in markers):
                return MeetingSummaryAgent._shorten(sentence, 140)
        return "根据转写内容自动生成，建议人工补充会议背景。"

    @staticmethod
    def _discussion(sentences: list[str]) -> list[str]:
        useful = [s for s in sentences if len(s) > 10 and not any(k in s for k in ["嗯", "啊"])]
        return [MeetingSummaryAgent._shorten(s, 120) for s in useful[:10]]

    @staticmethod
    def _shorten(text: str, limit: int = 50) -> str:
        return text if len(text) <= limit else text[: limit - 1] + "..."


class ClassroomSummaryAgent:
    name = "课堂知识提炼智能体"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm

    def run(self, text: str, chunks: list[str]) -> dict[str, Any]:
        if self.llm:
            result = self._run_llm(text, chunks)
            if result:
                return result
        return self._run_rules(text)

    def _run_llm(self, text: str, chunks: list[str]) -> dict[str, Any] | None:
        data = self.llm.generate_json(
            system_prompt="你是课堂知识提炼智能体。请严格输出 JSON，不要输出 Markdown。",
            user_prompt=(
                "请根据课堂转写文本生成结构化课堂笔记。必须包含这些顶层字段："
                "课程基本信息、本节课概要、核心知识点、重点难点、公式定理、解题思路、考点精华、易错点提醒、知识框架、复习建议。"
                "无法确认的信息写“待人工确认”。\n\n"
                f"分块数量：{len(chunks)}\n\n{text[:12000]}"
            ),
        )
        return data if isinstance(data, dict) and "核心知识点" in data else None

    def _run_rules(self, text: str) -> dict[str, Any]:
        sentences = split_sentences(text)
        knowledge = [s for s in sentences if any(k in s for k in ["定义", "概念", "原理", "方法", "结论", "知识点"])]
        formulas = [s for s in sentences if any(k in s for k in ["公式", "定理", "等于", "=", "推导", "函数"])]
        important = [s for s in sentences if any(k in s for k in ["重点", "难点", "注意", "容易错", "考点", "考试"])]
        examples = [s for s in sentences if any(k in s for k in ["例题", "题型", "解题", "步骤", "证明"])]
        title = self._course_title(text)
        return {
            "课程基本信息": {
                "课程名称": title,
                "章节主题": self._chapter(text),
                "本节课主题": title,
            },
            "本节课概要": self._overview(sentences),
            "核心知识点": knowledge[:12] or self._fallback_points(sentences),
            "重点难点": important[:10] or ["未识别到明确重点难点，请人工复核"],
            "公式定理": formulas[:10] or ["未识别到明确公式定理"],
            "解题思路": examples[:10] or ["未识别到明确解题步骤"],
            "考点精华": [s for s in important if "考" in s][:8] or important[:5] or ["待结合考试要求补充"],
            "易错点提醒": [s for s in sentences if any(k in s for k in ["错误", "误区", "混淆", "注意"])][:8]
            or ["暂无明确易错点"],
            "知识框架": self._framework(knowledge or sentences),
            "复习建议": [
                "先按知识框架回顾概念，再针对重点难点做例题。",
                "对公式、定理和适用条件进行单独整理。",
                "复盘易错点，形成错题或问题清单。",
            ],
        }

    @staticmethod
    def _course_title(text: str) -> str:
        match = re.search(r"(?:课程|今天讲|本节课|章节)[:：是为]?\s*([^。！？\n]{2,30})", text)
        return match.group(1).strip() if match else "待人工确认"

    @staticmethod
    def _chapter(text: str) -> str:
        match = re.search(r"第[一二三四五六七八九十\d]+[章节课][^。！？\n]{0,30}", text)
        return match.group(0) if match else "待人工确认"

    @staticmethod
    def _overview(sentences: list[str]) -> str:
        return "；".join(sentences[:3]) if sentences else "暂无内容"

    @staticmethod
    def _fallback_points(sentences: list[str]) -> list[str]:
        return [s for s in sentences if len(s) > 8][:8] or ["未识别到核心知识点"]

    @staticmethod
    def _framework(items: list[str]) -> list[str]:
        return [f"{idx}. {MeetingSummaryAgent._shorten(item, 70)}" for idx, item in enumerate(items[:8], start=1)]


class QualityReviewAgent:
    name = "质量校验智能体"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm

    def run(self, scene: SceneResult, content: dict[str, Any]) -> list[QualityIssue]:
        issues = self._run_rules(scene, content)
        if not self.llm:
            return issues
        data = self.llm.generate_json(
            system_prompt="你是质量校验智能体。只返回 JSON。",
            user_prompt=(
                "请检查结构化总结是否存在遗漏、矛盾、重复、不确定信息。"
                "返回格式：{\"issues\":[{\"level\":\"warning\",\"message\":\"...\"}]}。\n\n"
                f"场景：{scene.scene_type}\n结果：{content}"
            ),
        )
        if not isinstance(data, dict):
            return issues
        llm_issues = []
        for item in data.get("issues") or []:
            if isinstance(item, dict):
                level = item.get("level") if item.get("level") in {"info", "warning", "error"} else "info"
                message = str(item.get("message") or "").strip()
                if message:
                    llm_issues.append(QualityIssue(level, message))
        return llm_issues or issues

    @staticmethod
    def _run_rules(scene: SceneResult, content: dict[str, Any]) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        if scene.confidence < 0.65:
            issues.append(QualityIssue("warning", "场景识别置信度较低，建议人工确认处理模板。"))
        serialized = str(content)
        if "待人工确认" in serialized or "待确认" in serialized:
            issues.append(QualityIssue("warning", "结果中存在待确认信息，请人工复核责任人、时间或主题。"))
        if "未识别到明确" in serialized:
            issues.append(QualityIssue("info", "部分字段未从文本中明确识别，系统已保留复核提示。"))
        return issues


class AgentOrchestrator:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm if llm is not None else LLMClient.from_env()
        self.cleaner = TextCleaningAgent(self.llm)
        self.scene_recognizer = SceneRecognitionAgent(self.llm)
        self.planner = LongTextPlannerAgent()
        self.meeting_agent = MeetingSummaryAgent(self.llm)
        self.classroom_agent = ClassroomSummaryAgent(self.llm)
        self.quality_agent = QualityReviewAgent(self.llm)

    def run(self, document: InputDocument) -> SummaryResult:
        cleaned = self.cleaner.run(document.raw_text)
        scene = self.scene_recognizer.run(cleaned, document.mode)
        chunks = self.planner.run(cleaned)
        if scene.scene_type == "classroom":
            content = self.classroom_agent.run(cleaned, chunks)
        elif scene.scene_type == "mixed":
            content = {
                "会议部分": self.meeting_agent.run(cleaned, chunks),
                "课堂部分": self.classroom_agent.run(cleaned, chunks),
            }
        elif scene.scene_type == "meeting":
            content = self.meeting_agent.run(cleaned, chunks)
        else:
            content = {
                "摘要": "；".join(split_sentences(cleaned)[:5]) or "暂无内容",
                "要点": split_sentences(cleaned)[:10],
                "后续建议": ["可手动指定会议或课堂场景以获得更精确模板。"],
            }
        issues = self.quality_agent.run(scene, content)
        return SummaryResult(
            title=document.title or "结构化总结",
            scene=scene,
            cleaned_text=cleaned,
            chunks=chunks,
            content=content,
            quality_issues=issues,
        )

    @staticmethod
    def to_dict(result: SummaryResult) -> dict[str, Any]:
        return asdict(result)

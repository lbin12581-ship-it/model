from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


SceneType = Literal["meeting", "classroom", "mixed", "general"]


@dataclass
class InputDocument:
    title: str
    raw_text: str
    mode: SceneType | Literal["auto"] = "auto"
    source_name: str = "manual-input"


@dataclass
class SceneResult:
    scene_type: SceneType
    confidence: float
    reasons: list[str]
    recommended_template: str


@dataclass
class QualityIssue:
    level: Literal["info", "warning", "error"]
    message: str


@dataclass
class SummaryResult:
    title: str
    scene: SceneResult
    cleaned_text: str
    chunks: list[str]
    content: dict[str, Any]
    quality_issues: list[QualityIssue] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agents import AgentOrchestrator
from .models import SummaryResult


ARCHIVE_DIR = Path("data/archive")


def archive_result(result: SummaryResult) -> Path:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    safe_title = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in result.title)[:60]
    path = ARCHIVE_DIR / f"{result.created_at.replace(':', '-')}_{safe_title}.json"
    path.write_text(json.dumps(AgentOrchestrator.to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def list_archives() -> list[Path]:
    if not ARCHIVE_DIR.exists():
        return []
    return sorted(ARCHIVE_DIR.glob("*.json"), reverse=True)


def search_archives(keyword: str) -> list[dict[str, Any]]:
    matches = []
    if not keyword.strip():
        return matches
    for path in list_archives():
        text = path.read_text(encoding="utf-8")
        if keyword in text:
            data = json.loads(text)
            matches.append(
                {
                    "path": str(path),
                    "title": data.get("title", path.stem),
                    "created_at": data.get("created_at", ""),
                    "scene": data.get("scene", {}).get("scene_type", ""),
                }
            )
    return matches


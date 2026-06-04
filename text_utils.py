from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


FILLER_WORDS = [
    "嗯",
    "啊",
    "呃",
    "额",
    "然后呢",
    "然后",
    "这个",
    "那个",
    "就是说",
    "其实吧",
    "大家注意一下",
]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200b\ufeff]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;])\s*|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def dedupe_sentences(sentences: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for sentence in sentences:
        key = re.sub(r"\W+", "", sentence.lower())
        if len(key) < 4:
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(sentence)
    return result


def clean_transcript(text: str) -> str:
    text = normalize_text(text)
    for word in FILLER_WORDS:
        text = text.replace(word, "")
    text = re.sub(r"([，。！？；、])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    sentences = dedupe_sentences(split_sentences(text))
    return "\n".join(sentences)


def semantic_chunks(text: str, max_chars: int = 1800) -> list[str]:
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for sentence in sentences:
        if current and size + len(sentence) > max_chars:
            chunks.append("\n".join(current))
            current = []
            size = 0
        current.append(sentence)
        size += len(sentence)
    if current:
        chunks.append("\n".join(current))
    return chunks or [text]


def read_txt(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def read_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    paragraphs: list[str] = []
    for paragraph in root.iter():
        if paragraph.tag.endswith("}p"):
            text = "".join(node.text or "" for node in paragraph.iter() if node.tag.endswith("}t"))
            if text.strip():
                paragraphs.append(text.strip())
    return "\n".join(paragraphs)


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    return [word for word in keywords if word in text]


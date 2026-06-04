from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm_client import load_env_file


@dataclass
class ASRConfig:
    api_key: str
    model: str = "paraformer-8k-v1"
    base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    poll_interval: float = 3.0
    timeout: int = 600
    disfluency_removal_enabled: bool = True
    diarization_enabled: bool = False

    @classmethod
    def from_env(cls) -> "ASRConfig | None":
        load_env_file()
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ASR_API_KEY")
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            model=os.getenv("ASR_MODEL", "paraformer-8k-v1"),
            base_url=os.getenv("ASR_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"),
            poll_interval=float(os.getenv("ASR_POLL_INTERVAL", "3")),
            timeout=int(os.getenv("ASR_TIMEOUT", "600")),
            disfluency_removal_enabled=_env_bool("ASR_DISFLUENCY_REMOVAL", True),
            diarization_enabled=_env_bool("ASR_DIARIZATION", False),
        )


class ASRClient:
    """Aliyun DashScope Paraformer recorded speech recognition client."""

    def __init__(self, config: ASRConfig) -> None:
        self.config = config

    @classmethod
    def from_env(cls) -> "ASRClient | None":
        config = ASRConfig.from_env()
        return cls(config) if config else None

    def transcribe_url(self, file_url: str) -> str:
        task_id = self.submit_task([file_url], resolve_oss=file_url.startswith("oss://"))
        result = self.wait_for_complete(task_id)
        urls = self._successful_transcription_urls(result)
        if not urls:
            raise RuntimeError(f"语音识别任务完成，但没有成功的转写结果：{result}")
        transcripts = [self.fetch_transcription_text(url) for url in urls]
        return "\n".join(text for text in transcripts if text.strip()).strip()

    def transcribe_file_bytes(self, filename: str, data: bytes) -> str:
        oss_url = self.upload_bytes(filename, data)
        return self.transcribe_url(oss_url)

    def upload_bytes(self, filename: str, data: bytes) -> str:
        policy = self.get_upload_policy()
        safe_name = Path(filename).name or f"audio-{uuid.uuid4().hex}.wav"
        key = f"{policy['upload_dir'].rstrip('/')}/{safe_name}"
        fields = {
            "OSSAccessKeyId": str(policy["oss_access_key_id"]),
            "Signature": str(policy["signature"]),
            "policy": str(policy["policy"]),
            "x-oss-object-acl": str(policy["x_oss_object_acl"]),
            "x-oss-forbid-overwrite": str(policy["x_oss_forbid_overwrite"]),
            "key": key,
            "success_action_status": "200",
        }
        content_type = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
        body, boundary = _build_multipart_form(fields, "file", safe_name, data, content_type)
        request = urllib.request.Request(
            str(policy["upload_host"]),
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=max(self.config.timeout, 60)) as response:
                if response.status != 200:
                    raise RuntimeError(f"上传本地音频失败，HTTP {response.status}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"上传本地音频失败，HTTP {exc.code}: {detail}") from exc
        return f"oss://{key}"

    def get_upload_policy(self) -> dict[str, Any]:
        query = urllib.parse.urlencode({"action": "getPolicy", "model": self.config.model})
        url = self.config.base_url.rstrip("/") + f"/uploads?{query}"
        request = urllib.request.Request(
            url,
            method="GET",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"获取本地音频上传凭证失败，HTTP {exc.code}: {detail}") from exc
        data = payload.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"获取本地音频上传凭证失败：{payload}")
        return data

    def submit_task(self, file_urls: list[str], resolve_oss: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": {"file_urls": file_urls},
            "parameters": {
                "channel_id": [0],
                "disfluency_removal_enabled": self.config.disfluency_removal_enabled,
                "diarization_enabled": self.config.diarization_enabled,
            },
        }
        data = self._post_json(
            self.config.base_url.rstrip("/") + "/services/audio/asr/transcription",
            payload,
            async_header=True,
            resolve_oss=resolve_oss,
        )
        task_id = data.get("output", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"提交语音识别任务失败：{data}")
        return str(task_id)

    def wait_for_complete(self, task_id: str) -> dict[str, Any]:
        deadline = time.time() + self.config.timeout
        while time.time() < deadline:
            data = self._post_json(self.config.base_url.rstrip("/") + f"/tasks/{task_id}", None)
            output = data.get("output", data)
            status = output.get("task_status")
            if status == "SUCCEEDED":
                return output
            if status in {"FAILED", "CANCELED", "UNKNOWN"}:
                raise RuntimeError(f"语音识别任务失败：{data}")
            time.sleep(self.config.poll_interval)
        raise TimeoutError(f"语音识别超时，task_id={task_id}")

    def fetch_transcription_text(self, transcription_url: str) -> str:
        with urllib.request.urlopen(transcription_url, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        return self._extract_text(data)

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any] | None,
        async_header: bool = False,
        resolve_oss: bool = False,
    ) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if async_header:
            headers["X-DashScope-Async"] = "enable"
        if resolve_oss:
            headers["X-DashScope-OssResourceResolve"] = "enable"
        request = urllib.request.Request(url, data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"阿里云语音识别 HTTP {exc.code}: {detail}") from exc

    @staticmethod
    def _successful_transcription_urls(output: dict[str, Any]) -> list[str]:
        urls = []
        for item in output.get("results", []):
            if item.get("subtask_status") == "SUCCEEDED" and item.get("transcription_url"):
                urls.append(str(item["transcription_url"]))
        return urls

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        parts: list[str] = []
        for transcript in data.get("transcripts", []):
            sentences = transcript.get("sentences") or []
            if sentences:
                for sentence in sentences:
                    text = str(sentence.get("text") or "").strip()
                    if not text:
                        continue
                    speaker = sentence.get("speaker_id")
                    parts.append(f"说话人{speaker}：{text}" if speaker is not None else text)
            else:
                text = str(transcript.get("text") or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts)


def asr_status_text() -> str:
    config = ASRConfig.from_env()
    if not config:
        return "未配置阿里云语音识别 API。"
    return f"已配置阿里云语音识别：{config.model}"


def _build_multipart_form(
    fields: dict[str, str],
    file_field: str,
    filename: str,
    file_data: bytes,
    content_type: str,
) -> tuple[bytes, str]:
    boundary = "----CodexFormBoundary" + uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            file_data,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks), boundary


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

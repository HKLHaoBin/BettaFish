"""
Keyword-level labeling powered by the existing Keyword Optimizer configuration.

This module reuses the KEYWORD_OPTIMIZER_* settings (the lightweight SQL /
keyword optimization model the user mentioned) to request coarse semantic
labels for short text snippets. The output feeds StructuredContextBuilder so
multi-agent workflows get richer event metadata without invoking heavyweight
LLMs.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI

from config import settings
from utils.retry_helper import SEARCH_API_RETRY_CONFIG, with_graceful_retry

LABEL_SCHEMA = [
    "animal",
    "game_company",
    "game_title",
    "content_creator",
    "monetization",
    "platform",
    "organization",
    "geography",
    "legal",
    "sentiment",
    "numeric_expression",
    "temporal_reference",
    "technology",
    "financial_metric",
    "esports",
    "hardware",
    "release_milestone",
    "character",
    "issue_or_bug",
    "community",
    "other",
]


class KeywordLabelingClient:
    """Calls the Keyword Optimizer model to classify salient terms."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or settings.KEYWORD_OPTIMIZER_API_KEY
        if not self.api_key:
            raise ValueError("未配置 KEYWORD_OPTIMIZER_API_KEY，无法启用关键词标签器")
        self.base_url = base_url or settings.KEYWORD_OPTIMIZER_BASE_URL
        self.model_name = model_name or settings.KEYWORD_OPTIMIZER_MODEL_NAME
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def label_terms(
        self, text: str, *, topic: str = "", max_terms: int = 8
    ) -> List[Dict[str, Any]]:
        if not text:
            return []
        snippet = text[:1500]
        payload = self._call_model(snippet, topic, max_terms)
        if not payload.get("success"):
            return []
        return self._parse_annotations(payload.get("content"))

    @with_graceful_retry(
        SEARCH_API_RETRY_CONFIG,
        default_return={"success": False, "error": "关键词标签器不可用"},
    )
    def _call_model(
        self, snippet: str, topic: str, max_terms: int
    ) -> Dict[str, Any]:
        system_prompt = self._build_system_prompt(max_terms)
        user_prompt = self._build_user_prompt(snippet, topic)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        if response.choices:
            return {"success": True, "content": response.choices[0].message.content}
        return {"success": False, "error": "无有效响应"}

    def _build_system_prompt(self, max_terms: int) -> str:
        labels = ", ".join(LABEL_SCHEMA)
        return (
            "你是一个关键词分类器，只需找到文本中最重要的名词或短语。"
            "请只输出少量行级结果，每行格式为：`术语 | 标签1, 标签2, 标签3`。"
            f"标签只能来自：{labels}。没有合适标签时可使用 other。\n"
            "示例：\n"
            "原神 | game_title, monetization\n"
            "米哈游 | organization, content_creator\n"
            "如果你更擅长 JSON，也可以返回如下结构，程序依然可以解析：\n"
            "{\n"
            '  "annotations": [\n'
            '    {\n'
            '      "term": "原神",\n'
            '      "labels": [\n'
            '        {"name": "game_title", "confidence": 0.92},\n'
            '        {"name": "monetization", "confidence": 0.08}\n'
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
            f"最多 {max_terms} 个术语，每行不要添加解释或额外文本。"
        )

    @staticmethod
    def _build_user_prompt(snippet: str, topic: str) -> str:
        prompt = f"待分析文本：```\n{snippet}\n```"
        if topic:
            prompt += f"\n\n主题：{topic}"
        prompt += "\n请使用 `术语 | 标签1, 标签2` 的行级格式（或前文提供的 JSON 格式）输出，不要添加解释。"
        return prompt

    def _parse_annotations(self, content: Optional[str]) -> List[Dict[str, Any]]:
        if not content:
            return []

        json_annotations = self._parse_json_block(content)
        if json_annotations:
            return json_annotations

        fallback_annotations = self._parse_simple_lines(content)
        if fallback_annotations:
            return fallback_annotations

        logger.warning("关键词标签器返回的内容无法解析为JSON，也无法匹配行级格式")
        return []

    def _parse_json_block(self, content: str) -> List[Dict[str, Any]]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return []
        annotations = payload.get("annotations") or payload.get("keywords") or []
        return self._normalize_annotations(annotations)

    def _parse_simple_lines(self, content: str) -> List[Dict[str, Any]]:
        annotations: List[Dict[str, Any]] = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = line.lstrip("-•* ")
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            if not line:
                continue
            term, labels_chunk = self._split_term_and_labels(line)
            if not term:
                continue
            labels = self._parse_label_tokens(labels_chunk)
            annotations.append({"term": term, "labels": labels})
        return self._normalize_annotations(annotations)

    @staticmethod
    def _split_term_and_labels(line: str) -> tuple[str, str]:
        separators = ["|", "：", ":", "=>", "->", "—", "——", " - "]
        for sep in separators:
            if sep in line:
                left, right = line.split(sep, 1)
                return left.strip(), right.strip()
        return line.strip(), ""

    @staticmethod
    def _parse_label_tokens(chunk: str) -> List[Dict[str, Any]]:
        if not chunk:
            return [{"label": "other", "score": 0.5}]
        tokens = re.split(r"[，,、/]+|\s{2,}", chunk)
        labels = []
        for token in tokens:
            text = token.strip()
            if not text:
                continue
            match = re.match(r"(?P<label>[^(]+)(?:\((?P<score>[\d.]+)\))?", text)
            if not match:
                continue
            label_name = match.group("label").strip()
            if not label_name:
                continue
            raw_score = match.group("score")
            try:
                score = float(raw_score) if raw_score else 0.6
            except ValueError:
                score = 0.6
            labels.append({"label": label_name, "score": max(0.0, min(1.0, score))})
        return labels or [{"label": "other", "score": 0.5}]

    @staticmethod
    def _normalize_annotations(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        for entry in annotations:
            term = entry.get("term") or entry.get("keyword")
            if not term:
                continue
            labels = entry.get("labels") or entry.get("categories") or []
            normalized = []
            for label in labels:
                if isinstance(label, str):
                    normalized.append({"label": label, "score": 0.6})
                    continue
                name = label.get("name") or label.get("label")
                if not name:
                    continue
                score = label.get("confidence") or label.get("score") or 0.5
                normalized.append(
                    {"label": name, "score": float(max(0.0, min(1.0, score)))}
                )
            if not normalized:
                normalized.append({"label": "other", "score": 0.5})
            cleaned.append(
                {
                    "term": term.strip(),
                    "labels": normalized,
                }
            )
        return cleaned


@lru_cache(maxsize=1)
def get_keyword_labeler() -> Optional[KeywordLabelingClient]:
    try:
        return KeywordLabelingClient()
    except Exception as exc:  # pragma: no cover - environment specific
        logger.warning(f"关键词标签器初始化失败: {exc}")
        return None

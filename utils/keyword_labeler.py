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
            "你是一个关键词分类模型，负责读取短文本并找出其中最核心的名词或短语。"
            "请只关注有助于理解舆情的 term（例如人物、组织、游戏、动物、金额）。"
            "无需了解这些标签将被用在什么场景，只需完成分类任务。\n"
            f"对每个 term 尽量给出 3-5 个标签，标签必须来自以下集合：{labels}。\n"
            "返回 JSON，格式如下：\n"
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
            f"总 term 数量上限 {max_terms}，置信度范围 0-1，保留两位小数。"
        )

    @staticmethod
    def _build_user_prompt(snippet: str, topic: str) -> str:
        prompt = f"待分析文本：```\n{snippet}\n```"
        if topic:
            prompt += f"\n\n主题：{topic}"
        prompt += "\n请只输出 JSON，不要添加额外解释。"
        return prompt

    def _parse_annotations(self, content: Optional[str]) -> List[Dict[str, Any]]:
        if not content:
            return []
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("关键词标签器返回的内容无法解析为JSON")
            return []
        annotations = payload.get("annotations") or payload.get("keywords") or []
        cleaned: List[Dict[str, Any]] = []
        for entry in annotations:
            term = entry.get("term") or entry.get("keyword")
            if not term:
                continue
            labels = entry.get("labels") or entry.get("categories") or []
            normalized = []
            for label in labels:
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

"""
Structured context utilities for building timeline-aware inputs.

This module adds three core capabilities that multiple agents can share:

1. Token-level semantic labeling powered by lightweight prototypes so we can
   separate concepts such as `animal`, `game_title`, or `numeric_expression`
   without leaking the downstream query intent.
2. Event graph construction that groups raw search/database rows into
   chronological events with weights derived from label confidence.
3. Prompt decoration helpers that append the structured context back to the
   agents so LLMs receive explicit temporal + relational cues.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger

# --- Lightweight semantic prototypes ---------------------------------------------------------

DEFAULT_PROTOTYPES: Dict[str, List[str]] = {
    "animal": [
        "animal",
        "wildlife",
        "bear",
        "penguin",
        "polar",
        "猫",
        "狗",
        "动物",
        "北极熊",
        "企鹅",
        "野生",
    ],
    "game_company": [
        "studio",
        "developer",
        "publisher",
        "company",
        "米哈游",
        "霍格沃茨",
        "hoYoverse",
        "游戏公司",
        "腾讯",
        "网易",
    ],
    "game_title": [
        "game",
        "title",
        "原神",
        "崩坏",
        "绝区零",
        "star rail",
        "honkai",
        "Genshin",
        "手游",
        "MMO",
        "RPG",
        "剧情",
        "动作",
        "冒险",
    ],
    "content_creator": [
        "主播",
        "直播",
        "up主",
        "creator",
        "streamer",
        "平台",
        "b站",
        "douyin",
        "tiktok",
        "快手",
        "youtube",
    ],
    "monetization": [
        "充值",
        "氪金",
        "648",
        "顶级礼包",
        "抽卡",
        "付费",
        "purchase",
        "in-app",
        "销售",
        "售价",
        "金额",
    ],
    "geography": [
        "city",
        "country",
        "地区",
        "省份",
        "海外",
        "国内",
        "north",
        "south",
        "东部",
        "西部",
        "global",
        "国际",
    ],
    "organization": [
        "organization",
        "协会",
        "联盟",
        "group",
        "team",
        "committee",
        "机构",
        "官方",
    ],
    "platform": [
        "platform",
        "社交",
        "社区",
        "论坛",
        "微博",
        "知乎",
        "reddit",
        "discord",
        "快手",
        "抖音",
    ],
    "legal": [
        "law",
        "regulation",
        "法规",
        "法律",
        "政策",
        "compliance",
        "诉讼",
        "仲裁",
    ],
    "sentiment": [
        "支持",
        "反对",
        "正面",
        "负面",
        "controversy",
        "舆论",
        "声量",
        "热度",
    ],
    "technology": [
        "技术",
        "科技",
        "engine",
        "unity",
        "unreal",
        "算法",
        "云",
        "server",
        "AI",
        "模型",
        "渲染",
        "工具链",
    ],
    "financial_metric": [
        "营收",
        "收入",
        "profit",
        "利润",
        "亏损",
        "gmv",
        "估值",
        "现金流",
        "dau",
        "mau",
    ],
    "esports": [
        "战队",
        "联赛",
        "赛事",
        "冠军",
        "选手",
        "杯赛",
        "锦标赛",
        "赛季",
    ],
    "hardware": [
        "主机",
        "硬件",
        "显卡",
        "设备",
        "手柄",
        "耳机",
        "cpu",
        "gpu",
        "终端",
    ],
    "release_milestone": [
        "上线",
        "发布",
        "开测",
        "公测",
        "内测",
        "预约",
        "beta",
        "上市",
        "roadmap",
    ],
    "character": [
        "角色",
        "英雄",
        "npc",
        "boss",
        "角色名",
        "cv",
        "皮肤",
    ],
    "issue_or_bug": [
        "bug",
        "问题",
        "崩溃",
        "报错",
        "故障",
        "卡顿",
        "延迟",
        "漏洞",
    ],
    "community": [
        "玩家",
        "粉丝",
        "社区",
        "社群",
        "论坛",
        "discord",
        "群",
        "讨论",
        "口碑",
    ],
}

STOPWORDS = {
    "and",
    "or",
    "the",
    "of",
    "to",
    "with",
    "for",
    "在",
    "以及",
    "的",
    "了",
    "是",
    "一个",
    "我们",
    "他们",
    "这些",
}


@dataclass
class LabelConfidence:
    label: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "score": round(self.score, 4)}


class SemanticPrototypeLabeler:
    """Assigns coarse-grained semantic labels without requiring task context."""

    token_pattern = re.compile(r"[\u4e00-\u9fa5]{2,8}|[A-Za-z0-9_\-:#\./]+")

    def __init__(
        self,
        prototypes: Optional[Dict[str, List[str]]] = None,
        min_confidence: float = 0.18,
        max_labels: int = 4,
    ):
        self.prototypes = prototypes or DEFAULT_PROTOTYPES
        self.min_confidence = min_confidence
        self.max_labels = max_labels

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        for match in self.token_pattern.findall(text or ""):
            token = match.strip()
            token_lower = token.lower()
            if not token or len(token) <= 1:
                continue
            if token_lower in STOPWORDS:
                continue
            tokens.append(token)
        return tokens

    def label_token(self, token: str, context: str = "") -> List[LabelConfidence]:
        blob = f"{token} {context}".lower()
        raw_scores: Dict[str, float] = {}

        for label, keywords in self.prototypes.items():
            score = self._keyword_score(blob, token.lower(), keywords)
            if score > 0:
                raw_scores[label] = max(raw_scores.get(label, 0.0), score)

        numeric_score = self._numeric_score(token)
        if numeric_score:
            raw_scores["numeric_expression"] = max(
                raw_scores.get("numeric_expression", 0.0), numeric_score
            )

        if self._looks_like_time(token):
            raw_scores["temporal_reference"] = max(
                raw_scores.get("temporal_reference", 0.0), 0.45
            )

        if "-" in token or "/" in token:
            raw_scores["composite_term"] = max(
                raw_scores.get("composite_term", 0.0), 0.3
            )

        if not raw_scores:
            raw_scores["unknown"] = 0.2

        return self._normalize_scores(raw_scores)

    # ---- private helpers ------------------------------------------------------------------

    def _keyword_score(
        self, blob: str, token_lower: str, keywords: Iterable[str]
    ) -> float:
        if not keywords:
            return 0.0
        matches = 0.0
        blob = blob.lower()
        for kw in keywords:
            lowered = kw.lower()
            if lowered in blob:
                matches += 1.0
                continue
            similarity = self._ratio(token_lower, lowered)
            if similarity >= 0.82:
                matches += similarity * 0.8
        if matches == 0:
            return 0.0
        return min(1.0, matches / len(list(keywords)))

    @staticmethod
    def _ratio(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        matches = sum((1 for ch in a if ch in b))
        return matches / max(len(a), len(b))

    @staticmethod
    def _numeric_score(token: str) -> float:
        if any(char.isdigit() for char in token):
            # Boost when包含货币或单位
            if any(symbol in token for symbol in ["$", "¥", "元", "刀", "%"]):
                return 0.75
            return 0.55
        return 0.0

    @staticmethod
    def _looks_like_time(token: str) -> bool:
        return bool(re.match(r"\d{1,2}[:：]\d{2}", token)) or "年" in token or "月" in token

    def _normalize_scores(self, raw_scores: Dict[str, float]) -> List[LabelConfidence]:
        total = sum(raw_scores.values())
        if total == 0:
            total = 1.0
        normalized = []
        for label, value in sorted(
            raw_scores.items(), key=lambda item: item[1], reverse=True
        ):
            score = round(value / total, 4)
            if score < self.min_confidence and label != "unknown":
                continue
            normalized.append(LabelConfidence(label=label, score=score))
            if len(normalized) >= self.max_labels:
                break
        return normalized


# --- Structured context builder --------------------------------------------------------------


class StructuredContextBuilder:
    """Transforms raw search/database rows into a deterministic context payload."""

    def __init__(
        self,
        agent_name: str = "generic",
        labeler: Optional[SemanticPrototypeLabeler] = None,
        max_tokens_per_event: int = 8,
        keyword_labeler: Optional[Any] = None,
        keyword_labeler_max_snippets: int = 3,
    ):
        self.agent_name = agent_name
        self.labeler = labeler or SemanticPrototypeLabeler()
        self.max_tokens_per_event = max_tokens_per_event
        self.keyword_labeler = keyword_labeler
        self.keyword_labeler_max_snippets = keyword_labeler_max_snippets

    # Public API ---------------------------------------------------------------------------

    def empty_context(self) -> Dict[str, Any]:
        return {
            "events": {},
            "relations": [],
            "timeline": [],
            "token_statistics": {},
            "metadata": {
                "agent": self.agent_name,
                "created_at": datetime.utcnow().isoformat(),
                "origin_event_id": None,
                "origin_query": None,
                "origin_labels": [],
                "origin_dominant_labels": [],
            },
        }

    def enrich_with_search_results(
        self,
        search_results: List[Dict[str, Any]],
        *,
        query: str,
        paragraph_title: str,
        stage: str,
        origin_query: Optional[str] = None,
        existing_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        context = existing_context or self.empty_context()
        if origin_query:
            context = self._ensure_origin_node(origin_query, context)

        if not search_results:
            return search_results, context

        relations_index = {
            tuple(sorted((rel["source_event"], rel["target_event"])))
            for rel in context.get("relations", [])
        }

        decorated_results = []
        for idx, result in enumerate(search_results):
            event = self._build_event(
                result,
                query=query,
                paragraph_title=paragraph_title,
                stage=stage,
                ordinal=idx,
            )
            context["events"][event["event_id"]] = event
            self._update_token_statistics(context, event)
            self._update_relations(context, event, relations_index)
            self._update_origin_relation(context, event, relations_index)
            decorated_results.append(self._decorate_result(result, event))

        context["timeline"] = sorted(
            context["events"].values(),
            key=lambda ev: ev.get("timestamp") or "",
        )
        context["metadata"].update(
            {
                "last_query": query,
                "last_stage": stage,
                "paragraph_title": paragraph_title,
                "updated_at": datetime.utcnow().isoformat(),
            }
        )
        return decorated_results, context

    def ensure_origin_context(
        self,
        origin_query: str,
        existing_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Ensure the structured context已经包含原点节点。
        """
        context = existing_context or self.empty_context()
        return self._ensure_origin_node(origin_query, context)

    # Internal helpers --------------------------------------------------------------------

    def _ensure_origin_node(
        self, origin_query: Optional[str], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not origin_query:
            return context
        origin_query = origin_query.strip()
        if not origin_query:
            return context

        metadata = context.setdefault("metadata", {})
        events = context.setdefault("events", {})
        current_origin_id = metadata.get("origin_event_id")

        if current_origin_id and current_origin_id in events:
            if origin_query == metadata.get("origin_query"):
                return context
            events.pop(current_origin_id, None)
            context["relations"] = [
                rel
                for rel in context.get("relations", [])
                if current_origin_id
                not in (rel.get("source_event"), rel.get("target_event"))
            ]

        origin_event = self._build_origin_event(origin_query)
        events[origin_event["event_id"]] = origin_event
        metadata.update(
            {
                "origin_event_id": origin_event["event_id"],
                "origin_query": origin_query,
                "origin_labels": origin_event.get("label_distribution", []),
                "origin_dominant_labels": origin_event.get("dominant_labels", []),
                "origin_updated_at": datetime.utcnow().isoformat(),
            }
        )
        self._update_token_statistics(context, origin_event)
        return context

    def _build_event(
        self,
        result: Dict[str, Any],
        *,
        query: str,
        paragraph_title: str,
        stage: str,
        ordinal: int,
    ) -> Dict[str, Any]:
        source_url = result.get("url") or f"{self.agent_name.lower()}://local/{ordinal}"
        timestamp = self._normalize_timestamp(result.get("published_date"))
        text_blob = " ".join(
            filter(
                None,
                [
                    result.get("title", ""),
                    result.get("raw_content") or result.get("content") or "",
                ],
            )
        )
        tokens = self._token_annotations(text_blob, source_url, timestamp)
        label_distribution = self._aggregate_labels(tokens)
        dominant = [
            lbl["label"]
            for lbl in label_distribution[:3]
            if lbl["score"] >= 0.18
        ]
        event_id = self._event_id(source_url, timestamp, ordinal)
        summary = (result.get("content") or result.get("title") or "")[:400]
        weight = self._estimate_weight(tokens, label_distribution)
        return {
            "event_id": event_id,
            "timestamp": timestamp,
            "source": source_url,
            "weight": weight,
            "intrinsic_weight": weight,
            "summary": summary,
            "query": query,
            "paragraph": paragraph_title,
            "stage": stage,
            "tokens": tokens,
            "label_distribution": label_distribution,
            "top_labels": label_distribution[:3],
            "dominant_labels": dominant,
        }

    def _build_origin_event(self, origin_query: str) -> Dict[str, Any]:
        timestamp = datetime.utcnow().isoformat()
        source_url = f"{self.agent_name.lower()}://origin"
        tokens = self._token_annotations(origin_query, source_url, timestamp)
        label_distribution = self._aggregate_labels(tokens)
        dominant = [
            lbl["label"] for lbl in label_distribution[:3] if lbl["score"] >= 0.15
        ]
        event_id = f"origin-{hashlib.sha1(origin_query.encode('utf-8')).hexdigest()[:12]}"
        weight = max(1.0, self._estimate_weight(tokens, label_distribution))
        return {
            "event_id": event_id,
            "timestamp": timestamp,
            "source": source_url,
            "weight": weight,
            "intrinsic_weight": weight,
            "summary": origin_query[:400],
            "query": origin_query,
            "paragraph": "origin",
            "stage": "origin",
            "tokens": tokens,
            "label_distribution": label_distribution,
            "top_labels": label_distribution[:3],
            "dominant_labels": dominant,
            "origin_similarity": 1.0,
            "origin_shared_labels": dominant,
        }

    def _token_annotations(
        self, text_blob: str, source_url: str, timestamp: str
    ) -> List[Dict[str, Any]]:
        annotations: List[Dict[str, Any]] = []
        seen_terms = set()

        for annotation in self._keyword_labeler_annotations(
            text_blob, source_url, timestamp
        ):
            annotations.append(annotation)
            seen_terms.add(annotation["text"].lower())

        for token in self.labeler.tokenize(text_blob)[: self.max_tokens_per_event]:
            token_lower = token.lower()
            if token_lower in seen_terms:
                continue
            context_snippet = self._context_snippet(text_blob, token)
            labels = [label.to_dict() for label in self.labeler.label_token(token, context_snippet)]
            annotations.append(
                {
                    "text": token,
                    "context": context_snippet,
                    "labels": labels,
                    "source": source_url,
                    "timestamp": timestamp,
                }
            )
            seen_terms.add(token_lower)
        return annotations

    def _keyword_labeler_annotations(
        self, text_blob: str, source_url: str, timestamp: str
    ) -> List[Dict[str, Any]]:
        if not self.keyword_labeler:
            return []
        snippets = self._slice_text_for_labeler(text_blob)
        annotations: List[Dict[str, Any]] = []
        for snippet in snippets:
            try:
                suggestions = self.keyword_labeler.label_terms(
                    snippet,
                    topic=self.agent_name,
                    max_terms=self.max_tokens_per_event,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning(f"关键词标签器调用失败: {exc}")
                break

            for item in suggestions or []:
                term = item.get("term") or item.get("text")
                if not term:
                    continue
                labels = item.get("labels") or []
                normalized = []
                for label in labels:
                    name = label.get("label") or label.get("name")
                    if not name:
                        continue
                    score = float(label.get("score") or label.get("confidence") or 0.5)
                    normalized.append(
                        {"label": name, "score": max(0.0, min(1.0, score))}
                    )
                if not normalized:
                    normalized.append({"label": "other", "score": 0.5})
                annotations.append(
                    {
                        "text": term.strip(),
                        "context": snippet[:200],
                        "labels": normalized,
                        "source": source_url,
                        "timestamp": timestamp,
                    }
                )
        return annotations

    def _slice_text_for_labeler(self, text_blob: str) -> List[str]:
        if not text_blob:
            return []
        import re

        raw_sentences = re.split(r'(?<=[。！？!?])\s+|\n+', text_blob)
        sentences = [
            sentence.strip()
            for sentence in raw_sentences
            if len(sentence.strip()) >= 4
        ]
        selected = sentences[: self.keyword_labeler_max_snippets]
        if not selected:
            selected = [text_blob[:240]]
        return [sentence[:400] for sentence in selected]

    def _aggregate_labels(self, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scores: Dict[str, float] = {}
        for token in tokens:
            for label in token.get("labels", []):
                scores[label["label"]] = scores.get(label["label"], 0.0) + label["score"]
        total = sum(scores.values()) or 1.0
        aggregated = [
            {"label": label, "score": round(value / total, 4)}
            for label, value in scores.items()
        ]
        return sorted(aggregated, key=lambda item: item["score"], reverse=True)

    @staticmethod
    def _estimate_weight(
        tokens: List[Dict[str, Any]], label_distribution: List[Dict[str, Any]]
    ) -> float:
        if not tokens:
            return 0.0
        score = sum(label["score"] for label in label_distribution[:3])
        weight = len(tokens) * score
        return round(weight, 3)

    @staticmethod
    def _context_snippet(text: str, token: str, window: int = 32) -> str:
        lower = text.lower()
        token_lower = token.lower()
        idx = lower.find(token_lower)
        if idx == -1:
            idx = 0
        start = max(0, idx - window)
        end = min(len(text), idx + len(token) + window)
        snippet = text[start:end].strip()
        return re.sub(r"\s+", " ", snippet)

    @staticmethod
    def _normalize_timestamp(value: Optional[str]) -> str:
        if not value:
            return datetime.utcnow().isoformat()
        value = value.strip()
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(value).isoformat()
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt).isoformat()
            except ValueError:
                continue
        return datetime.utcnow().isoformat()

    @staticmethod
    def _event_id(source: str, timestamp: str, ordinal: int) -> str:
        raw = f"{source}-{timestamp}-{ordinal}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _update_token_statistics(
        self, context: Dict[str, Any], event: Dict[str, Any]
    ) -> None:
        stats = context.setdefault("token_statistics", {})
        for token in event.get("tokens", []):
            key = token["text"].lower()
            record = stats.setdefault(
                key,
                {
                    "token": token["text"],
                    "count": 0,
                    "labels": {},
                    "examples": [],
                    "sources": [],
                    "last_seen": None,
                    "weight": 0.0,
                },
            )
            record["count"] += 1
            record["last_seen"] = event["timestamp"]
            for label in token.get("labels", []):
                record["labels"][label["label"]] = max(
                    record["labels"].get(label["label"], 0.0), label["score"]
                )
            if len(record["examples"]) < 5:
                record["examples"].append(token["context"])
            if token["source"] and token["source"] not in record["sources"]:
                record["sources"].append(token["source"])
            max_label = max(record["labels"].values(), default=0.1)
            record["weight"] = round(record["count"] * max_label, 3)

    def _update_relations(
        self,
        context: Dict[str, Any],
        event: Dict[str, Any],
        relation_index: set,
    ) -> None:
        if not event.get("dominant_labels"):
            return
        events = context.get("events", {})
        for other_id, other in events.items():
            if other_id == event["event_id"]:
                continue
            shared = set(event["dominant_labels"]).intersection(
                other.get("dominant_labels", [])
            )
            if not shared:
                continue
            pair = tuple(sorted((event["event_id"], other_id)))
            if pair in relation_index:
                continue
            relation = {
                "source_event": pair[0],
                "target_event": pair[1],
                "shared_labels": sorted(shared),
                "weight": len(shared),
            }
            context.setdefault("relations", []).append(relation)
            relation_index.add(pair)

    def _update_origin_relation(
        self,
        context: Dict[str, Any],
        event: Dict[str, Any],
        relation_index: set,
    ) -> None:
        metadata = context.get("metadata", {})
        origin_id = metadata.get("origin_event_id")
        if not origin_id or origin_id == event["event_id"]:
            return
        origin_event = context.get("events", {}).get(origin_id)
        if not origin_event:
            return

        similarity = self._label_similarity(
            event.get("label_distribution"),
            origin_event.get("label_distribution"),
        )
        shared = sorted(
            set(event.get("dominant_labels", [])).intersection(
                origin_event.get("dominant_labels", [])
            )
        )

        pair = tuple(sorted((origin_id, event["event_id"])))
        if pair not in relation_index:
            relation_index.add(pair)
            context.setdefault("relations", []).insert(
                0,
                {
                    "source_event": origin_id,
                    "target_event": event["event_id"],
                    "shared_labels": shared,
                    "weight": round(similarity, 4),
                    "type": "origin_link",
                },
            )

        event["origin_similarity"] = similarity
        event["origin_shared_labels"] = shared
        intrinsic = event.get("intrinsic_weight", event.get("weight", 0.0))
        event["weight"] = round(max(intrinsic, similarity * 10), 3)

    def _decorate_result(self, result: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        annotated = dict(result)
        annotation = self._render_annotation(event)
        content = annotated.get("content") or ""
        annotated["content"] = f"{content}\n\n[structured_context]\n{annotation}".strip()
        annotated["structured_event_id"] = event["event_id"]
        return annotated

    @staticmethod
    def _label_similarity(
        primary: Optional[List[Dict[str, Any]]],
        secondary: Optional[List[Dict[str, Any]]],
    ) -> float:
        if not primary or not secondary:
            return 0.0
        reference = {item["label"]: item["score"] for item in secondary}
        overlap = 0.0
        for item in primary:
            label = item.get("label")
            if label in reference:
                overlap += min(item.get("score", 0.0), reference[label])
        return round(overlap, 4)

    @staticmethod
    def _render_annotation(event: Dict[str, Any]) -> str:
        label_line = ", ".join(
            f"{lbl['label']}({lbl['score']:.2f})" for lbl in event.get("top_labels", [])
        )
        token_lines = []
        for token in event.get("tokens", [])[:5]:
            labels = ", ".join(
                f"{lbl['label']}:{lbl['score']:.2f}" for lbl in token.get("labels", [])
            )
            token_lines.append(f"- {token['text']}: {labels}")
        token_section = "\n".join(token_lines)
        origin_lines = ""
        if event.get("origin_similarity") is not None:
            shared = ",".join(event.get("origin_shared_labels", []))
            shared_str = f"\nORIGIN_SHARED_LABELS={shared}" if shared else ""
            origin_lines = (
                f"\nORIGIN_SIMILARITY={event['origin_similarity']:.3f}{shared_str}"
            )
        return (
            f"EVENT_ID={event['event_id']}\n"
            f"TIMESTAMP={event['timestamp']}\n"
            f"PARAGRAPH={event.get('paragraph')}\n"
            f"STAGE={event.get('stage')}\n"
            f"WEIGHT={event.get('weight')}\n"
            f"LABELS={label_line}\n"
            f"TOKENS:\n{token_section}"
            f"{origin_lines}"
        )

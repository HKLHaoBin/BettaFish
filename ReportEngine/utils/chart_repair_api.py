"""
图表API修复模块。

提供调用4个Engine（ReportEngine, ForumEngine, InsightEngine, MediaEngine）的LLM API
来修复图表数据的功能。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from loguru import logger

from ReportEngine.utils.config import settings


# 图表修复提示词
CHART_REPAIR_SYSTEM_PROMPT = """你是一个专业的图表数据修复助手。你的任务是修复Chart.js图表数据中的格式错误，确保图表能够正常渲染。

**Chart.js标准数据格式：**

1. 标准图表（line, bar, pie, doughnut, radar, polarArea）：
```json
{
  "type": "widget",
  "widgetType": "chart.js/bar",
  "widgetId": "chart-001",
  "props": {
    "type": "bar",
    "title": "图表标题",
    "options": {
      "responsive": true,
      "plugins": {
        "legend": {
          "display": true
        }
      }
    }
  },
  "data": {
    "labels": ["A", "B", "C"],
    "datasets": [
      {
        "label": "系列1",
        "data": [10, 20, 30]
      }
    ]
  }
}
```

2. 特殊图表（scatter, bubble）：
```json
{
  "data": {
    "datasets": [
      {
        "label": "系列1",
        "data": [
          {"x": 10, "y": 20},
          {"x": 15, "y": 25}
        ]
      }
    ]
  }
}
```

**修复原则：**
1. **宁愿不改，也不要改错** - 如果不确定如何修复，保持原始数据
2. **最小改动** - 只修复明确的错误，不要过度修改
3. **保持数据完整性** - 不要丢失原始数据
4. **验证修复结果** - 确保修复后符合Chart.js格式

**常见错误及修复方法：**
1. 缺少labels字段 → 根据数据生成默认labels
2. datasets不是数组 → 转换为数组格式
3. 数据长度不匹配 → 截断或补null
4. 非数值数据 → 尝试转换或设为null
5. 缺少必需字段 → 添加默认值

请根据错误信息修复图表数据，并返回修复后的完整widget block（JSON格式）。
"""


def build_chart_repair_prompt(
    widget_block: Dict[str, Any],
    validation_errors: List[str]
) -> str:
    """
    构建图表修复提示词。

    Args:
        widget_block: 原始widget block
        validation_errors: 验证错误列表

    Returns:
        str: 提示词
    """
    block_json = json.dumps(widget_block, ensure_ascii=False, indent=2)
    errors_text = "\n".join(f"- {error}" for error in validation_errors)

    prompt = f"""请修复以下图表数据中的错误：

**原始数据：**
```json
{block_json}
```

**检测到的错误：**
{errors_text}

**要求：**
1. 返回修复后的完整widget block（JSON格式）
2. 只修复明确的错误，保持其他数据不变
3. 确保修复后的数据符合Chart.js格式要求
4. 如果无法确定如何修复，保持原始数据

**重要的输出格式要求：**
1. 只返回纯JSON对象，不要添加任何说明文字
2. 不要使用```json```标记包裹
3. 确保JSON语法完全正确
4. 所有字符串使用双引号
"""
    return prompt


def create_llm_repair_functions() -> List:
    """
    创建LLM修复函数列表。

    返回4个Engine的修复函数：
    1. ReportEngine
    2. ForumEngine (通过ForumHost)
    3. InsightEngine
    4. MediaEngine

    Returns:
        List[Callable]: 修复函数列表
    """
    repair_functions = []

    # 1. ReportEngine修复函数
    if settings.REPORT_ENGINE_API_KEY and settings.REPORT_ENGINE_BASE_URL:
        def repair_with_report_engine(widget_block: Dict[str, Any], errors: List[str]) -> Optional[Dict[str, Any]]:
            """使用ReportEngine的LLM修复图表"""
            try:
                from ReportEngine.llms import LLMClient

                client = LLMClient(
                    api_key=settings.REPORT_ENGINE_API_KEY,
                    base_url=settings.REPORT_ENGINE_BASE_URL,
                    model_name=settings.REPORT_ENGINE_MODEL_NAME or "gpt-4",
                )

                prompt = build_chart_repair_prompt(widget_block, errors)
                response = client.invoke(
                    CHART_REPAIR_SYSTEM_PROMPT,
                    prompt,
                    temperature=0.0,
                    top_p=0.05
                )

                if not response:
                    return None

                # 解析响应
                repaired = json.loads(response)
                return repaired

            except Exception as e:
                logger.error(f"ReportEngine图表修复失败: {e}")
                return None

        repair_functions.append(repair_with_report_engine)

    # 2. ForumEngine修复函数
    if settings.FORUM_HOST_API_KEY and settings.FORUM_HOST_BASE_URL:
        def repair_with_forum_engine(widget_block: Dict[str, Any], errors: List[str]) -> Optional[Dict[str, Any]]:
            """使用ForumEngine的LLM修复图表"""
            try:
                from ReportEngine.llms import LLMClient

                client = LLMClient(
                    api_key=settings.FORUM_HOST_API_KEY,
                    base_url=settings.FORUM_HOST_BASE_URL,
                    model_name=settings.FORUM_HOST_MODEL_NAME or "gpt-4",
                )

                prompt = build_chart_repair_prompt(widget_block, errors)
                response = client.invoke(
                    CHART_REPAIR_SYSTEM_PROMPT,
                    prompt,
                    temperature=0.0,
                    top_p=0.05
                )

                if not response:
                    return None

                repaired = json.loads(response)
                return repaired

            except Exception as e:
                logger.error(f"ForumEngine图表修复失败: {e}")
                return None

        repair_functions.append(repair_with_forum_engine)

    # 3. InsightEngine修复函数
    if settings.INSIGHT_ENGINE_API_KEY and settings.INSIGHT_ENGINE_BASE_URL:
        def repair_with_insight_engine(widget_block: Dict[str, Any], errors: List[str]) -> Optional[Dict[str, Any]]:
            """使用InsightEngine的LLM修复图表"""
            try:
                from ReportEngine.llms import LLMClient

                client = LLMClient(
                    api_key=settings.INSIGHT_ENGINE_API_KEY,
                    base_url=settings.INSIGHT_ENGINE_BASE_URL,
                    model_name=settings.INSIGHT_ENGINE_MODEL_NAME or "gpt-4",
                )

                prompt = build_chart_repair_prompt(widget_block, errors)
                response = client.invoke(
                    CHART_REPAIR_SYSTEM_PROMPT,
                    prompt,
                    temperature=0.0,
                    top_p=0.05
                )

                if not response:
                    return None

                repaired = json.loads(response)
                return repaired

            except Exception as e:
                logger.error(f"InsightEngine图表修复失败: {e}")
                return None

        repair_functions.append(repair_with_insight_engine)

    # 4. MediaEngine修复函数
    if settings.MEDIA_ENGINE_API_KEY and settings.MEDIA_ENGINE_BASE_URL:
        def repair_with_media_engine(widget_block: Dict[str, Any], errors: List[str]) -> Optional[Dict[str, Any]]:
            """使用MediaEngine的LLM修复图表"""
            try:
                from ReportEngine.llms import LLMClient

                client = LLMClient(
                    api_key=settings.MEDIA_ENGINE_API_KEY,
                    base_url=settings.MEDIA_ENGINE_BASE_URL,
                    model_name=settings.MEDIA_ENGINE_MODEL_NAME or "gpt-4",
                )

                prompt = build_chart_repair_prompt(widget_block, errors)
                response = client.invoke(
                    CHART_REPAIR_SYSTEM_PROMPT,
                    prompt,
                    temperature=0.0,
                    top_p=0.05
                )

                if not response:
                    return None

                repaired = json.loads(response)
                return repaired

            except Exception as e:
                logger.error(f"MediaEngine图表修复失败: {e}")
                return None

        repair_functions.append(repair_with_media_engine)

    if not repair_functions:
        logger.warning("未配置任何Engine API，图表API修复功能将不可用")

    return repair_functions

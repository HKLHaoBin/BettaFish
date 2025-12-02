"""
Report Engine渲染器集合。

提供 HTMLRenderer 和 PDFRenderer，支持HTML和PDF输出。
"""

from .html_renderer import HTMLRenderer
from .pdf_renderer import PDFRenderer
from .pdf_layout_optimizer import (
    PDFLayoutOptimizer,
    PDFLayoutConfig,
    PageLayout,
    KPICardLayout,
    CalloutLayout,
    TableLayout,
    ChartLayout,
    GridLayout,
)

__all__ = [
    "HTMLRenderer",
    "PDFRenderer",
    "PDFLayoutOptimizer",
    "PDFLayoutConfig",
    "PageLayout",
    "KPICardLayout",
    "CalloutLayout",
    "TableLayout",
    "ChartLayout",
    "GridLayout",
]

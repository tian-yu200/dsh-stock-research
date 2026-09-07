"""Cross-border (美股港股 QDII) exclusion shared helper.

Centralizes the keyword list so universe discovery, data_layer, rolling_scan
and research all use the same definition. Stocks are never considered
cross-border (e.g. 亚太实业), only etf/otc_fund.
"""
from __future__ import annotations

from typing import Iterable

CROSS_BORDER_KEYWORDS: tuple[str, ...] = (
    "标普", "纳斯达克", "纳指", "道琼斯", "美国", "美股", "美债", "美元",
    "恒生", "恒指", "港股", "沪港深", "港通", "中概", "海外", "全球", "环球",
    "亚太", "日经", "东京", "德国", "法国", "欧洲", "新兴", "金砖", "REIT", "QDII",
    "韩国", "中韩", "越南", "英国",
)

def is_cross_border(name: str | None, asset_type: str | None = None, keywords: Iterable[str] | None = None) -> bool:
    """Return True if name looks like a cross-border / QDII product.

    * stock is always False (avoid false positives like 亚太实业)
    * empty name -> False (fail-open, same as _pool_quality_filter)
    * latin keywords are matched case-insensitively via lower()
    """
    if asset_type == "stock":
        return False
    if not name:
        return False
    text = str(name)
    low = text.lower()
    kws = keywords if keywords is not None else CROSS_BORDER_KEYWORDS
    for kw in kws:
        if not kw:
            continue
        # latin case-insensitive, Chinese exact
        if kw.lower() in low:
            return True
        if kw in text:
            return True
    return False

"""研究结果微信推送钩子（Server酱）。

数据源:
  quant/results/assistant/current.json       最新发布（决策支持/数据截止日）
  quant/results/assistant/rolling_scan.json  滚动初选池（30只）

行为:
  current.json 的 run_id 未推送过且更新时间在 36 小时内 -> 组装摘要推送一次;
  否则静默退出（非交易日/研究未更新时不打扰）。
去重标记: quant/logs/wechat_push_state.json

由 Windows 任务计划「研究结果微信推送」每 30 分钟调用（15:30-22:30）。
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

QUANT = Path(__file__).resolve().parents[1] / "quant"
KEY_PATH = Path(__file__).resolve().parents[1] / "notify" / "serverchan.key"
STATE_PATH = QUANT / "logs" / "wechat_push_state.json"
CURRENT = QUANT / "results" / "assistant" / "current.json"
ROLLING = QUANT / "results" / "assistant" / "rolling_scan.json"
FRESH_WINDOW = timedelta(hours=36)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def send_wechat(title: str, body: str) -> None:
    key = KEY_PATH.read_text(encoding="utf-8").strip()
    data = urllib.parse.urlencode({"title": title[:32], "desp": body[:30000]}).encode()
    req = urllib.request.Request(f"https://sctapi.ftqq.com/{key}.send", data=data, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 绕过本机代理直连
    with opener.open(req, timeout=20) as resp:
        result = json.loads(resp.read())
    if result.get("code") != 0:
        raise RuntimeError(f"serverchan code={result.get('code')} {result.get('message')}")


def build_message(cur: dict, rolling: dict) -> tuple[str, str]:
    ds = cur.get("decision_support", {})
    paths = ds.get("asset_paths", {})

    def brief(kind: str) -> str:
        node = paths.get(kind, {})
        status = node.get("status", "?")
        action = node.get("action", "")
        weights = node.get("target_weights") or {}
        weight_text = "、".join(
            f"{code} {round(float(w) * 100, 1)}%" for code, w in weights.items()
        ) or "无个股权重"
        return f"- {kind}: {action or status}（{weight_text}）"

    lines = [
        f"**数据截止**: {cur.get('data_end_date', '?')}  |  运行 {cur.get('run_id', '?')}",
    ]
    blocked = (cur.get("publication") or {}).get("blocked_asset_types") or []
    if cur.get("status") == "partial" and blocked:
        lines.append(f"⚠️ **部分发布**：{'、'.join(blocked)} 未通过门槛，该部分按现金处理")
    lines += [
        "",
        "**模型建议**",
        brief("stock"),
        brief("etf"),
        brief("otc_fund"),
        f"- 组合现金比例: {round(float(ds.get('cash_weight') or 0) * 100, 1)}%",
        "",
        "**滚动初选池**（技术面得分降序）",
    ]
    pool = (rolling or {}).get("pool") or []
    for i, s in enumerate(pool[:30], 1):
        if s.get("asset_type") != "stock":
            continue
        trend = s.get("trend_score") or 0
        r60 = (s.get("return_60d") or 0) * 100
        amt = (s.get("average_amount_20d") or 0) / 1e8
        lines.append(
            f"{i}. `{s.get('symbol')}` {s.get('name')} — 趋势 {trend:.2f}, 60日 {r60:+.0f}%, 20日均额 {amt:.1f}亿"
        )
    lines += ["", f"覆盖 {((rolling or {}).get('coverage') or {}).get('coverage', 0):.1%} · "
              "非投资建议，目标权重基于假设资金，不连接券商不自动下单"]
    title = f"股票研究发布 {cur.get('data_end_date', '')} · 初选池{len([p for p in pool if p.get('asset_type')=='stock'])}只"
    return title, "\n".join(lines)


def main() -> int:
    cur = read_json(CURRENT)
    # partial 发布（如个股模型门槛未过、仅 ETF/场外基金就绪）也推送：
    # 没有它时用户完全收不到当日结果，而 partial 里的可用部分依然有价值。
    if not cur or cur.get("status") not in ("success", "partial"):
        print("no publishable result yet; skip")
        return 0
    updated = cur.get("updated_at") or ""
    try:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(updated)
    except Exception:
        print(f"cannot parse updated_at {updated!r}; skip")
        return 0
    if age > FRESH_WINDOW:
        print(f"publication is {age} old; not fresh; skip")
        return 0

    state = read_json(STATE_PATH) or {}
    run_id = cur.get("run_id", "")
    if state.get("last_pushed_run_id") == run_id:
        print(f"run {run_id} already pushed; skip")
        return 0

    rolling = read_json(ROLLING) or {}
    title, body = build_message(cur, rolling)
    send_wechat(title, body)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(
            {"last_pushed_run_id": run_id, "last_pushed_at": datetime.now(timezone.utc).isoformat()},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"pushed run {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

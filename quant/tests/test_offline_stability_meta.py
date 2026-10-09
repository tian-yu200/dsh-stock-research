"""离线稳定性元数据只读关联回归测试（阶段 2B 可观测性改进）。

锁定三件事：
1. ``research._offline_stability_summary`` 永远 fail-open（缺失/损坏/
   形态不符都不抛异常），且只做有界摘录、不触发任何训练。
2. 摘要键与门槛键无交集——``risk.model_gate_checks`` 的输入输出不受影响
   （稳定性检查**不接入阻断**，见《研究管线不变量》§6）。
3. manifest 新增 ``offline_stability`` 键不破坏既有读者的 ``.get`` 契约。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assistant import research  # noqa: E402
from assistant.risk import model_gate_checks  # noqa: E402

VALID_REPORT = {
    "status": "ok",
    "segment": "test",
    "dataset": {"fit_start_time": "2023-04-01", "fit_end_time": "2025-09-02"},
    "stability": {
        "stable": True,
        "ic_mean": 0.0727,
        "ic_std": 0.0029,
        "threshold": 0.02,
        "windows_requested": 3,
        "windows_used": 3,
    },
}


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_missing_report_is_fail_open(tmp_path: Path):
    summary = research._offline_stability_summary(tmp_path / "nope.json")
    assert summary == {"available": False, "reason": "unreadable report file"}


def test_corrupt_report_is_fail_open(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    summary = research._offline_stability_summary(bad)
    assert summary["available"] is False


def test_non_mapping_report_is_fail_open(tmp_path: Path):
    summary = research._offline_stability_summary(_write(tmp_path / "list.json", [1, 2]))
    assert summary == {"available": False, "reason": "report is not a mapping"}


def test_missing_stability_section_is_fail_open(tmp_path: Path):
    summary = research._offline_stability_summary(_write(tmp_path / "thin.json", {"status": "ok"}))
    assert summary == {"available": False, "reason": "missing stability section"}


def test_valid_report_summary_is_bounded(tmp_path: Path):
    summary = research._offline_stability_summary(_write(tmp_path / "rep.json", VALID_REPORT))
    assert summary["available"] is True
    assert summary["stable"] is True
    assert summary["ic_mean"] == 0.0727
    assert summary["ic_std"] == 0.0029
    assert summary["threshold"] == 0.02
    assert summary["windows_used"] == 3
    assert summary["fit_window"] == ["2023-04-01", "2025-09-02"]
    # 有界：不得把整份报告（含 timings/明细）原样搬进快照。
    assert "timings" not in summary
    assert "details" not in summary


def test_gate_checks_ignore_offline_keys():
    gate = {
        "min_mean_rank_ic": 0.02,
        "min_rank_ic_ir": 0.10,
        "min_top_bottom_excess_return": 0.0,
        "max_test_drawdown": 0.25,
        "max_excess_test_drawdown": 0.15,
        "max_ic_std": 0.02,
    }
    metrics = {
        "model_available": True,
        "mean_rank_ic": 0.06,
        "rank_ic_ir": 0.4,
        "top_bottom_excess_return": 0.01,
        "test_max_drawdown": -0.05,
        "benchmark": {"max_drawdown": -0.10},
        "stability_stable": None,
        "stability_ic_std": float("nan"),
    }
    before = model_gate_checks(metrics, gate)
    # 模拟"有人误把 offline 摘要塞进 metrics"：门槛结果必须纹丝不动。
    after = model_gate_checks({**metrics, "offline_stability": {"available": True}}, gate)
    assert before == after
    assert "offline_stability" not in before

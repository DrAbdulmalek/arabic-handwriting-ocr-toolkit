#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only, privacy-preserving analysis of manual crop-box adjustments.

Only geometry is retained in the report. Labels, page names, paths, and image
contents are intentionally excluded. Recommendations are advisory only.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


_BOX_KEYS = ("x1", "y1", "x2", "y2")


def _box(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    result = {}
    for key in _BOX_KEYS:
        number = value.get(key)
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            return None
        number = float(number)
        if not math.isfinite(number):
            return None
        result[key] = number
    if result["x2"] <= result["x1"] or result["y2"] <= result["y1"]:
        return None
    return result


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def analyze_adjustments(
    log_path: str | Path,
    *,
    min_samples: int = 20,
    consistency_threshold: float = 0.70,
) -> dict[str, Any]:
    """Summarize net geometry edits per crop without exposing text or paths."""
    if min_samples < 3:
        raise ValueError("min_samples must be at least 3")
    if not 0.5 <= consistency_threshold <= 1.0:
        raise ValueError("consistency_threshold must be between 0.5 and 1.0")
    path = Path(log_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Adjustment log does not exist: {path}")

    grouped: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    malformed = 0
    ignored = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                malformed += 1
                continue
            if not isinstance(event, dict) or event.get("action") != "update":
                ignored += 1
                continue
            old = _box(event.get("old"))
            new = _box(event.get("new"))
            if old is None or new is None:
                ignored += 1
                continue
            crop_id = str(event.get("id", ""))
            page = str(event.get("page", ""))
            if not crop_id or not page:
                ignored += 1
                continue
            key = (page, crop_id)
            if key not in grouped:
                grouped[key] = {"initial": old, "final": new}
            else:
                grouped[key]["final"] = new

    deltas = []
    for values in grouped.values():
        old, new = values["initial"], values["final"]
        old_w, new_w = old["x2"] - old["x1"], new["x2"] - new["x1"]
        old_h, new_h = old["y2"] - old["y1"], new["y2"] - new["y1"]
        deltas.append({
            "dx1": new["x1"] - old["x1"],
            "dy1": new["y1"] - old["y1"],
            "dx2": new["x2"] - old["x2"],
            "dy2": new["y2"] - old["y2"],
            "width_delta": new_w - old_w,
            "height_delta": new_h - old_h,
            "width_delta_pct": (new_w - old_w) / old_w * 100.0,
            "height_delta_pct": (new_h - old_h) / old_h * 100.0,
        })

    n = len(deltas)
    width_widen_fraction = (
        sum(1 for row in deltas if row["width_delta"] > 0) / n if n else 0.0
    )
    width_narrow_fraction = (
        sum(1 for row in deltas if row["width_delta"] < 0) / n if n else 0.0
    )
    height_widen_fraction = (
        sum(1 for row in deltas if row["height_delta"] > 0) / n if n else 0.0
    )
    height_narrow_fraction = (
        sum(1 for row in deltas if row["height_delta"] < 0) / n if n else 0.0
    )

    recommendations = []
    if n < min_samples:
        recommendations.append({
            "status": "INSUFFICIENT_DATA",
            "message": f"Need at least {min_samples} distinct crops; found {n}.",
        })
    else:
        for axis, median_key, widen, narrow in (
            ("horizontal", "width_delta_pct", width_widen_fraction, width_narrow_fraction),
            ("vertical", "height_delta_pct", height_widen_fraction, height_narrow_fraction),
        ):
            if widen >= consistency_threshold:
                direction = "widen"
                confidence = widen
            elif narrow >= consistency_threshold:
                direction = "narrow"
                confidence = narrow
            else:
                recommendations.append({
                    "axis": axis,
                    "status": "NO_CONSISTENT_SIGNAL",
                    "message": "Manual adjustments are mixed; do not change crop parameters from this sample.",
                })
                continue
            recommendations.append({
                "axis": axis,
                "status": "ADVISORY_CALIBRATION_CANDIDATE",
                "direction": direction,
                "median_relative_change_pct": _median([r[median_key] for r in deltas]),
                "consistency": round(confidence, 3),
                "message": (
                    f"Manual edits consistently {direction} {axis} bounds. "
                    "Test a small parameter change on a held-out page set; do not auto-apply."
                ),
            })

    return {
        "report_version": 1,
        "source_type": "BOX_ADJUSTMENT_GEOMETRY_ONLY",
        "distinct_crops": n,
        "malformed_lines": malformed,
        "ignored_events": ignored,
        "geometry_medians_px": {
            key: _median([row[key] for row in deltas])
            for key in ("dx1", "dy1", "dx2", "dy2", "width_delta", "height_delta")
        },
        "median_relative_change_pct": {
            "width": _median([row["width_delta_pct"] for row in deltas]),
            "height": _median([row["height_delta_pct"] for row in deltas]),
        },
        "direction_consistency": {
            "horizontal_widen": round(width_widen_fraction, 3),
            "horizontal_narrow": round(width_narrow_fraction, 3),
            "vertical_widen": round(height_widen_fraction, 3),
            "vertical_narrow": round(height_narrow_fraction, 3),
        },
        "recommendations": recommendations,
        "auto_applied": False,
        "privacy": "No labels, page names, crop paths, image bytes, or identifiers are included.",
    }


def main():
    parser = argparse.ArgumentParser(description="تحليل هندسي محلي لتعديلات مربعات القص")
    parser.add_argument("--log", required=True, help="مسار box_adjustments.jsonl")
    parser.add_argument("--out", required=True, help="ملف JSON جديد؛ الموجود يُرفض")
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--consistency", type=float, default=0.70)
    args = parser.parse_args()
    report = analyze_adjustments(
        args.log, min_samples=args.min_samples,
        consistency_threshold=args.consistency,
    )
    out = Path(args.out).expanduser().resolve(strict=False)
    if out.exists():
        parser.error("ملف التقرير موجود مسبقاً؛ لن يُستبدل.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

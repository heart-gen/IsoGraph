"""Aggregate multiplex stress benchmark reports against WGCNA.

Usage:
    python scripts/stress_multiplex_summary.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_STAGES = {
    "vae": "stress_multiplex_vae",
    "graph": "stress_multiplex_graph",
    "latent": "stress_multiplex_latent",
    "wgcna": "stress_multiplex_wgcna",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required report: {path}")
    return json.loads(path.read_text())


def _flatten_role_recall(row: dict[str, Any]) -> dict[str, float | None]:
    role = row.get("role_recall") or {}
    return {
        f"role_recall_{key}": role.get(key)
        for key in ("switch_only", "abundance_only", "coupled", "discordant")
    }


def _calibration_by_fixture(report_root: Path, stage: str) -> dict[str, dict[str, Any]]:
    path = report_root / f"{stage}-calibration.json"
    if not path.exists():
        return {}
    payload = _read_json(path)
    rows = payload.get("calibration_by_fixture", [])
    return {str(row["fixture"]): row for row in rows if "fixture" in row}


def _load_stage(report_root: Path, backend: str, stage: str) -> list[dict[str, Any]]:
    benchmark = _read_json(report_root / f"{stage}-benchmark.json")
    runtime = _read_json(report_root / f"{stage}-runtime-memory.json")
    peak_by_dataset = {
        str(row["dataset"]): row.get("peak_memory_bytes") for row in runtime.get("results", [])
    }
    cal_by_dataset = _calibration_by_fixture(report_root, stage)

    rows: list[dict[str, Any]] = []
    for row in benchmark.get("results", []):
        dataset = str(row["dataset"])
        cal = cal_by_dataset.get(dataset, {})
        out = {
            "backend": backend,
            "stage": stage,
            "dataset": dataset,
            "n_samples": row.get("n_samples"),
            "n_genes": row.get("n_genes"),
            "n_modules": row.get("n_modules"),
            "n_edges": row.get("n_edges"),
            "recovery": row.get("recovery"),
            "runtime_seconds": row.get("runtime_seconds"),
            "peak_memory_bytes": peak_by_dataset.get(dataset),
            "giant_component_fraction": row.get("giant_component_fraction"),
            "selected_alpha_abundance": row.get("selected_alpha_abundance"),
            "wgcna_soft_threshold_power": cal.get("wgcna_soft_threshold_power"),
            "wgcna_sft_r2": cal.get("wgcna_sft_r2"),
            "wgcna_n_modules": cal.get("wgcna_n_modules"),
            "wgcna_n_unassigned_genes": cal.get("wgcna_n_unassigned_genes"),
        }
        out.update(_flatten_role_recall(row))
        rows.append(out)
    return rows


def build_summary(report_root: Path, stages: dict[str, str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for backend, stage in stages.items():
        rows.extend(_load_stage(report_root, backend, stage))

    wgcna_by_dataset = {row["dataset"]: row for row in rows if row["backend"] == "wgcna"}
    for row in rows:
        ref = wgcna_by_dataset.get(row["dataset"])
        if ref is None or row["backend"] == "wgcna":
            row["delta_recovery_vs_wgcna"] = None
            row["delta_runtime_seconds_vs_wgcna"] = None
            row["runtime_ratio_vs_wgcna"] = None
            continue
        recovery = row.get("recovery")
        ref_recovery = ref.get("recovery")
        runtime = row.get("runtime_seconds")
        ref_runtime = ref.get("runtime_seconds")
        row["delta_recovery_vs_wgcna"] = (
            None if recovery is None or ref_recovery is None else recovery - ref_recovery
        )
        row["delta_runtime_seconds_vs_wgcna"] = (
            None if runtime is None or ref_runtime is None else runtime - ref_runtime
        )
        row["runtime_ratio_vs_wgcna"] = (
            None if runtime is None or not ref_runtime else runtime / ref_runtime
        )

    return {
        "suite": "multiplex_v1",
        "reference_backend": "wgcna",
        "stages": stages,
        "results": sorted(rows, key=lambda r: (r["dataset"], r["backend"])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", default="artifacts/reports")
    parser.add_argument(
        "--output",
        default="artifacts/reports/stress-multiplex-backend-summary.json",
    )
    args = parser.parse_args(argv)

    report_root = Path(args.report_root)
    output = Path(args.output)
    summary = build_summary(report_root, DEFAULT_STAGES)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

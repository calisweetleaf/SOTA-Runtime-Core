"""
One-off smoke test for neural_router.NeuralPromptRouter.

Runs a single forward pass with synthetic inputs and writes JSON/Markdown
reports plus a detailed operation manifest to both:
- ./reports/
- ./output/
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

import torch

from neural_router import NeuralPromptRouter, RouterConfig


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _log_step(
    operation_log: List[Dict[str, Any]],
    step: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
    started_at: Optional[float] = None,
) -> None:
    entry: Dict[str, Any] = {
        "timestamp": _utc_iso_now(),
        "step": step,
        "status": status,
    }
    if details:
        entry["details"] = details
    if started_at is not None:
        entry["duration_ms"] = round((perf_counter() - started_at) * 1000, 3)
    operation_log.append(entry)


def _safe_print(text: str) -> None:
    """Print text even on narrow Windows console encodings."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def run_single_pass(seed: int = 42) -> Dict[str, Any]:
    """Execute one router forward pass with reproducible random inputs."""
    operation_log: List[Dict[str, Any]] = []
    run_started = perf_counter()

    t0 = perf_counter()
    torch.manual_seed(seed)
    _log_step(operation_log, "set_seed", "ok", {"seed": seed}, t0)

    t0 = perf_counter()
    config = RouterConfig()
    _log_step(
        operation_log,
        "init_config",
        "ok",
        {"context_dim": config.context_dim, "num_templates": config.num_templates},
        t0,
    )

    t0 = perf_counter()
    router = NeuralPromptRouter(config)
    _log_step(operation_log, "init_router", "ok", {"router_class": "NeuralPromptRouter"}, t0)

    # Synthetic but structured metadata to exercise validation paths.
    context_metadata = {
        "user_tier": "free",
        "message_count": 2,
        "has_tool_calls": True,
        "requires_determinism": False,
    }

    t0 = perf_counter()
    batch = {
        "message_embs": torch.randn(1, 10, config.context_dim),
        "user_profile": torch.randn(1, 128),
        "metadata": torch.randn(1, 64),
        "context_metadata": context_metadata,
    }
    _log_step(
        operation_log,
        "build_batch",
        "ok",
        {
            "message_embs_shape": list(batch["message_embs"].shape),
            "user_profile_shape": list(batch["user_profile"].shape),
            "metadata_shape": list(batch["metadata"].shape),
        },
        t0,
    )

    t0 = perf_counter()
    prompt, trace = router(**batch, return_trace=True)
    _log_step(
        operation_log,
        "router_forward",
        "ok",
        {
            "prompt_length": len(prompt),
            "selected_template": (trace or {}).get("selected_template"),
            "output_issues_count": len((trace or {}).get("output_issues") or []),
        },
        t0,
    )

    _log_step(
        operation_log,
        "run_complete",
        "ok",
        {"total_runtime_ms": round((perf_counter() - run_started) * 1000, 3)},
    )

    return {
        "timestamp": _utc_iso_now(),
        "seed": seed,
        "config": asdict(config),
        "context_metadata": context_metadata,
        "prompt": prompt,
        "trace": trace,
        "prompt_preview": prompt[:400],
        "operation_log": operation_log,
    }


def write_reports(report: Dict[str, Any], out_dir: Path) -> Dict[str, str]:
    """Persist JSON/Markdown reports plus operation manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "neural_router_smoke.json"
    md_path = out_dir / "neural_router_smoke.md"
    prompt_path = out_dir / "neural_router_prompt.txt"
    manifest_path = out_dir / "neural_router_smoke_manifest.json"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    prompt_path.write_text(report["prompt"], encoding="utf-8")

    trace = report["trace"] or {}
    safety_violations = trace.get("safety_violations") or []
    output_issues = trace.get("output_issues") or []
    operation_log = report.get("operation_log") or []

    md_lines = [
        "# Neural Router Smoke Test",
        f"- Timestamp: {report['timestamp']}",
        f"- Seed: {report['seed']}",
        f"- Prompt length: {trace.get('prompt_length', 'n/a')}",
        f"- Selected template: {trace.get('selected_template', 'n/a')}",
        f"- Confidence: {trace.get('slot_predictions', {}).get('confidence', 'n/a')}",
        "",
        "## Safety Violations",
        "None" if not safety_violations else "",
    ]

    for violation in safety_violations:
        md_lines.append(
            f"- [{violation.get('severity','')}] {violation.get('rule','')}: "
            f"{violation.get('message','')}"
        )

    md_lines.extend(["", "## Output Issues", "None" if not output_issues else ""])
    for issue in output_issues:
        md_lines.append(f"- {issue}")

    md_lines.extend(
        [
            "",
            "## Prompt Preview",
            "```",
            report["prompt_preview"],
            "```",
            "",
            "## Operation Log",
        ]
    )
    for op in operation_log:
        duration = op.get("duration_ms")
        duration_text = f"{duration}ms" if duration is not None else "n/a"
        md_lines.append(
            f"- {op.get('timestamp')} | {op.get('step')} | {op.get('status')} | {duration_text}"
        )

    md_lines.extend(["", "## Trace (truncated)", "```json", json.dumps(trace, indent=2)[:2000], "```"])

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    manifest = {
        "generated_utc": _utc_iso_now(),
        "suite": "neural_router_smoke",
        "artifacts": {
            "json": str(json_path),
            "markdown": str(md_path),
            "prompt": str(prompt_path),
        },
        "summary": {
            "selected_template": trace.get("selected_template"),
            "prompt_length": trace.get("prompt_length"),
            "safety_violations_count": len(safety_violations),
            "output_issues_count": len(output_issues),
        },
        "operation_log": operation_log,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "prompt": str(prompt_path),
        "manifest": str(manifest_path),
    }


def main() -> None:
    report = run_single_pass()
    reports_paths = write_reports(report, Path("reports"))
    output_paths = write_reports(report, Path("output"))
    print("Smoke test complete.")
    print(f"Reports JSON: {reports_paths['json']}")
    print(f"Reports Markdown: {reports_paths['markdown']}")
    print(f"Reports Manifest: {reports_paths['manifest']}")
    print(f"Output JSON: {output_paths['json']}")
    print(f"Output Markdown: {output_paths['markdown']}")
    print(f"Output Manifest: {output_paths['manifest']}")
    print("\n=== Final Generated Prompt (Begin) ===\n")
    _safe_print(report["prompt"])
    print("\n=== Final Generated Prompt (End) ===")


if __name__ == "__main__":
    main()

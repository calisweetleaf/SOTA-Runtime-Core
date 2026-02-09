"""
Standalone smoke test for integrated_system.IntegratedChatSystem.

Installs a compatibility shim for router import wiring, executes one integrated
request, and writes JSON/Markdown reports plus an operation manifest to:
- ./reports/
- ./output/
"""

from __future__ import annotations

import asyncio
import builtins
import json
import sys
import types
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

import numpy as np
from memory_injection_system import ConversationContext
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


def install_router_shim() -> None:
    shim = types.ModuleType("router_core_implementation")
    shim.NeuralPromptRouter = NeuralPromptRouter
    shim.RouterConfig = RouterConfig
    sys.modules["router_core_implementation"] = shim


async def run_single_pass() -> Dict[str, Any]:
    operation_log: List[Dict[str, Any]] = []
    run_started = perf_counter()

    t0 = perf_counter()
    install_router_shim()
    _log_step(operation_log, "install_router_shim", "ok", {"module": "router_core_implementation"}, t0)

    t0 = perf_counter()
    builtins.ConversationContext = ConversationContext
    builtins.np = np
    try:
        import integrated_system  # Imported after shim installation.
    finally:
        if hasattr(builtins, "ConversationContext"):
            delattr(builtins, "ConversationContext")
        if hasattr(builtins, "np"):
            delattr(builtins, "np")
    _log_step(operation_log, "import_integrated_system", "ok", {"module": "integrated_system"}, t0)

    # Bridge missing symbol expected by integrated_system.process_chat.
    t0 = perf_counter()
    integrated_system.ConversationContext = ConversationContext
    _log_step(operation_log, "bridge_conversation_context", "ok", {}, t0)

    t0 = perf_counter()
    system = integrated_system.IntegratedChatSystem(
        router_config={"context_dim": 768, "num_templates": 16, "num_tools": 32},
        memory_config={
            "auto_extract": True,
            "extraction_interval": 5,
            "max_context_tokens": 2000,
            "storage_backend": "in_memory",
        },
        enable_memory=True,
        enable_neural_router=True,
    )
    _log_step(
        operation_log,
        "init_integrated_system",
        "ok",
        {"enable_memory": True, "enable_neural_router": True},
        t0,
    )

    request = integrated_system.ChatRequest(
        user_id="integrated_smoke_user",
        conversation_id="integrated_smoke_conv",
        message="Help me package this router and memory toolkit for release.",
        message_history=[
            {"role": "user", "content": "I need a production-ready release workflow."},
            {"role": "assistant", "content": "I can help design one."},
        ],
        user_profile={"tier": "free", "name": "SmokeUser"},
        metadata={"message_count": 3, "has_code": True, "user_tier": "free"},
    )

    t0 = perf_counter()
    response = await system.process_chat(request)
    _log_step(
        operation_log,
        "process_chat",
        "ok",
        {
            "processing_time_seconds": response.processing_time,
            "router_trace_keys": sorted(list(response.router_trace.keys())),
            "memories_used": response.memories_used,
        },
        t0,
    )

    t0 = perf_counter()
    await asyncio.sleep(0.05)
    _log_step(operation_log, "await_background_cycle", "ok", {"sleep_seconds": 0.05}, t0)

    t0 = perf_counter()
    memory_summary = await system.memory_manager.get_user_memory_summary(request.user_id)
    _log_step(
        operation_log,
        "summarize_memory",
        "ok",
        {"total_memories": memory_summary.get("total_memories", 0)},
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
        "request": asdict(request),
        "response": {
            "response_preview": response.response[:300],
            "system_prompt_preview": response.system_prompt[:500],
            "memories_used": response.memories_used,
            "router_trace_keys": sorted(list(response.router_trace.keys())),
            "processing_time_seconds": response.processing_time,
        },
        "memory_summary": memory_summary,
        "compatibility_shim": {
            "router_core_implementation": True,
            "conversation_context_bridge": True,
        },
        "operation_log": operation_log,
    }


def write_reports(report: Dict[str, Any], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "integrated_system_smoke.json"
    md_path = out_dir / "integrated_system_smoke.md"
    manifest_path = out_dir / "integrated_system_smoke_manifest.json"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    operation_log = report.get("operation_log") or []
    md_lines = [
        "# Integrated System Smoke Test",
        f"- Timestamp: {report['timestamp']}",
        f"- Compatibility shim: {report['compatibility_shim']}",
        f"- Processing time (s): {report['response']['processing_time_seconds']}",
        f"- Router trace keys: {', '.join(report['response']['router_trace_keys'])}",
        "",
        "## Response Preview",
        "```",
        report["response"]["response_preview"] or "(none)",
        "```",
        "",
        "## System Prompt Preview",
        "```",
        report["response"]["system_prompt_preview"] or "(none)",
        "```",
        "",
        "## Memory Summary",
        "```json",
        json.dumps(report["memory_summary"], indent=2, default=str),
        "```",
        "",
        "## Operation Log",
    ]

    for op in operation_log:
        duration = op.get("duration_ms")
        duration_text = f"{duration}ms" if duration is not None else "n/a"
        md_lines.append(
            f"- {op.get('timestamp')} | {op.get('step')} | {op.get('status')} | {duration_text}"
        )

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    manifest = {
        "generated_utc": _utc_iso_now(),
        "suite": "integrated_system_smoke",
        "artifacts": {
            "json": str(json_path),
            "markdown": str(md_path),
        },
        "summary": {
            "processing_time_seconds": report["response"].get("processing_time_seconds"),
            "router_trace_keys_count": len(report["response"].get("router_trace_keys", [])),
            "memories_used": report["response"].get("memories_used", []),
            "total_memories": report.get("memory_summary", {}).get("total_memories", 0),
        },
        "operation_log": operation_log,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {"json": str(json_path), "markdown": str(md_path), "manifest": str(manifest_path)}


async def main() -> None:
    report = await run_single_pass()
    reports_paths = write_reports(report, Path("reports"))
    output_paths = write_reports(report, Path("output"))
    print("Integrated system smoke test complete.")
    print(f"Reports JSON: {reports_paths['json']}")
    print(f"Reports Markdown: {reports_paths['markdown']}")
    print(f"Reports Manifest: {reports_paths['manifest']}")
    print(f"Output JSON: {output_paths['json']}")
    print(f"Output Markdown: {output_paths['markdown']}")
    print(f"Output Manifest: {output_paths['manifest']}")


if __name__ == "__main__":
    asyncio.run(main())

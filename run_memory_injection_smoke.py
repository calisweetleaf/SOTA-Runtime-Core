"""
Standalone smoke test for memory_injection_system.MemoryManager.

Runs a deterministic in-process scenario and writes JSON/Markdown reports plus
operation manifest to:
- ./reports/
- ./output/
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

import numpy as np

from memory_injection_system import (
    ConversationContext,
    MemoryItem,
    MemoryManager,
    MemoryType,
)


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


async def run_single_pass(seed: int = 42) -> Dict[str, Any]:
    operation_log: List[Dict[str, Any]] = []
    run_started = perf_counter()

    t0 = perf_counter()
    np.random.seed(seed)
    _log_step(operation_log, "set_seed", "ok", {"seed": seed}, t0)

    t0 = perf_counter()
    manager = MemoryManager(
        config={
            "auto_extract": True,
            "extraction_interval": 5,
            "max_context_tokens": 2000,
            "storage_backend": "in_memory",
        }
    )
    _log_step(operation_log, "init_memory_manager", "ok", {"storage_backend": "in_memory"}, t0)

    user_id = "smoke_user"
    t0 = perf_counter()
    conversation = ConversationContext(
        conversation_id="smoke_conv_001",
        user_id=user_id,
        messages=[
            {"role": "user", "content": "I prefer concise architecture summaries."},
            {"role": "assistant", "content": "Noted. I will keep outputs concise."},
            {"role": "user", "content": "I am building a neural routing sandbox."},
        ],
        current_topic="architecture",
        metadata={"message_count": 3, "user_tier": "free"},
    )
    _log_step(operation_log, "build_conversation_context", "ok", {"message_count": 3}, t0)

    t0 = perf_counter()
    extracted = await manager.process_conversation_turn(conversation, should_extract=True)
    _log_step(operation_log, "extract_memories", "ok", {"extracted_count": len(extracted)}, t0)

    seeded_memories: List[MemoryItem] = [
        MemoryItem(
            id="smoke_mem_pref",
            user_id=user_id,
            memory_type=MemoryType.PREFERENCE,
            content="Prefers concise technical responses.",
            importance_score=0.9,
        ),
        MemoryItem(
            id="smoke_mem_fact",
            user_id=user_id,
            memory_type=MemoryType.FACT,
            content="Building a neural prompt router toolkit.",
            importance_score=0.8,
        ),
        MemoryItem(
            id="smoke_mem_topic",
            user_id=user_id,
            memory_type=MemoryType.TOPIC,
            content="Interested in production-grade validation workflows.",
            importance_score=0.7,
        ),
    ]

    t0 = perf_counter()
    for memory in seeded_memories:
        await manager.memory_store.store_memory(memory)
    _log_step(operation_log, "seed_memories", "ok", {"seeded_count": len(seeded_memories)}, t0)

    query_messages = [{"role": "user", "content": "Help me validate my router release."}]
    t0 = perf_counter()
    with_memory = await manager.prepare_prompt_with_memory(messages=query_messages, user_id=user_id)
    _log_step(
        operation_log,
        "inject_memory_block",
        "ok",
        {
            "messages_returned": len(with_memory),
            "developer_memory_block": bool(with_memory and with_memory[0].get("role") == "developer"),
        },
        t0,
    )

    t0 = perf_counter()
    summary = await manager.get_user_memory_summary(user_id)
    _log_step(
        operation_log,
        "summarize_memory",
        "ok",
        {"total_memories": summary.get("total_memories", 0)},
        t0,
    )

    memory_block_preview = ""
    if with_memory and with_memory[0].get("role") == "developer":
        memory_block_preview = with_memory[0].get("content", "")[:500]

    _log_step(
        operation_log,
        "run_complete",
        "ok",
        {"total_runtime_ms": round((perf_counter() - run_started) * 1000, 3)},
    )

    return {
        "timestamp": _utc_iso_now(),
        "seed": seed,
        "config": {
            "auto_extract": True,
            "extraction_interval": 5,
            "max_context_tokens": 2000,
            "storage_backend": "in_memory",
        },
        "conversation": asdict(conversation),
        "extracted_count": len(extracted),
        "seeded_memory_ids": [m.id for m in seeded_memories],
        "messages_injected_count": len(with_memory),
        "has_developer_memory_block": bool(with_memory and with_memory[0].get("role") == "developer"),
        "memory_block_preview": memory_block_preview,
        "memory_summary": summary,
        "operation_log": operation_log,
    }


def write_reports(report: Dict[str, Any], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "memory_injection_smoke.json"
    md_path = out_dir / "memory_injection_smoke.md"
    manifest_path = out_dir / "memory_injection_smoke_manifest.json"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    operation_log = report.get("operation_log") or []
    md_lines = [
        "# Memory Injection Smoke Test",
        f"- Timestamp: {report['timestamp']}",
        f"- Seed: {report['seed']}",
        f"- Extracted memories: {report['extracted_count']}",
        f"- Seeded memory entries: {len(report['seeded_memory_ids'])}",
        f"- Messages returned after injection: {report['messages_injected_count']}",
        f"- Developer memory block injected: {report['has_developer_memory_block']}",
        "",
        "## Memory Summary",
        "```json",
        json.dumps(report["memory_summary"], indent=2, default=str),
        "```",
        "",
        "## Memory Block Preview",
        "```",
        report["memory_block_preview"] or "(none)",
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
        "suite": "memory_injection_smoke",
        "artifacts": {
            "json": str(json_path),
            "markdown": str(md_path),
        },
        "summary": {
            "extracted_count": report.get("extracted_count", 0),
            "seeded_count": len(report.get("seeded_memory_ids", [])),
            "developer_memory_block": report.get("has_developer_memory_block", False),
            "messages_injected_count": report.get("messages_injected_count", 0),
        },
        "operation_log": operation_log,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {"json": str(json_path), "markdown": str(md_path), "manifest": str(manifest_path)}


async def main() -> None:
    report = await run_single_pass()
    reports_paths = write_reports(report, Path("reports"))
    output_paths = write_reports(report, Path("output"))
    print("Memory smoke test complete.")
    print(f"Reports JSON: {reports_paths['json']}")
    print(f"Reports Markdown: {reports_paths['markdown']}")
    print(f"Reports Manifest: {reports_paths['manifest']}")
    print(f"Output JSON: {output_paths['json']}")
    print(f"Output Markdown: {output_paths['markdown']}")
    print(f"Output Manifest: {output_paths['manifest']}")


if __name__ == "__main__":
    asyncio.run(main())

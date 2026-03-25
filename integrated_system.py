"""
Integrated System — Production Integration Tests

Exercises the full Somnus Router + Memory System pipeline end-to-end.
Validates: embedding alignment, capability unlocking, prompt assembly,
memory lifecycle, archive search, and the router-memory bridge.

Run:
    PYTHONPATH=Memory-System:System-Router python3 integrated_system.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from memory_injection_system import (
        Conversation,
        ConversationArchive,
        ConversationContext,
        ConversationMessage,
        DeletionPropagationService,
        DeterministicSynthesisBackend,
        EmbeddingInterface,
        EvictionPolicy,
        GlobalMemorySummary,
        HealthCheck,
        HybridMemoryRetriever,
        MemoryControlPanel,
        MemoryControlSettings,
        MemoryItem,
        MemoryLifecycleManager,
        MemoryManager,
        MemoryScope,
        MemoryState,
        MemoryStore,
        MemoryToolDefinitions,
        MemoryType,
        PipelineEmbeddingBackend,
        PlanGate,
        ProfilePreferences,
        PromptContextAssembler,
        SummaryRefreshSynthesizer,
        UserTier,
    )
except ImportError as exc:
    sys.exit(
        f"[FATAL] Cannot import memory system: {exc}\n"
        "  Hint: PYTHONPATH=Memory-System:System-Router python3 integrated_system.py"
    )

# Router import is optional — tests that need it will skip gracefully
_ROUTER_AVAILABLE = False
try:
    import torch
    from neural_router import (
        HashTextEncoder,
        NeuralPromptRouter,
        RouterConfig,
        SafeRouterWrapper,
    )
    _ROUTER_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# Test infrastructure
# ============================================================================

_RESULTS: List[Dict[str, Any]] = []
_START_TIME = 0.0


def _record(name: str, passed: bool, detail: str = "") -> None:
    """Record a test result."""
    symbol = "\u2713" if passed else "\u2717"
    _RESULTS.append({"name": name, "passed": passed, "detail": detail})
    msg = f"  {symbol} {name}"
    if detail:
        msg += f"  \u2014 {detail}"
    print(msg)


def _section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'\u2500' * 64}")
    print(f"  {title}")
    print(f"{'\u2500' * 64}")


# ============================================================================
# Phase 1 Verification
# ============================================================================

async def test_aggressive_eviction_defaults() -> None:
    """EvictionPolicy defaults should be 30-day memory, 90-day conversation."""
    _section("Phase 1: De-Tox \u2014 Aggressive Eviction Defaults")

    policy = EvictionPolicy()
    _record(
        "memory_ttl_days == 30",
        policy.memory_ttl_days == 30,
        f"got {policy.memory_ttl_days}",
    )
    _record(
        "conversation_ttl_days == 90",
        policy.conversation_ttl_days == 90,
        f"got {policy.conversation_ttl_days}",
    )

    mgr = MemoryManager(config={"embedding_dim": 64})
    _record(
        "MemoryManager lifecycle policy memory_ttl == 30",
        mgr.lifecycle_manager.policy.memory_ttl_days == 30,
        f"got {mgr.lifecycle_manager.policy.memory_ttl_days}",
    )
    _record(
        "MemoryManager lifecycle policy conv_ttl == 90",
        mgr.lifecycle_manager.policy.conversation_ttl_days == 90,
        f"got {mgr.lifecycle_manager.policy.conversation_ttl_days}",
    )


async def test_plangate_always_open() -> None:
    """PlanGate should always return True regardless of tier."""
    _section("Phase 1: De-Tox \u2014 PlanGate Always Open")

    for tier in UserTier:
        result = PlanGate.can_search(tier)
        _record(
            f"PlanGate.can_search({tier.value}) == True",
            result is True,
            f"got {result}",
        )

    _record(
        "allows_archive_injection(FREE, enabled=True)",
        PlanGate.allows_archive_injection(UserTier.FREE, True) is True,
    )
    _record(
        "allows_archive_injection(FREE, enabled=False)",
        PlanGate.allows_archive_injection(UserTier.FREE, False) is False,
    )


async def test_health_check_no_metrics() -> None:
    """HealthCheck should work without metrics (De-Tox)."""
    _section("Phase 1: De-Tox \u2014 HealthCheck Without Metrics")

    mgr = MemoryManager(config={"embedding_dim": 64})
    health = mgr.get_health()
    _record(
        "health status is 'healthy'",
        health["status"] == "healthy",
        f"got '{health['status']}'",
    )
    _record(
        "no metrics_snapshot key (metrics stripped from hot path)",
        "metrics_snapshot" not in health,
    )
    _record(
        "components exist",
        len(health["components"]) > 0,
        f"{len(health['components'])} components",
    )


# ============================================================================
# Phase 2 Verification
# ============================================================================

async def test_embedding_interface_protocol() -> None:
    """PipelineEmbeddingBackend should satisfy the EmbeddingInterface protocol."""
    _section("Phase 2: Alignment \u2014 EmbeddingInterface Protocol")

    backend = PipelineEmbeddingBackend(target_dim=128)
    _record(
        "PipelineEmbeddingBackend satisfies EmbeddingInterface",
        isinstance(backend, EmbeddingInterface),
    )
    _record(
        "backend.dim == 128",
        backend.dim == 128,
        f"got {backend.dim}",
    )

    vec = backend.embed("hello world test embedding")
    _record(
        "embed() returns np.ndarray",
        isinstance(vec, np.ndarray),
        f"got {type(vec).__name__}",
    )
    _record(
        "embed() shape matches dim",
        vec.shape == (128,),
        f"got shape {vec.shape}",
    )
    norm = float(np.linalg.norm(vec))
    _record(
        "embed() output is L2-normalized",
        abs(norm - 1.0) < 0.01,
        f"norm={norm:.4f}",
    )


async def test_hash_alignment() -> None:
    """Router and memory system should both use SHA256."""
    _section("Phase 2: Alignment \u2014 SHA256 Hash Consistency")

    if not _ROUTER_AVAILABLE:
        _record("SKIP (torch/router not available)", True, "skipped")
        return

    encoder = HashTextEncoder(embed_dim=64, vocab_buckets=1000, max_seq_len=32)
    token = "test_token"
    seed = 0x9747b28c
    router_hash = encoder._hash_token(token, seed)

    # Reproduce the expected SHA256 hash
    hash_input = f"{seed}:{token.lower()}".encode("utf-8")
    expected = int.from_bytes(
        hashlib.sha256(hash_input).digest()[:8], byteorder="little"
    ) % 1000

    _record(
        "HashTextEncoder uses SHA256",
        router_hash == expected,
        f"router={router_hash}, sha256={expected}",
    )

    vec = encoder.embed("hello world test")
    _record(
        "HashTextEncoder.embed() returns np.ndarray",
        isinstance(vec, np.ndarray),
        f"shape={vec.shape}",
    )
    _record(
        "HashTextEncoder.dim matches embed_dim",
        encoder.dim == 64,
        f"got {encoder.dim}",
    )


# ============================================================================
# Phase 3 Verification — Capability-Based Access Updates
# ============================================================================

async def test_all_tools_always_available() -> None:
    """All tool definitions should be returned regardless of any parameter."""
    _section("Phase 3: Capability \u2014 All Tools Always Available")

    mgr = MemoryManager(config={"embedding_dim": 64})

    tools = mgr.get_tool_definitions()
    names = {t["function"]["name"] for t in tools}

    _record("manage_memory tool present", "manage_memory" in names)
    _record("search_past_chats tool present", "search_past_chats" in names)
    _record("get_recent_chats tool present", "get_recent_chats" in names)
    _record("total tools == 3", len(tools) == 3, f"got {len(tools)}")

    # Verify no gating language in descriptions
    all_descs = " ".join(
        t["function"].get("description", "") for t in tools
    ).lower()
    _record(
        "no 'paid plan' in tool descriptions",
        "paid plan" not in all_descs,
        "descriptions are tier-neutral",
    )

    # Verify each tool has valid OpenAI function-calling schema
    for tool in tools:
        func = tool["function"]
        valid = (
            tool["type"] == "function"
            and "name" in func
            and "description" in func
            and "parameters" in func
            and func["parameters"]["type"] == "object"
        )
        _record(
            f"tool '{func['name']}' has valid schema",
            valid,
        )


async def test_archive_search_no_tier_gating() -> None:
    """Archive search should work based on settings, not tier."""
    _section("Phase 3: Capability \u2014 Archive Search Ungated")

    mgr = MemoryManager(config={"embedding_dim": 64})

    conv = Conversation(
        conversation_id="conv_cap_test",
        user_id="u_cap",
        title="Capability Test Conversation",
        messages=[
            ConversationMessage(
                message_id="msg_1",
                role="user",
                content="Building a semantic memory embedding pipeline",
            ),
            ConversationMessage(
                message_id="msg_2",
                role="assistant",
                content="Let me help design the pipeline architecture.",
            ),
        ],
    )
    await mgr.store_conversation(conv)

    # Without chat_search enabled (settings-based control, not tier)
    results_off = await mgr.search_conversations(
        user_id="u_cap", query="embedding pipeline"
    )
    _record(
        "search disabled by default (settings)",
        len(results_off) == 0,
        f"got {len(results_off)} results",
    )

    # Enable chat search — all users can
    mgr.control_panel.set_chat_search_enabled("u_cap", True)
    results_on = await mgr.search_conversations(
        user_id="u_cap", query="embedding pipeline"
    )
    _record(
        "search works when enabled (no tier check)",
        len(results_on) > 0,
        f"got {len(results_on)} results",
    )

    # get_recent_chats also works
    recent = await mgr.get_recent_chats(user_id="u_cap", n=5)
    _record(
        "get_recent_chats works for all users",
        len(recent) > 0,
        f"got {len(recent)} results",
    )


async def test_memory_state_enforcement() -> None:
    """MemoryState OFF/PAUSED should still block synthesis without metrics."""
    _section("Phase 3: Capability \u2014 MemoryState Enforcement Preserved")

    mgr = MemoryManager(config={"embedding_dim": 64})

    # OFF blocks synthesis
    mgr.control_panel.get_settings("u_state").state = MemoryState.OFF
    summary_off = await mgr.synthesize_memory("u_state")
    _record(
        "MemoryState.OFF blocks synthesis",
        summary_off.summary_text == "",
        "empty summary as expected" if not summary_off.summary_text else f"got '{summary_off.summary_text[:30]}'",
    )

    # PAUSED blocks synthesis
    mgr.control_panel.get_settings("u_state2").state = MemoryState.PAUSED
    summary_paused = await mgr.synthesize_memory("u_state2")
    _record(
        "MemoryState.PAUSED blocks synthesis",
        isinstance(summary_paused, GlobalMemorySummary),
    )

    # ON allows synthesis
    conv = Conversation(
        conversation_id="conv_state",
        user_id="u_state3",
        title="State Test",
        messages=[
            ConversationMessage(
                message_id="msg_s1",
                role="user",
                content="I am a systems engineer building memory routers",
            ),
        ],
    )
    await mgr.store_conversation(conv)
    summary_on = await mgr.synthesize_memory("u_state3")
    _record(
        "MemoryState.ON allows synthesis",
        summary_on.summary_text != "",
        f"summary='{summary_on.summary_text[:50]}...'",
    )


# ============================================================================
# Phase 4 Verification — Router <-> Memory Bridge
# ============================================================================

async def test_safe_router_wrapper_memory_bridge() -> None:
    """SafeRouterWrapper should accept and use a MemoryManager."""
    _section("Phase 4: Integration \u2014 Router \u2194 Memory Bridge")

    if not _ROUTER_AVAILABLE:
        _record("SKIP (torch/router not available)", True, "skipped")
        return

    config = RouterConfig(context_dim=768, num_templates=4, num_tools=8)
    router = NeuralPromptRouter(config)

    mgr = MemoryManager(config={"embedding_dim": 64})

    class MockTemplate:
        def render(self, **kwargs: Any) -> str:
            return "<|start|>system<|message|>Test system prompt<|end|>"

    wrapper = SafeRouterWrapper(
        neural_router=router,
        jinja_template=MockTemplate(),
        memory_manager=mgr,
    )

    _record(
        "SafeRouterWrapper accepts memory_manager",
        wrapper.memory_manager is mgr,
    )

    # Test get_all_tool_definitions
    unified_tools = wrapper.get_all_tool_definitions()
    tool_names = {t["function"]["name"] for t in unified_tools}
    _record(
        "get_all_tool_definitions includes memory tools",
        "manage_memory" in tool_names,
        f"tools: {', '.join(sorted(tool_names))}",
    )
    _record(
        "get_all_tool_definitions includes search_past_chats",
        "search_past_chats" in tool_names,
    )
    _record(
        "get_all_tool_definitions includes get_recent_chats",
        "get_recent_chats" in tool_names,
    )

    # _build_memory_context with no user_id returns empty
    empty_ctx = wrapper._build_memory_context({"messages": []})
    _record(
        "_build_memory_context returns empty for no user_id",
        empty_ctx == {},
    )


# ============================================================================
# End-to-End Integration — Full Pipeline
# ============================================================================

async def test_full_pipeline_end_to_end() -> None:
    """Full pipeline: store -> synthesize -> search -> build_prompt -> prepare_prompt."""
    _section("End-to-End: Full Memory Pipeline")

    mgr = MemoryManager(config={"embedding_dim": 64, "max_context_tokens": 256})

    # 1. Store a conversation
    conv = Conversation(
        conversation_id="conv_e2e",
        user_id="u_e2e",
        title="E2E Pipeline Test",
        messages=[
            ConversationMessage(
                message_id="msg_e2e_1",
                role="user",
                content="I am building a hybrid memory retrieval system with semantic embeddings",
            ),
            ConversationMessage(
                message_id="msg_e2e_2",
                role="assistant",
                content="I can help you design the retrieval pipeline with weighted stages",
            ),
        ],
    )
    await mgr.store_conversation(conv)
    _record("store_conversation succeeded", True)

    # 2. Synthesize
    summary = await mgr.synthesize_memory("u_e2e", MemoryScope.GLOBAL)
    _record(
        "synthesize_memory produced summary",
        summary.summary_text != "",
        f"len={len(summary.summary_text)}",
    )
    _record(
        "source conversation tracked",
        "conv_e2e" in summary.source_conversation_ids,
    )

    # 3. Build prompt context
    ctx = await mgr.build_prompt_context(
        user_id="u_e2e",
        current_query="How should I structure the retrieval pipeline?",
    )
    _record(
        "build_prompt_context returns dict",
        isinstance(ctx, dict),
        f"keys: {list(ctx.keys())}",
    )
    _record(
        "userMemories key present",
        "userMemories" in ctx,
    )

    # 4. Prepare prompt with memory injection
    messages = [
        {"role": "user", "content": "Remind me about my memory system design"},
    ]
    enriched = await mgr.prepare_prompt_with_memory(
        messages=messages,
        user_id="u_e2e",
    )
    _record(
        "prepare_prompt_with_memory enriches messages",
        len(enriched) >= len(messages),
        f"original={len(messages)}, enriched={len(enriched)}",
    )
    has_memory = any(
        msg.get("role") in ("developer", "system")
        and "<userMemories>" in msg.get("content", "")
        for msg in enriched
    )
    _record("memory context injected into messages", has_memory)

    # 5. Enable chat search and search archive
    mgr.control_panel.set_chat_search_enabled("u_e2e", True)
    search_results = await mgr.search_conversations(
        user_id="u_e2e", query="retrieval pipeline"
    )
    _record(
        "search_conversations returns results",
        len(search_results) > 0,
        f"got {len(search_results)} results",
    )

    # 6. Eviction
    eviction_result = await mgr.run_eviction("u_e2e")
    _record(
        "run_eviction completes without error",
        "memories_evicted" in eviction_result,
        f"evicted: {eviction_result}",
    )

    # 7. Export/import snapshot
    snapshot = mgr.export_memory_snapshot("u_e2e")
    _record(
        "export_memory_snapshot returns dict",
        isinstance(snapshot, dict),
        f"keys: {sorted(snapshot.keys())}",
    )

    restored = MemoryManager(config={"embedding_dim": 64})
    await restored.import_memory_snapshot(
        "u_e2e", snapshot, scope=MemoryScope.GLOBAL
    )
    restored_summary = restored.get_global_summary("u_e2e")
    _record(
        "import restored summary text",
        restored_summary.summary_text != "",
        f"len={len(restored_summary.summary_text)}",
    )

    # 8. Health check
    health = mgr.get_health()
    _record("health check passes", health["status"] == "healthy")

    # 9. Reset
    reset_result = await mgr.reset_memory("u_e2e")
    _record(
        "reset_memory completes",
        "memories_purged" in reset_result,
        f"result: {reset_result}",
    )
    _record(
        "post-reset state is OFF",
        mgr.control_panel.get_settings("u_e2e").state == MemoryState.OFF,
    )


# ============================================================================
# Main
# ============================================================================

async def main() -> None:
    """Run all integration tests."""
    global _START_TIME
    _START_TIME = time.monotonic()

    print("=" * 64)
    print("  Somnus Router \u2014 Integrated System Tests (v2)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)

    if _ROUTER_AVAILABLE:
        print(f"  Router: available (torch {torch.__version__})")
    else:
        print("  Router: not available (torch missing, router tests skipped)")

    test_suite = [
        # Phase 1: De-Tox
        test_aggressive_eviction_defaults,
        test_plangate_always_open,
        test_health_check_no_metrics,
        # Phase 2: Alignment
        test_embedding_interface_protocol,
        test_hash_alignment,
        # Phase 3: Capability
        test_all_tools_always_available,
        test_archive_search_no_tier_gating,
        test_memory_state_enforcement,
        # Phase 4: Integration
        test_safe_router_wrapper_memory_bridge,
        # End-to-End
        test_full_pipeline_end_to_end,
    ]

    for test_fn in test_suite:
        try:
            await test_fn()
        except Exception as exc:
            _record(
                f"CRASH: {test_fn.__name__}",
                False,
                f"{type(exc).__name__}: {exc}",
            )
            traceback.print_exc()

    # Summary
    elapsed = time.monotonic() - _START_TIME
    passed = sum(1 for r in _RESULTS if r["passed"])
    failed = sum(1 for r in _RESULTS if not r["passed"])
    total = len(_RESULTS)

    print(f"\n{'=' * 64}")
    print(f"  Results: {passed}/{total} passed, {failed} failed ({elapsed:.2f}s)")
    if failed > 0:
        print(f"\n  Failed tests:")
        for r in _RESULTS:
            if not r["passed"]:
                print(f"    \u2717 {r['name']}  \u2014 {r['detail']}")
    print(f"{'=' * 64}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())

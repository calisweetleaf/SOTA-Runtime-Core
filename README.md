# Somnus Router Toolkit

### Last Updated 03-24-2026 — v2 update to Drop 2 of Operation SOTA

### Neural routing and long-horizon memory infrastructure for local AI systems

![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=yellow)

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=orange)

![Drop One: RLHF](https://img.shields.io/badge/Toolkit_Drop_One-RLHF%20Pipeline-0A66C2?style=for-the-badge)

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blueviolet.svg)

![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18607898.svg)

**Quick Start** • **Module Overview** • **Architecture** • **Verification** • **License**

---

## Overview

This repository contains two standalone runtime modules for local AI systems:

- `file neural_router.py` for prompt-policy generation, routing, tool gating, template assembly, and fallback handling.
- `file memory_injection_system.py` for persistent memory synthesis, semantic retrieval, archive search, scoped isolation, explicit edits, and prompt-context assembly.

The current repository is not a monolith and it is not a packaged framework. It is a toolkit surface: two reusable Python modules plus lightweight documentation in `docs/`.

If you want composition, that path already exists in the codebase: `SafeRouterWrapper` in `file neural_router.py` can integrate with `MemoryManager` from `file memory_injection_system.py`.

## Mission

The point of this repo is practical runtime sovereignty. Routing, memory, context control, and prompt assembly should not be locked behind closed service boundaries when they can be built locally, inspected directly, and integrated into personal or production systems.

This is **Drop 2** of Operation SOTA, a five-drop open-source program delivering production-grade AI infrastructure.

**Versioning note:** This is the **v3.0.0** release. What appeared as v1 was baseline establishment; this is the first complete, timed rollout. Staged releases, not continuous deployment — the memory system is where it needs to be. The router has room to grow, but this was always the vision for Drop 2.

This v3.0.0 release reflects: unified embedding protocols, capability-based (not tier-gated) access, and validated integration (55/55 tests).

This release is now GPLv3.

## Current Repository Surface

```text
Somnus-Router/
├── neural_router.py              # Core routing module
├── memory_injection_system.py    # Memory & context injection
├── requirements.txt              # Dependencies
├── run_neural_router_smoke.py    # Smoke test runner
├── run_memory_injection_smoke.py # Smoke test runner
├── run_integrated_system_smoke.py# Integration smoke test
├── router-full-terminal-validation.md
├── test_input_preparer.py
├── verify_prompt_integration.py
├── CITATION.cff
├── BLAKE2BSUMS.txt
├── zenodo.json / zenodo_doi.txt
├── DOCTOR_CONFIG.json
├── prompt_templates/             # Template assets
├── output/                       # Generated outputs
├── reports/                      # Validation reports
├── docs/
│   ├── MODULE_GUIDE.md
│   ├── USAGE_GUIDE.md
│   ├── PROVENANCE.md
│   └── RELEASE_CHECKLIST.md
└── .metadata/ / .reports/ / .sbom/  # Provenance & SBOM
```

**Verification status:** `run_integrated_system_smoke.py` — 55/55 tests passing (router↔memory bridge, capability ungating, embedding alignment, health checks, prompt assembly).

## Module Overview

### `file neural_router.py`

Primary capabilities:

- `ContextEncoder` for message, profile, and metadata fusion.
- `SlotPredictorNetwork` for reasoning-effort and tool-gate prediction.
- `SafetyValidator` for constraints around routing behavior.
- `TemplateLibrary` for prompt asset loading and prompt assembly.
- `RouterTrainer` for multi-objective training workflows.
- `SafeRouterWrapper` for production inference with Jinja fallback and validation.

Current behavior notes:

- The router supports template loading from `prompt_templates/` when present.
- It also includes legacy fallback resolution for template files located alongside the module.
- The module includes a direct example entrypoint under `if __name__ == "__main__":`.

Run directly:

```bash
python neural_router.py
```

### `file memory_injection_system.py`

Primary capabilities:

- Scoped memory isolation via `MemoryScope` with `GLOBAL`, `PROJECT`, and `INCOGNITO` modes.
- Summary memory via `GlobalMemorySummary` and `ProjectMemorySummary`.
- User preference persistence via `ProfilePreferences` and `UserStyle`.
- Atomic memory storage and retrieval via `MemoryItem`, `MemoryStore`, and `MemoryScorer`.
- Local deterministic embeddings via `PipelineEmbeddingBackend` and `MemoryEmbedder`.
- Full archive storage and RAG-style recall via `ConversationArchive`.
- Structured synthesis via `DeterministicSynthesisBackend`, optional spaCy enrichment, and refresh orchestration.
- Explicit user memory edits, deletion propagation, controls, metrics, health checks, rate limiting, and lifecycle management.
- Prompt assembly via `PromptContextAssembler`, `ContextInjectionManager`, and `MemoryManager`.

The file is materially broader than a simple memory injector. It is a layered memory runtime with synthesis, retrieval, control-plane logic, and prompt-context assembly in one standalone module.

Run directly:

```bash
python memory_injection_system.py
```

## Smoke Test Runners

Three standalone smoke test runners validate each module:

- `file run_neural_router_smoke.py` — Validates router initialization, template loading, slot prediction, and safety checks.
- `file run_memory_injection_smoke.py` — Tests memory extraction, synthesis, embedding, retrieval, and prompt assembly.
- `file run_integrated_system_smoke.py` — End-to-end integration test suite (55/55 tests) exercising the router↔memory bridge, capability ungating, embedding alignment, health checks, and full prompt assembly pipeline.

These are validation/demonstration scripts — not production infrastructure. They prove the modules integrate correctly and provide reference usage patterns.

Run any smoke test:

```bash
python run_neural_router_smoke.py
python run_memory_injection_smoke.py
python run_integrated_system_smoke.py
```

## Memory Architecture

The current memory system is organized into five layers:

1. Summary-based persistent memory
2. Atomic memory items for semantic retrieval
3. Full conversation archive with on-demand RAG search
4. Profile preferences and styles that survive incognito boundaries
5. Control, audit, deletion propagation, and operational infrastructure

In practice this gives you:

- always-on memory summaries for prompt injection
- semantic recall over atomic memories
- archive search over stored conversations
- project isolation to prevent cross-chat bleed
- explicit user-editable memory state
- bounded prompt assembly through token budgeting

## Prompt Context Model

`file memory_injection_system.py` does not just return one blob. It assembles structured prompt context through `PromptContext`, including:

- `user_memories_xml`
- `profile_preferences_xml`
- `project_instructions_xml`
- `styles_xml`
- `semantic_memories_xml`
- `archive_references_xml`

Those sections can be combined into a single developer-context block or serialized back into message form.

## Training

The router is **trainable** — see `RouterTrainer` in `file neural_router.py` for the SFT/DPO/GRPO infrastructure. A complete training proposal using the RLHF pipeline from [Drop 1](https://github.com/calisweetleaf/Reinforcement-Learning-Full-Pipeline) is documented in `file docs/ROUTER_TRAINING_PROPOSAL.md`.

**Default stance:** The reference implementation runs as-is. Training is available if you have your own data and compute substrate, but it's not required to use the system. Custom training is not a default service offering — the code is released so you can train it yourself.

## Quick Start

### Minimal verification

```bash
python -m py_compile neural_router.py memory_injection_system.py
python neural_router.py
python memory_injection_system.py
python run_integrated_system_smoke.py
```

### Suggested virtual environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy torch jinja2
python -m py_compile neural_router.py memory_injection_system.py
python memory_injection_system.py
```

### Optional memory enrichment

If you want the optional spaCy stages to activate instead of gracefully no-oping:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## Dependency Notes

Core dependencies (see `file requirements.txt`):

- `numpy`
- `torch`
- `jinja2`

Optional:

- `spacy` (for enrichment stages)

This is source-first infrastructure — bring your own substrate (hardware, Python environment, PyTorch) and output destination.

## Verification

Verified in this workspace:

```bash
python -m py_compile neural_router.py memory_injection_system.py
```

That compile check passes against the current files in this repo.

## Documentation

Current docs in this repository:

- `file docs/MODULE_GUIDE.md` — Class-level surface map and integration notes
- `file docs/USAGE_GUIDE.md` — Usage patterns and examples
- `file docs/PROVENANCE.md` — Manifest workflow and hash snapshots
- `file docs/RELEASE_CHECKLIST.md` — Release readiness checklist

These notes capture implementation behavior, validation breadcrumbs, and repo-specific gotchas around memory scope, archive gating, XML injection, router fallback behavior, and current validation posture.

## Important Behavior Notes

- Scope discipline matters. If you want project isolation, pass a `ConversationContext(project_id=...)` instead of relying on global defaults.
- Archive recall is intentionally gated in the memory control logic.
- XML memory context is assembled as message-level context, not as a magic hidden template variable.
- The router integrates with memory through `SafeRouterWrapper` — see `file run_integrated_system_smoke.py` for validated integration patterns.

## Related Project

- `Reinforcement-Learning-Full-Pipeline`: https://github.com/calisweetleaf/Reinforcement-Learning-Full-Pipeline

## License

This repository is GPLv3.

- Full license text: https://www.gnu.org/licenses/gpl-3.0

## Citation

```bibtex
@misc{https://doi.org/10.5281/zenodo.18607898, doi = {10.5281/ZENODO.18607898}, url = {https://zenodo.org/doi/10.5281/zenodo.18607898}, author = {Rowell, Christian Trey Levi}, title = {SOTA Runtime Core: Neural Prompt Router and Dual-Method Memory System}, publisher = {Zenodo}, year = {2026}, copyright = {GPLv3 }}
```

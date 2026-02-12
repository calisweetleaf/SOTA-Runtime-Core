<div align="center">

# Somnus Router Toolkit

### Standalone SOTA Runtime Components (Drop 2)

![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=yellow)

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=orange)

![License](https://img.shields.io/badge/License-Sovereign-blueviolet?style=for-the-badge)

[![SOTA-Toolkit-Drop-One: RLHF](https://img.shields.io/badge/Toolkit_Drop_One-RLHF%20Pipeline-0A66C2?style=for-the-badge)](https://github.com/calisweetleaf/Reinforcement-Learning-Full-Pipeline)

[![License Repo: somnus-license](https://img.shields.io/badge/License_Repo-somnus--license-111111?style=for-the-badge)](https://github.com/calisweetleaf/somnus-license)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18607898.svg)](10.5281/zenodo.18607898)

<br/>

[**Quick Start**](#quick-start-windows--powershell) • [**Module Overview**](#module-overview) • [**Verification**](#end-to-end-verification-latest-run) • [**Related Projects**](#related-projects) • [**License**](#license-governance)

<br/>
</div>

---

## Overview

This repository provides production-grade implementations of three runtime tools built for SOTA-tier local AI systems. `neural_router.py` handles prompt policy generation and tool-routing decisions, `memory_injection_system.py` handles cross-chat memory extraction/retrieval/injection, and `integrated_system.py` demonstrates composition. Each file is intentionally reusable as a standalone module so teams can copy, drop-in, and integrate without adopting a full monolith.

This is positioned as installment #2 in a broader open toolkit line. The design goal is practical decentralization: high-leverage runtime infrastructure available through open research reconstruction and non-proprietary implementation, with reproducible smoke evidence and integrity artifacts.

These tools have standalone test, and an integration example. You can do whatever you like, adhering to the Anti-Exploit license governing all contents that touch this repository. The goal is to make it easy as Copy/Paste into your personal projects, integrate an already finished Production Module, and get an SaaS worthy experience free, easy, and local.

Be looking for drop 3 in this trilogy which involves even larger advancements. Drop 3 will be the biggest drop and will broadly be about "infinite" effective context with an 80-92% compression enabling pretrained open models to be able to do long horizon task, guided reasoning, and context conversion.(it is not full compression more than it is a reprentational conversion.)

---

## Mission

The exclusive control over post-training infrastructure has allowed a few organizations to artificially monopolize AI capabilities. They claim innovation while simply gating access to standard reinforcement learning techniques. THIS REPOSITORY IS GOVERNED BY THE Sovereign Anti-Exploitation Software License

This repository dismantles that barrier by open-sourcing runtime infrastructure that is usually hidden behind service boundaries. The goal is direct and practical: put state-of-the-art routing and memory tooling into local developer workflows so users are not blocked by closed deployment gates.

---

# Somnus Router Toolkit

Tool-first release focused on three standalone Python modules:

- `neural_router.py`: neural prompt routing stack with safety validation and prompt assembly.
- `memory_injection_system.py`: cross-chat memory extraction, storage, retrieval, and context injection.
- `integrated_system.py`: orchestration demo showing how router + memory can work together.

This repository is structured as reusable components, not a locked monolith. Each core module can be copied into another project and integrated independently.

Execution model:

- Standalone tool validation:
  - `python run_neural_router_smoke.py`
  - `python run_memory_injection_smoke.py`
- Composition validation:
  - `python run_integrated_system_smoke.py`
- Integration rule: run standalone smokes first, then run integrated smoke to validate orchestration.

## Scope

- Preserve core logic and behavior of existing implementations.
- Document how to run each tool standalone.
- Keep prompt-routing asset loading supported in router flow.
- Keep active prompt assets in the runtime prompt asset folder and archive legacy prompt/UI assets in the legacy archive folder.

## Module Overview

### `neural_router.py`

- Context encoder (`ContextEncoder`) for message/profile/metadata fusion.
- Slot prediction (`SlotPredictorNetwork`) for reasoning effort + tool gates/weights.
- Safety enforcement (`SafetyValidator`) for tier/tool/reasoning constraints.
- Prompt policy selection and assembly via the router library stack.
- Production wrapper (`SafeRouterWrapper`) with fallback path and prompt validation.
- Includes `RouterTrainer` for multi-objective training.

Standalone run:

```powershell
python run_neural_router_smoke.py
```

Outputs:

- `reports/neural_router_smoke.json`
- `reports/neural_router_smoke.md`
- `reports/neural_router_prompt.txt`
- `reports/neural_router_smoke_manifest.json`
- `output/neural_router_smoke.json`
- `output/neural_router_smoke.md`
- `output/neural_router_prompt.txt`
- `output/neural_router_smoke_manifest.json`

### `memory_injection_system.py`

- Structured memory model (`MemoryItem`, `MemoryType`, `ConversationContext`).
- Extraction pipeline (`MemoryExtractor`) for facts/preferences/entities/topics/tool patterns.
- Embedding + semantic retrieval (`MemoryEmbedder`, `MemoryStore`).
- Prompt injection manager (`ContextInjectionManager`) for memory block insertion.
- Orchestrator (`MemoryManager`) for end-to-end conversation turn processing.

Provenance framing for this module:

- Implemented from open research and public technical references.
- Reverse-engineered as a public-data reconstruction of frontier memory behavior.
- Public provider systems (including OpenAI) are treated as benchmark examples only.
- No proprietary source code is used in this implementation.

Standalone run:

```powershell
python memory_injection_system.py
```

Smoke run with report output:

```powershell
python run_memory_injection_smoke.py
```

Memory system docs (standalone behavior):

1. Ingestion: parses recent conversation turns and extracts candidate facts, preferences, entities, topics, and tool-usage patterns.
2. Normalization: maps extracted items to typed `MemoryItem` records with confidence, importance, and relationship metadata.
3. Embedding: generates vector representations for both memory entries and runtime queries.
4. Retrieval: combines semantic similarity, recency, and importance to select context-relevant memories.
5. Injection: writes a bounded `<memory> ... </memory>` block into the active prompt context.
6. Persistence: keeps memory state available across sessions for the same user id.

Standalone acceptance signal for this module:

- `reports/memory_injection_smoke.md` must show `Developer memory block injected: True`.
- `reports/memory_injection_smoke.json` must exist and include non-empty seeded memory summary.

### `integrated_system.py`

- `IntegratedChatSystem` for request processing flow:
  1. memory retrieval/injection
  2. neural routing (or Jinja fallback)
  3. response generation hook
  4. background memory extraction
- Additional helpers: `MemoryInsights`, `MemoryOptimizer`.
- Includes end-to-end demo entrypoint.

Standalone run:

```powershell
python integrated_system.py
```

Smoke run with report output:

```powershell
python run_integrated_system_smoke.py
```

Note: this file is a demo scaffold and may require import wiring in your target project (for example, router module path alignment).
The integrated smoke is the third proof run that combines all three components while keeping each module independently reusable.

## Quick Start (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_neural_router_smoke.py
python run_memory_injection_smoke.py
python run_integrated_system_smoke.py
```

## Dependency Profile (Unpinned)

This repository intentionally keeps dependency messaging unpinned for portability in local setups.

Core runtime packages:

- `numpy`
- `scipy`
- `scikit-learn`
- `torch`
- `transformers`
- `tqdm`
- `rich`
- `jinja2`

## End-to-End Verification (Latest Run)

Verified with `.venv` on **2026-02-09 (UTC)**:

```powershell
python run_neural_router_smoke.py
python run_memory_injection_smoke.py
python run_integrated_system_smoke.py
python -m py_compile neural_router.py memory_injection_system.py integrated_system.py run_neural_router_smoke.py run_memory_injection_smoke.py run_integrated_system_smoke.py
python tools/build_smoke_manifest.py
```

Latest pass summary:

- `neural_router_smoke`: pass (`2026-02-09T03:00:27Z`)
- `memory_injection_smoke`: pass (`2026-02-09T03:00:36Z`)
- `integrated_system_smoke`: pass (`2026-02-09T03:00:46Z`)
- `smoke_manifest`: generated (`2026-02-09T03:02:07Z`)

## Evidence Artifacts

- `reports/neural_router_smoke.json`
- `reports/neural_router_smoke.md`
- `reports/memory_injection_smoke.json`
- `reports/memory_injection_smoke.md`
- `reports/integrated_system_smoke.json`
- `reports/integrated_system_smoke.md`
- `reports/smoke_manifest.json`
- `reports/_hash_smoke/SHA256SUMS.txt`
- `reports/_hash_smoke/SHA256SUMS.json`

## Production Readiness Snapshot

- `neural_router.py` doctor scan: **6 findings** (`1 serious`, `5 minor`), no critical findings.
- `memory_injection_system.py` doctor scan: **8 findings** (`1 serious`, `7 minor`), no critical findings.
- Current serious findings are test-gap findings (expected in single-file standalone packaging) and are explicitly tracked in the reports.

Reports:

- `reports/neural_router_production_doctor.md`
- `reports/neural_router_production_doctor.json`
- `reports/memory_injection_production_doctor.md`
- `reports/memory_injection_production_doctor.json`

## SOTA Readiness Bar

This release is positioned as SOTA-grade engineering through:

- Neural routing with context fusion, slot prediction, and constraint enforcement.
- Standalone cross-session memory extraction, storage, retrieval, and injection.
- Integrated orchestrator path proving router + memory composition.
- Reproducible verification artifacts generated on every validation cycle.
- Local-first, open-research, non-proprietary implementation path.

## Related Projects

- `Reinforcement-Learning-Full-Pipeline` (installment #1): foundational behavior/alignment toolkit release.
  - `https://github.com/calisweetleaf/Reinforcement-Learning-Full-Pipeline`
- `somnus-license`: canonical license governance repository used across toolkit releases.
  - `https://github.com/calisweetleaf/somnus-license`

## License Governance

Canonical license governance lives in the dedicated license repository:

- `https://github.com/calisweetleaf/somnus-license`

For repo handoff, include:

- `LICENSE` (license text copy for this release)
- `tools/hash-index.ps1` (integrity + provenance indexing)

## Integrity Workflow (`hash-index.ps1`)

Generate a fresh integrity index before release:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File tools/hash-index.ps1 -Path . -Algorithm SHA256 -UseGitIgnore -ExportJSON -GenerateReport
```

Verify integrity later:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File tools/hash-index.ps1 -Path . -Verify -UseGitIgnore
```

Note: `hash-index.ps1` is validated with PowerShell 7 (`pwsh`).

## Current Documentation

- `README.md`: repository-level overview and commands.
- `CITATION.cff`: software citation metadata with related toolkit references.
- `docs/MODULE_GUIDE.md`: class-level surface map and integration notes.
- `docs/PROVENANCE.md`: manifest workflow and current hash snapshot.
- `docs/RELEASE_CHECKLIST.md`: release-readiness checklist.

### Provenance

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18607898.svg)](https://doi.org/10.5281/zenodo.18607898)

- **Author ORCID:** [0009-0008-6550-6316](https://orcid.org/0009-0008-6550-6316)
- **Prior Work:** [Reinforcement-Learning-Full-Pipeline](https://github.com/calisweetleaf/Reinforcement-Learning-Full-Pipeline)
- **License:** [Sovereign Anti-Exploitation Software License](https://github.com/calisweetleaf/somnus-license)

## Citation

```bibtex
@misc{https://doi.org/10.5281/zenodo.18607898,  doi = {10.5281/ZENODO.18607898},  url = {https://zenodo.org/doi/10.5281/zenodo.18607898},  author = {Rowell, Christian Trey Levi},  title = {SOTA Runtime Core: Neural Prompt Router and Dual-Method Memory System},  publisher = {Zenodo},  year = {2026},  copyright = {Somnus Anti-Exploitation License }}
```

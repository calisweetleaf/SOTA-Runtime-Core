# Provenance Notes

This document defines how provenance is captured for the standalone toolkit release.

## Scope

Protected core modules:

- `neural_router.py`
- `memory_injection_system.py`
- `integrated_system.py`

Runtime template set:

- `prompt_templates/system_prompt.md`
- `prompt_templates/jinja2_template.md`
- `prompt_templates/message_metadata.md`
- `prompt_templates/tool_list.md`

## Method

1. Generate a pre-change manifest.
2. Apply approved content/packaging updates.
3. Generate a post-change manifest.
4. Compare manifests and retain a human-readable diff report.

## Artifacts

- `reports/provenance_manifest_pre.json`
- `reports/provenance_manifest_post.json`
- `reports/provenance_diff.md`

## Commands

Generate pre manifest:

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/generate_provenance_manifest.py --label pre_bleach --output reports/provenance_manifest_pre.json
```

Generate post manifest:

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/generate_provenance_manifest.py --label post_bleach --output reports/provenance_manifest_post.json
```

Generate diff report:

```powershell
.\.venv\Scripts\python.exe -X utf8 tools/compare_provenance_manifests.py --pre reports/provenance_manifest_pre.json --post reports/provenance_manifest_post.json --output reports/provenance_diff.md
```

## Current Snapshot

- Pre manifest aggregate SHA-256: `37f23923c63066c9437413d29b48748b3aaf6b333cdef6b8c5353bb903217182`
- Post manifest aggregate SHA-256: `8c16cfdd0c15b1bafa44425c409db4552248fa6b231400b6aedda343cb295dc2`
- Diff summary: 5 changed, 2 added, 0 removed

Changed files in current diff:

- `README.md`
- `prompt_templates/jinja2_template.md`
- `prompt_templates/message_metadata.md`
- `prompt_templates/system_prompt.md`
- `run_neural_router_smoke.py`

Added files in current diff:

- `run_integrated_system_smoke.py`
- `run_memory_injection_smoke.py`

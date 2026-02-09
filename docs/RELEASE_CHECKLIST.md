# Release Checklist

## Environment

- [ ] `.venv` exists and is active.
- [ ] `pip install -r requirements.txt` completed in `.venv`.

## Layout

- [ ] Root contains core modules:
  - [ ] `neural_router.py`
  - [ ] `memory_injection_system.py`
  - [ ] `integrated_system.py`
- [ ] Active prompt assets are under `prompt_templates/`.
- [ ] Legacy prompt/UI assets are archived under `old_templates/`.
- [ ] `LICENSE` is present at repository root.

## Smoke Validation

- [ ] Router smoke:
  - Command: `python run_neural_router_smoke.py`
  - Reports: `reports/neural_router_smoke.md`, `reports/neural_router_smoke.json`
- [ ] Memory smoke:
  - Command: `python run_memory_injection_smoke.py`
  - Reports: `reports/memory_injection_smoke.md`, `reports/memory_injection_smoke.json`
- [ ] Integrated smoke:
  - Command: `python run_integrated_system_smoke.py`
  - Reports: `reports/integrated_system_smoke.md`, `reports/integrated_system_smoke.json`

## Production Doctor

- [ ] Router standalone scan generated:
  - `reports/neural_router_production_doctor.md`
  - `reports/neural_router_production_doctor.json`
- [ ] Memory standalone scan generated:
  - `reports/memory_injection_production_doctor.md`
  - `reports/memory_injection_production_doctor.json`

## Provenance

- [ ] Pre manifest generated: `reports/provenance_manifest_pre.json`
- [ ] Post manifest generated: `reports/provenance_manifest_post.json`
- [ ] Diff generated: `reports/provenance_diff.md`
- [ ] Smoke artifact manifest generated: `reports/smoke_manifest.json`
- [ ] `docs/PROVENANCE.md` updated with current aggregate hashes.

## Documentation

- [ ] `README.md` reflects current layout and smoke commands.
- [ ] `docs/MODULE_GUIDE.md` still matches module interfaces.
- [ ] `CONTEXT.md` and `MEMORY.md` updated with latest project state.

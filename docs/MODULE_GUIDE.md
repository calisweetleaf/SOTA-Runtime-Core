# Module Guide

This guide documents the current public surface of the three core modules.

## 1) `neural_router.py`

Primary classes and responsibilities:

- `RouterConfig`: model and training configuration.
- `ContextEncoder`: transforms conversation tensors into fused context embeddings.
- `SlotPredictorNetwork`: predicts reasoning/tool slots from context embeddings.
- `TemplateSelectorNetwork`: chooses template weighting from slot outputs.
- `SafetyValidator`: enforces safety and policy constraints on predicted slots/output.
- `TemplateLibrary`: loads Jinja/Markdown templates and assembles final prompt text.
- `NeuralPromptRouter`: full forward pass and trace output.
- `RouterTrainer`: training step/loss pipeline.
- `SafeRouterWrapper`: deterministic/fallback wrapper with prompt validity checks.

Core input/output contract:

- Inputs:
  - `message_embs`: `[batch, seq_len, context_dim]`
  - `user_profile`: `[batch, 128]`
  - `metadata`: `[batch, 64]`
  - `context_metadata`: dict (tier/message_count/tool-call flags, etc.)
- Outputs:
  - `prompt`: rendered system prompt string
  - `trace` (optional): slot predictions, template choice, safety violations, prompt length

Standalone validation:

- `run_neural_router_smoke.py` executes one deterministic pass and writes:
  - `reports/neural_router_smoke.json`
  - `reports/neural_router_smoke.md`
  - `reports/neural_router_prompt.txt`

## 2) `memory_injection_system.py`

Primary classes and responsibilities:

- `MemoryType`, `MemoryItem`, `ConversationContext`: core data models.
- `MemoryExtractor`: structured extraction from recent conversation turns.
- `MemoryEmbedder`: memory/query embedding generation.
- `MemoryStore`: persistence, deduplication, and semantic retrieval.
- `ContextInjectionManager`: builds/injects `<memory>` block into message list.
- `MemoryManager`: orchestration for extraction, storage, and prompt prep.

Core input/output contract:

- Inputs:
  - conversation context (`user_id`, `conversation_id`, `messages`, metadata)
  - user query/messages for retrieval
- Outputs:
  - new memory entries
  - memory-injected messages
  - memory summary stats

Standalone run:

- `python memory_injection_system.py` runs `example_usage()` demo.

## 3) `integrated_system.py`

Primary classes and responsibilities:

- `ChatRequest`, `ChatResponse`: request/response envelopes.
- `IntegratedChatSystem`: orchestrates memory retrieval, routing, generation, and memory update.
- `MemoryInsights`: timeline/search-oriented memory inspection utilities.
- `MemoryOptimizer`: consolidation/pruning/importance boosting helpers.

Core flow:

1. Inject memory into message context (if enabled).
2. Route to system prompt via neural router (or fallback Jinja path).
3. Generate assistant response (generation call is currently scaffolded).
4. Queue memory extraction for the new turn.

Standalone run:

- `python integrated_system.py` runs `complete_example()` demo path.

Implementation note:

- Integration file is intentionally demo-oriented; adapt import paths and API call hooks for production embedding.

## Standalone-First Positioning

- Router, memory, and integrated orchestrator are treated as separate tools.
- Router does not require memory module for smoke validation.
- Memory module does not require router module for extraction/retrieval/injection demo.


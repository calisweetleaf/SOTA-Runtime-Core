# USAGE GUIDE: Neural Router System

> **Production Documentation for `neural_router.py`**

This guide explains how to integrate and use the Neural Router system. The router converts Jinja2-style template logic into a learnable neural architecture that dynamically generates system prompts based on conversation context.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding the Architecture](#understanding-the-architecture)
3. [Core Classes](#core-classes)
4. [Template System](#template-system)
5. [Personality Integration](#personality-integration)
6. [Production Usage](#production-usage)
7. [Training the Router](#training-the-router)
8. [Verification & Testing](#verification--testing)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimal Usage

```python
from neural_router import RouterConfig, NeuralPromptRouter
import torch

# Initialize with defaults
config = RouterConfig()
router = NeuralPromptRouter(config)

# Create sample inputs
inputs = {
    'message_embs': torch.randn(1, 10, 768),      # Conversation embeddings
    'user_profile': torch.randn(1, 128),           # User profile vector
    'metadata': torch.randn(1, 64),                # Session metadata
    'context_metadata': {
        'user_tier': 'free',                       # User access tier
        'message_count': 5,                        # Conversation turn count
        'has_tool_calls': False                    # Whether tools are active
    }
}

# Generate prompt
prompt, trace = router(**inputs, return_trace=True)
print(prompt)
```

### Using the Template Library Directly

```python
from neural_router import RouterConfig, TemplateLibrary

config = RouterConfig()
lib = TemplateLibrary(config)

# Render the system prompt template
rendered = lib._render_jinja_template('system_prompt', {
    'reasoning_effort': 'high',
    'builtin_tools': ['browser', 'python']
})
print(rendered)
```

---

## Understanding the Architecture

The Neural Router follows a pipeline architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Neural Router Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [Message Embeddings]  [User Profile]  [Metadata]                   │
│           │                  │              │                        │
│           ▼                  ▼              ▼                        │
│       ┌───────────────────────────────────────┐                     │
│       │          ContextEncoder               │                     │
│       │  (Transformer + Profile Projection)   │                     │
│       └───────────────────────────────────────┘                     │
│                          │                                           │
│                          ▼                                           │
│       ┌───────────────────────────────────────┐                     │
│       │        SlotPredictorNetwork           │                     │
│       │  - Reasoning effort (low/med/high)    │                     │
│       │  - Tool enables (browser/python/web)  │                     │
│       │  - Tool weights (attention scores)    │                     │
│       └───────────────────────────────────────┘                     │
│                          │                                           │
│                          ▼                                           │
│       ┌───────────────────────────────────────┐                     │
│       │          SafetyValidator              │                     │
│       │  - Tier-based tool access             │                     │
│       │  - Reasoning bounds enforcement       │                     │
│       │  - Tool sparsity constraints          │                     │
│       └───────────────────────────────────────┘                     │
│                          │                                           │
│                          ▼                                           │
│       ┌───────────────────────────────────────┐                     │
│       │      TemplateSelectorNetwork          │                     │
│       │  (Selects from template library)      │                     │
│       └───────────────────────────────────────┘                     │
│                          │                                           │
│                          ▼                                           │
│       ┌───────────────────────────────────────┐                     │
│       │        TemplateLibrary.assemble()     │                     │
│       │  - Renders Jinja2 template            │                     │
│       │  - Injects personality                │                     │
│       │  - Appends tools & appendices         │                     │
│       └───────────────────────────────────────┘                     │
│                          │                                           │
│                          ▼                                           │
│                 [Generated Prompt]                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Classes

### `RouterConfig`

Configuration dataclass for the router.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context_dim` | int | 768 | Embedding dimension |
| `num_transformer_layers` | int | 4 | Encoder depth |
| `num_attention_heads` | int | 8 | Attention heads |
| `num_templates` | int | 16 | Template library size |
| `num_tools` | int | 32 | Max tool count |
| `learning_rate` | float | 1e-4 | Training LR |
| `dropout` | float | 0.1 | Dropout rate |
| `reasoning_levels` | List[str] | ['low', 'medium', 'high'] | Reasoning tiers |
| `builtin_tools` | List[str] | ['browser', 'python', 'web_search'] | Default tools |

### `NeuralPromptRouter`

Main router module. Call with:

```python
prompt, trace = router(
    message_embs=torch.Tensor,      # [batch, seq_len, dim]
    user_profile=torch.Tensor,      # [batch, 128]
    metadata=torch.Tensor,          # [batch, 64]
    context_metadata=dict,          # Runtime context
    message_mask=Optional[Tensor],  # Padding mask
    return_trace=bool               # Enable debug trace
)
```

### `SafeRouterWrapper`

Production wrapper with automatic Jinja2 fallback.

```python
from neural_router import NeuralPromptRouter, SafeRouterWrapper
from jinja2 import Template

router = NeuralPromptRouter(config)
jinja_fallback = Template("...")

wrapper = SafeRouterWrapper(router, jinja_fallback)

# Use in production
prompt, info = wrapper.route(context, use_neural=True)
if info['method'] == 'jinja2_fallback':
    print(f"Fallback triggered: {info['reason']}")
```

---

## Template System

### Template Files

The router loads templates from `prompt_templates/`:

| File | Key | Purpose |
|------|-----|---------|
| `system_prompt.jinja2` | `system_prompt` | Main system prompt |
| `offline_reasoning_agent.md` | `offline_personality` | AI personality/identity |
| `tool_manifest.jinja2` | `current_tools` | Tool definitions |
| `channel_format.jinja2` | `channel_format` | Channel routing |
| `reference_appendix.jinja2` | `reference_appendix` | Additional context |
| `tokenizer_profile.jinja2` | `tokenizer_profile` | Tokenization hints |

### Template Variables

When rendering `system_prompt.jinja2`, these variables are available:

| Variable | Source | Description |
|----------|--------|-------------|
| `current_date` | Auto | Today's date |
| `knowledge_cutoff` | Config | Training cutoff |
| `model_identity` | `offline_personality` | Full personality text |
| `reasoning_effort` | Slot Prediction | low/medium/high |
| `builtin_tools` | Slot Prediction | Active tool list |
| `memories` | Memory file | Persistent memories |

---

## Personality Integration

The router injects `offline_reasoning_agent.md` into the `{{ model_identity }}` slot of `system_prompt.jinja2`.

### How It Works

1. `TemplateLibrary` loads `offline_reasoning_agent.md` as raw text
2. During `_render_jinja_template()`, this text becomes `model_identity`
3. `system_prompt.jinja2` receives it via `{{ model_identity }}`

### Customizing Personality

Edit `prompt_templates/offline_reasoning_agent.md` to change the AI's:
- Core identity and behavioral principles
- Reasoning modes and thinking processes
- Tool usage patterns
- Communication style guidelines

### Verify Integration

```powershell
python verify_prompt_integration.py
```

This outputs the full generated prompt and saves to `output/`.

---

## Production Usage

### Recommended Pattern

```python
from neural_router import RouterConfig, NeuralPromptRouter, SafeRouterWrapper

# Initialize
config = RouterConfig(
    context_dim=768,
    num_tools=32,
    builtin_tools=['browser', 'python', 'web_search']
)
router = NeuralPromptRouter(config)

# Load your Jinja2 fallback
from neural_router import load_system_prompt_template
jinja_template = load_system_prompt_template()

# Production wrapper
wrapper = SafeRouterWrapper(router, jinja_template)

def generate_prompt(conversation_context: dict) -> str:
    """Generate system prompt for current context."""
    prompt, info = wrapper.route(
        conversation_context,
        use_neural=True,
        timeout=5.0
    )
    return prompt
```

### Context Metadata Keys

Pass these in `context_metadata`:

| Key | Type | Effect |
|-----|------|--------|
| `user_tier` | str | `'free'` disables Python tool |
| `message_count` | int | < 3 prevents "high" reasoning |
| `has_tool_calls` | bool | Forces tool enablement |
| `requires_determinism` | bool | Forces Jinja2 fallback |

---

## Training the Router

### Basic Training Loop

```python
from neural_router import RouterConfig, NeuralPromptRouter, RouterTrainer

config = RouterConfig()
router = NeuralPromptRouter(config)
trainer = RouterTrainer(router, config)

for epoch in range(100):
    for batch in dataloader:
        losses = trainer.train_step(batch)
        print(f"Loss: {losses['total']:.4f}")
    
    trainer.scheduler.step()
```

### Batch Format

```python
batch = {
    'message_embs': torch.Tensor,      # [batch, seq, dim]
    'user_profile': torch.Tensor,      # [batch, 128]
    'metadata': torch.Tensor,          # [batch, 64]
    'targets': {
        'target_reasoning': torch.LongTensor,  # [batch]
        'target_browser': torch.FloatTensor,   # [batch, 1]
        'target_python': torch.FloatTensor,    # [batch, 1]
        'target_template': torch.LongTensor    # [batch]
    }
}
```

---

## Verification & Testing

### Smoke Tests

```powershell
# Router-only validation
python run_neural_router_smoke.py

# Full system validation
python run_integrated_system_smoke.py
```

### Prompt Integration Check

```powershell
# Full output to console + file
python verify_prompt_integration.py

# Validation only (no console output)
python verify_prompt_integration.py --no-output

# Save only (no console)
python verify_prompt_integration.py --no-save --no-output
```

### Expected Outputs

- `reports/neural_router_smoke.json` - Test results
- `reports/neural_router_smoke.md` - Human-readable report
- `output/generated_system_prompt_*.txt` - Full rendered prompts

---

## Troubleshooting

### "Template render returned None"

**Cause:** Jinja2 template not found or syntax error.

**Fix:**
1. Ensure `prompt_templates/system_prompt.jinja2` exists
2. Check template syntax with: `python -c "from neural_router import TemplateLibrary; TemplateLibrary(RouterConfig())"`

### "Python executable not found"

**Cause:** SQLite history features require Python on PATH.

**Fix:** Database features are optional. The router works without them.

### "Missing required section"

**Cause:** Generated prompt lacks channel/tool definitions.

**Fix:** The `SafetyValidator` auto-injects missing sections. Check `REQUIRED_SECTIONS` in code.

### Personality not appearing

**Cause:** `offline_reasoning_agent.md` not being loaded.

**Fix:**
```python
# Debug check
lib = TemplateLibrary(RouterConfig())
print('offline_personality' in lib.raw_texts)  # Should be True
print(lib.raw_texts.get('offline_personality', '')[:100])
```

---

## File Reference

| File | Purpose |
|------|---------|
| `neural_router.py` | Main router implementation |
| `memory_injection_system.py` | Cross-chat memory system |
| `integrated_system.py` | Orchestration demo |
| `verify_prompt_integration.py` | Prompt output verification |
| `run_neural_router_smoke.py` | Router smoke test |
| `prompt_templates/` | Jinja2 templates & personality |

---

*For module-level API documentation, see `docs/MODULE_GUIDE.md`.*
*For release workflow, see `docs/RELEASE_CHECKLIST.md`.*

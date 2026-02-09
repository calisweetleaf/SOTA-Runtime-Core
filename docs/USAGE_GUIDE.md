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

## Memory Injection System

> **Production Documentation for `memory_injection_system.py`**

The memory system provides cross-chat memory extraction, storage, retrieval, and injection. It learns facts, preferences, and patterns from conversations and uses them to personalize future interactions.

---

### Memory System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Memory Injection Pipeline                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [Conversation Messages]                                             │
│           │                                                          │
│           ▼                                                          │
│  ┌───────────────────┐                                              │
│  │  MemoryExtractor  │ → LLM-based structured extraction            │
│  │  (facts/prefs/    │   - Facts: "User is a Python developer"      │
│  │   entities/topics)│   - Preferences: "Prefers concise explanations"│
│  └───────────────────┘   - Entities: "Works at CompanyX"            │
│           │              - Topics: "Interested in ML"               │
│           ▼                                                          │
│  ┌───────────────────┐                                              │
│  │  MemoryEmbedder   │ → Vector embeddings with contextual prefix   │
│  │  (3072-dim)       │                                              │
│  └───────────────────┘                                              │
│           │                                                          │
│           ▼                                                          │
│  ┌───────────────────┐                                              │
│  │   MemoryStore     │ → Semantic search + deduplication            │
│  │  (in-memory/      │                                              │
│  │   vector DB)      │                                              │
│  └───────────────────┘                                              │
│           │                                                          │
│           ▼                                                          │
│  ┌───────────────────┐                                              │
│  │ContextInjection   │ → Builds <memory> block for prompt           │
│  │    Manager        │                                              │
│  └───────────────────┘                                              │
│           │                                                          │
│           ▼                                                          │
│  [Enriched Prompt with User Context]                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Memory Types

| Type | Description | Example | Importance |
|------|-------------|---------|------------|
| `FACT` | Verifiable user information | "User is 23 years old" | 0.7 |
| `PREFERENCE` | Likes, dislikes, working styles | "Prefers detailed explanations" | 0.8 |
| `ENTITY` | People, orgs, projects | "Works on Project X" | 0.6 |
| `TOPIC` | Interest areas | "Neural networks", "ML research" | 0.5 |
| `TOOL_USAGE` | Tool usage patterns | "Frequently uses code artifacts" | 0.6 |

---

### Core Classes

#### `MemoryItem`

A single memory entry with metadata.

```python
@dataclass
class MemoryItem:
    id: str                              # Unique ID (mem_xxxx)
    user_id: str                         # Owner
    memory_type: MemoryType              # FACT/PREFERENCE/ENTITY/TOPIC
    content: str                         # The actual memory text
    embedding: Optional[np.ndarray]      # Vector representation
    created_at: datetime                 # Creation timestamp
    last_accessed: datetime              # Last retrieval time
    access_count: int                    # Retrieval count
    confidence: float                    # Extraction confidence (0-1)
    importance_score: float              # Priority (0-1)
    related_memories: Set[str]           # Linked memory IDs
    tags: Set[str]                       # User-defined tags
```

#### `MemoryManager`

The main orchestrator - handles extraction, storage, retrieval, and injection.

```python
from memory_injection_system import MemoryManager, ConversationContext

# Initialize
memory_manager = MemoryManager(config={
    'auto_extract': True,
    'extraction_interval': 5,       # Extract every 5 messages
    'max_context_tokens': 2000,     # Max tokens for memory block
    'storage_backend': 'in_memory', # Or 'pinecone', 'qdrant'
    'embedding_model': 'text-embedding-3-large'
})
```

#### `MemoryStore`

Persistent storage with semantic search.

```python
from memory_injection_system import MemoryStore, MemoryEmbedder, MemoryType

store = MemoryStore(
    storage_backend='in_memory',
    embedder=MemoryEmbedder()
)

# Store a memory
memory_id = await store.store_memory(memory_item)

# Retrieve by semantic similarity
result = await store.retrieve_memories(
    user_id='user_123',
    query='What programming languages does the user know?',
    top_k=10,
    memory_types=[MemoryType.FACT, MemoryType.PREFERENCE],
    min_importance=0.3
)

# Retrieve by type only (no semantic search)
facts = await store.retrieve_by_type(
    user_id='user_123',
    memory_type=MemoryType.FACT,
    limit=20
)
```

#### `ContextInjectionManager`

Builds and injects `<memory>` blocks into prompts.

```python
from memory_injection_system import ContextInjectionManager

injector = ContextInjectionManager(
    memory_store=store,
    max_context_tokens=2000
)

# Build context block
context_block = await injector.build_context_block(
    user_id='user_123',
    current_query='Help me with Python async'
)
# Returns:
# <memory>
# ## User Preferences
# - Prefers concise explanations
# ## User Context
# - User is a Python developer
# </memory>

# Or inject directly into messages
enriched_messages = await injector.inject_into_messages(
    messages=[{'role': 'user', 'content': 'Help me'}],
    user_id='user_123',
    current_query='Help me with Python async'
)
```

---

### Usage Patterns

#### Pattern 1: Full Pipeline

```python
import asyncio
from memory_injection_system import MemoryManager, ConversationContext

async def main():
    manager = MemoryManager(config={'auto_extract': True})
    
    # 1. Process conversation turn (extracts memories)
    conversation = ConversationContext(
        conversation_id='conv_123',
        user_id='user_456',
        messages=[
            {'role': 'user', 'content': 'I prefer TypeScript over JavaScript'}
        ]
    )
    new_memories = await manager.process_conversation_turn(
        conversation, should_extract=True
    )
    
    # 2. Later: prepare prompt with memory
    messages_with_memory = await manager.prepare_prompt_with_memory(
        messages=[{'role': 'user', 'content': 'Help me with web dev'}],
        user_id='user_456'
    )
    
    # 3. Get summary
    summary = await manager.get_user_memory_summary('user_456')
    print(summary)

asyncio.run(main())
```

#### Pattern 2: Manual Memory Management

```python
from memory_injection_system import MemoryStore, MemoryItem, MemoryType

store = MemoryStore()

# Manually add a memory
memory = MemoryItem(
    id='mem_custom_001',
    user_id='user_123',
    memory_type=MemoryType.PREFERENCE,
    content='User prefers dark mode documentation',
    importance_score=0.8
)
await store.store_memory(memory)

# Update a memory
await store.update_memory('mem_custom_001', {
    'content': 'User prefers dark mode with high contrast',
    'importance_score': 0.9
})

# Delete a memory
await store.delete_memory('mem_custom_001')
```

---

### Memory Retrieval Scoring

Memories are ranked by a combination of:

1. **Semantic Similarity** - Cosine similarity between query and memory embeddings
2. **Recency** - Recently accessed memories get priority
3. **Importance Score** - User-defined or auto-assigned priority (0-1)
4. **Access Count** - Frequently retrieved memories may be more relevant

The deduplication logic:
- `> 0.95` similarity → Merge into existing memory
- `0.8 - 0.95` similarity → Mark as related memories
- `< 0.8` similarity → Store as new memory

---

### Memory Block Format

The injected `<memory>` block follows this structure:

```xml
<memory>

## User Preferences
- Prefers detailed technical explanations
- Likes code examples in responses

## User Context
- User is a 23-year-old developer
- Works at TechCorp

## Relevant Entities
- Working on Project Phoenix

## Interest Areas: Neural networks, ML research, TypeScript

</memory>
```

---

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `auto_extract` | bool | `True` | Auto-extract memories during processing |
| `extraction_interval` | int | `5` | Messages between extractions |
| `max_context_tokens` | int | `2000` | Max tokens for memory block |
| `storage_backend` | str | `'in_memory'` | `'in_memory'`, `'pinecone'`, `'qdrant'` |
| `embedding_model` | str | `'text-embedding-3-large'` | Embedding model name |

---

### Verification

```powershell
# Smoke test
python run_memory_injection_smoke.py

# Direct execution demo
python memory_injection_system.py
```

Expected output files:
- `reports/memory_injection_smoke.json`
- `reports/memory_injection_smoke.md`

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

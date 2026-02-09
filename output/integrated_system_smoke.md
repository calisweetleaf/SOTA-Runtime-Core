# Integrated System Smoke Test
- Timestamp: 2026-02-09T03:53:34.118437Z
- Compatibility shim: {'router_core_implementation': True, 'conversation_context_bridge': True}
- Processing time (s): 0.2924532890319824
- Router trace keys: context_embedding_norm, output_issues, prompt_length, safety_violations, selected_template, slot_predictions, template_weights

## Response Preview
```
This is a mock response. In production, this would call the LLM API.
```

## System Prompt Preview
```
<|start|>system<|message|>
You are a large language model assistant.
Knowledge cutoff: 2024-06
Current date: 2026-02-08

Reasoning: low

# Tools
Builtin tools:
- `browser`
- `web_search`

// Cite information from the tool using the following format:
// `【{cursor}†L{line_start}(-L{line_end})?】`

# Valid channels: analysis, commentary, final.
Calls to these tools must go to the commentary channel: 'functions'.
<|end|>

# Tools

## browser
// Web retrieval and navigation for time-sensitive or uncer
```

## Memory Summary
```json
{
  "total_memories": 0,
  "by_type": {
    "fact": 0,
    "preference": 0,
    "entity": 0,
    "topic": 0,
    "conversation": 0,
    "tool_usage": 0
  },
  "top_topics": [],
  "last_updated": null
}
```

## Operation Log
- 2026-02-09T03:53:33.369576Z | install_router_shim | ok | 0.029ms
- 2026-02-09T03:53:33.373521Z | import_integrated_system | ok | 3.785ms
- 2026-02-09T03:53:33.373521Z | bridge_conversation_context | ok | 0.007ms
- 2026-02-09T03:53:33.764497Z | init_integrated_system | ok | 391.057ms
- 2026-02-09T03:53:34.056951Z | process_chat | ok | 292.593ms
- 2026-02-09T03:53:34.118437Z | await_background_cycle | ok | 61.348ms
- 2026-02-09T03:53:34.118437Z | summarize_memory | ok | 0.066ms
- 2026-02-09T03:53:34.118437Z | run_complete | ok | n/a
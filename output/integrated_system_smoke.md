# Integrated System Smoke Test
- Timestamp: 2026-02-09T03:18:01.465944Z
- Compatibility shim: {'router_core_implementation': True, 'conversation_context_bridge': True}
- Processing time (s): 2.6657350063323975
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

Reasoning: medium

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
// Web retrieval and navigation for time-sensitive or un
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
- 2026-02-09T03:17:57.591094Z | install_router_shim | ok | 0.062ms
- 2026-02-09T03:17:57.598505Z | import_integrated_system | ok | 6.706ms
- 2026-02-09T03:17:57.598505Z | bridge_conversation_context | ok | 0.01ms
- 2026-02-09T03:17:58.743974Z | init_integrated_system | ok | 1145.697ms
- 2026-02-09T03:18:01.409709Z | process_chat | ok | 2665.509ms
- 2026-02-09T03:18:01.465944Z | await_background_cycle | ok | 56.48ms
- 2026-02-09T03:18:01.465944Z | summarize_memory | ok | 0.075ms
- 2026-02-09T03:18:01.465944Z | run_complete | ok | n/a
# Memory Injection Smoke Test
- Timestamp: 2026-02-09T03:53:08.238724Z
- Seed: 42
- Extracted memories: 0
- Seeded memory entries: 3
- Messages returned after injection: 2
- Developer memory block injected: True

## Memory Summary
```json
{
  "total_memories": 3,
  "by_type": {
    "fact": 1,
    "preference": 1,
    "entity": 0,
    "topic": 1,
    "conversation": 0,
    "tool_usage": 0
  },
  "top_topics": [
    "Interested in production-grade validation workflows."
  ],
  "last_updated": "2026-02-08 21:53:08.225260"
}
```

## Memory Block Preview
```
<memory>

## User Preferences
- Prefers concise technical responses.

## User Context
- Building a neural prompt router toolkit.

## Interest Areas: Interested in production-grade validation workflows.

</memory>
```

## Operation Log
- 2026-02-09T03:53:08.225260Z | set_seed | ok | 1472.25ms
- 2026-02-09T03:53:08.225260Z | init_memory_manager | ok | 0.074ms
- 2026-02-09T03:53:08.225260Z | build_conversation_context | ok | 0.029ms
- 2026-02-09T03:53:08.225260Z | extract_memories | ok | 0.105ms
- 2026-02-09T03:53:08.227810Z | seed_memories | ok | 1.861ms
- 2026-02-09T03:53:08.238724Z | inject_memory_block | ok | 11.151ms
- 2026-02-09T03:53:08.238724Z | summarize_memory | ok | 0.072ms
- 2026-02-09T03:53:08.238724Z | run_complete | ok | n/a
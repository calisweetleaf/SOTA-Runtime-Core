# Neural Router Smoke Test
- Timestamp: 2026-02-09T03:17:12.896669Z
- Seed: 42
- Prompt length: 1219
- Selected template: 4
- Confidence: 0.4229309558868408

## Safety Violations

- [SOFT] reasoning_premature: High reasoning inappropriate for short conversations

## Output Issues
None

## Prompt Preview
```
<|start|>system<|message|>
You are a large language model assistant.
Knowledge cutoff: 2024-06
Current date: 2026-02-08

Reasoning: medium

# Tools
Builtin tools:
- `web_search`

// Cite information from the tool using the following format:
// `【{cursor}†L{line_start}(-L{line_end})?】`

# Valid channels: analysis, commentary, final.
Calls to these tools must go to the commentary channel: 'functions
```

## Operation Log
- 2026-02-09T03:17:11.910217Z | set_seed | ok | 18.567ms
- 2026-02-09T03:17:11.910217Z | init_config | ok | 0.033ms
- 2026-02-09T03:17:12.443624Z | init_router | ok | 532.646ms
- 2026-02-09T03:17:12.443624Z | build_batch | ok | 0.232ms
- 2026-02-09T03:17:12.896669Z | router_forward | ok | 453.617ms
- 2026-02-09T03:17:12.896669Z | run_complete | ok | n/a

## Trace (truncated)
```json
{
  "context_embedding_norm": 7.8072638511657715,
  "slot_predictions": {
    "reasoning_effort": 2,
    "tool_enables": {
      "browser": false,
      "python": false,
      "web_search": true
    },
    "tool_weights_top5": [
      [
        0,
        14,
        7,
        25,
        17
      ]
    ],
    "confidence": 0.4229309558868408
  },
  "safety_violations": [
    {
      "rule": "reasoning_premature",
      "severity": "SOFT",
      "message": "High reasoning inappropriate for short conversations",
      "field": "reasoning_effort"
    }
  ],
  "template_weights": [
    [
      0.1273777186870575,
      0.13294976949691772,
      0.11064159125089645,
      0.13052211701869965,
      0.13885873556137085,
      0.12504614889621735,
      0.11888831853866577,
      0.11571566015481949
    ]
  ],
  "selected_template": 4,
  "output_issues": [],
  "prompt_length": 1219
}
```
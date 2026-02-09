# Neural Router Smoke Test
- Timestamp: 2026-02-09T06:06:47.131835Z
- Seed: 42
- Prompt length: 1216
- Selected template: 4
- Confidence: 0.4256449341773987

## Safety Violations

- [HARD] tier_restriction: Python tool requires paid tier
- [SOFT] reasoning_premature: High reasoning inappropriate for short conversations
- [HARD] tool_dependency: Tool calls require enabled tools

## Output Issues
None

## Prompt Preview
```
<|start|>system<|message|>
You are a large language model assistant.
Knowledge cutoff: 2024-06
Current date: 2026-02-09

Reasoning: medium

# Tools
Builtin tools:
- `browser`

// Cite information from the tool using the following format:
// `【{cursor}†L{line_start}(-L{line_end})?】`

# Valid channels: analysis, commentary, final.
Calls to these tools must go to the commentary channel: 'functions'.

```

## Operation Log
- 2026-02-09T06:06:46.604917Z | set_seed | ok | 10.71ms
- 2026-02-09T06:06:46.604917Z | init_config | ok | 0.032ms
- 2026-02-09T06:06:46.923197Z | init_router | ok | 318.317ms
- 2026-02-09T06:06:46.923197Z | build_batch | ok | 0.19ms
- 2026-02-09T06:06:47.131835Z | router_forward | ok | 208.776ms
- 2026-02-09T06:06:47.131835Z | run_complete | ok | n/a

## Trace (truncated)
```json
{
  "context_embedding_norm": 7.770624160766602,
  "slot_predictions": {
    "reasoning_effort": 2,
    "tool_enables": {
      "browser": false,
      "python": true,
      "web_search": false
    },
    "tool_weights_top5": [
      [
        26,
        24,
        17,
        22,
        27
      ]
    ],
    "confidence": 0.4256449341773987
  },
  "safety_violations": [
    {
      "rule": "tier_restriction",
      "severity": "HARD",
      "message": "Python tool requires paid tier",
      "field": "tool_enables.python"
    },
    {
      "rule": "reasoning_premature",
      "severity": "SOFT",
      "message": "High reasoning inappropriate for short conversations",
      "field": "reasoning_effort"
    },
    {
      "rule": "tool_dependency",
      "severity": "HARD",
      "message": "Tool calls require enabled tools",
      "field": "tool_enables"
    }
  ],
  "template_weights": [
    [
      0.1054811179637909,
      0.11627567559480667,
      0.11044839024543762,
      0.10693895816802979,
      0.12322030961513519,
      0.10859724879264832,
      0.11376117914915085,
      0.10108709335327148,
      0.11419011652469635
    ]
  ],
  "selected_template": 4,
  "output_issues": [],
  "prompt_length": 1216
}
```
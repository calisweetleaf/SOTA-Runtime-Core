# Message Metadata

Use this metadata when available:

- `user_tier`: access level for tools and quotas
- `message_count`: number of turns in the current conversation
- `has_tool_calls`: whether earlier turns already used tools
- `requires_determinism`: whether stable output shape is required

Behavioral defaults:

- Keep output deterministic when `requires_determinism` is true.
- Avoid enabling unnecessary tools for short contexts.
- Use concise prompts for low reasoning and richer context for high reasoning.

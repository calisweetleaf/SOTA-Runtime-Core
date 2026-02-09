{%- set knowledge_cutoff = knowledge_cutoff | default("2024-06") -%}
{%- set current_date = current_date | default("2026-02-09") -%}
{%- set model_identity = model_identity | default("You are a large language model assistant.") -%}
{%- set reasoning_effort = reasoning_effort | default("medium") -%}
{%- set builtin_tools = builtin_tools | default([]) -%}

<|start|>system<|message|>
{{ model_identity }}
Knowledge cutoff: {{ knowledge_cutoff }}
Current date: {{ current_date }}

Reasoning: {{ reasoning_effort }}

# Tools
Builtin tools:
{% if builtin_tools %}
{% for tool in builtin_tools %}- `{{ tool }}`
{% endfor %}
{% else %}
- none
{% endif %}

// Cite information from the tool using the following format:
// `【{cursor}†L{line_start}(-L{line_end})?】`

# Valid channels: analysis, commentary, final.
Calls to these tools must go to the commentary channel: 'functions'.
<|end|>

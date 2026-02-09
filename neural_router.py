"""
Neural Prompt Router - Complete Production Implementation
Converts Jinja2 template logic into learnable neural architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, Template

# ============================================================================
# Configuration & Data Structures
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
PROMPT_TEMPLATE_DIR = PROJECT_ROOT / "prompt_templates"
TOOL_LIST_PATH = PROMPT_TEMPLATE_DIR / "tool_list.md"
LEGACY_TOOL_LIST_PATH = PROJECT_ROOT / "tool_list.md"
JINJA_TEMPLATE_PATH = PROMPT_TEMPLATE_DIR / "jinja2_template.md"
LEGACY_JINJA_TEMPLATE_PATH = PROJECT_ROOT / "jinja2_template.md"


def read_tool_list(path: Path = TOOL_LIST_PATH) -> List[str]:
    """Load enabled tool names from tool_list.md (one per line)."""
    if not path.exists() and LEGACY_TOOL_LIST_PATH.exists():
        path = LEGACY_TOOL_LIST_PATH
    try:
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except FileNotFoundError:
        return []


def load_system_prompt_template(path: Path = JINJA_TEMPLATE_PATH) -> Template:
    """Load the compiled system prompt template from disk."""
    if not path.exists() and LEGACY_JINJA_TEMPLATE_PATH.exists():
        path = LEGACY_JINJA_TEMPLATE_PATH
    env = Environment(
        loader=FileSystemLoader(str(path.parent)),
        autoescape=False,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    return env.get_template(path.name)


@dataclass
class RouterConfig:
    """Configuration for the neural router"""
    context_dim: int = 768
    num_transformer_layers: int = 4
    num_attention_heads: int = 8
    num_templates: int = 16
    num_tools: int = 32
    learning_rate: float = 1e-4
    dropout: float = 0.1
    weight_decay: float = 0.01
    
    # Slot configuration
    reasoning_levels: List[str] = None
    builtin_tools: List[str] = None
    
    def __post_init__(self):
        if self.reasoning_levels is None:
            self.reasoning_levels = ['low', 'medium', 'high']
        if self.builtin_tools is None:
            self.builtin_tools = ['browser', 'python', 'web_search']


class ReasoningEffort(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass
class SlotPredictions:
    """Output from slot predictor"""
    reasoning_effort: torch.Tensor  # Shape: [batch, 3]
    tool_enables: Dict[str, torch.Tensor]  # Each: [batch, 1]
    tool_weights: torch.Tensor  # Shape: [batch, num_tools]
    confidence: float = 0.0


@dataclass
class ContextFeatures:
    """Encoded conversation context"""
    message_embeddings: torch.Tensor  # [batch, seq_len, dim]
    user_profile: torch.Tensor  # [batch, profile_dim]
    metadata: Dict[str, any]


# ============================================================================
# Core Neural Components
# ============================================================================

class ContextEncoder(nn.Module):
    """
    Encodes conversation history and user profile into context embedding
    """
    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config
        
        # Transformer for message sequence encoding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.context_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.context_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_transformer_layers
        )
        
        # Profile projection
        self.profile_proj = nn.Linear(128, config.context_dim)
        
        # Metadata encoding
        self.metadata_encoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.context_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Linear(config.context_dim * 3, config.context_dim)
        
    def forward(
        self,
        message_embs: torch.Tensor,
        user_profile: torch.Tensor,
        metadata: torch.Tensor,
        message_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            message_embs: [batch, seq_len, dim]
            user_profile: [batch, profile_dim]
            metadata: [batch, metadata_dim]
            message_mask: [batch, seq_len] optional padding mask
            
        Returns:
            context_embedding: [batch, dim]
        """
        # Encode message sequence
        encoded_msgs = self.transformer(
            message_embs,
            src_key_padding_mask=message_mask
        )
        # Pool over sequence
        msg_pooled = encoded_msgs.mean(dim=1)
        
        # Project profile
        profile_encoded = self.profile_proj(user_profile)
        
        # Encode metadata
        metadata_encoded = self.metadata_encoder(metadata)
        
        # Fuse all sources
        fused = torch.cat([msg_pooled, profile_encoded, metadata_encoded], dim=-1)
        context = self.fusion(fused)
        
        return context


class SlotPredictorNetwork(nn.Module):
    """
    Predicts configuration slots from context embedding
    """
    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config
        
        # Reasoning effort head (3-way classification)
        self.reasoning_head = nn.Sequential(
            nn.Linear(config.context_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # low, medium, high
        )
        
        # Tool enable gates (binary for each tool)
        self.tool_gates = nn.ModuleDict({
            tool_name: nn.Sequential(
                nn.Linear(config.context_dim, 128),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(128, 1)
            )
            for tool_name in config.builtin_tools
        })
        
        # Tool weight attention
        self.tool_query = nn.Linear(config.context_dim, config.context_dim)
        self.tool_key = nn.Linear(config.context_dim, config.context_dim)
        self.tool_value = nn.Linear(config.context_dim, config.context_dim)
        
        self.tool_attention = nn.MultiheadAttention(
            embed_dim=config.context_dim,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        
    def forward(
        self,
        context_emb: torch.Tensor,
        tool_embeddings: torch.Tensor
    ) -> SlotPredictions:
        """
        Args:
            context_emb: [batch, dim]
            tool_embeddings: [num_tools, dim]
            
        Returns:
            SlotPredictions object
        """
        batch_size = context_emb.shape[0]
        
        # Predict reasoning effort
        reasoning_logits = self.reasoning_head(context_emb)
        reasoning_probs = F.softmax(reasoning_logits, dim=-1)
        
        # Predict tool enables
        tool_enables = {}
        for tool_name, gate in self.tool_gates.items():
            logit = gate(context_emb)
            prob = torch.sigmoid(logit)
            tool_enables[tool_name] = prob
        
        # Compute tool weights via attention
        query = self.tool_query(context_emb).unsqueeze(1)  # [batch, 1, dim]
        
        # Expand tool embeddings for batch
        tool_embs_expanded = tool_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, num_tools, dim]
        
        # Attention over tools
        attn_output, attn_weights = self.tool_attention(
            query=query,
            key=tool_embs_expanded,
            value=tool_embs_expanded
        )
        tool_weights = attn_weights.squeeze(1)  # [batch, num_tools]
        
        # Compute confidence (mean max probability across slots)
        confidence = (
            reasoning_probs.max(dim=-1)[0].mean() +
            torch.stack(list(tool_enables.values())).mean()
        ) / 2
        
        return SlotPredictions(
            reasoning_effort=reasoning_probs,
            tool_enables=tool_enables,
            tool_weights=tool_weights,
            confidence=confidence.item()
        )


class TemplateSelectorNetwork(nn.Module):
    """
    Selects template from library based on slot predictions
    """
    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config
        
        # Compute total slot dimension
        slot_dim = (
            3 +  # reasoning effort
            len(config.builtin_tools) +  # tool enables
            config.num_tools  # tool weights
        )
        
        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(slot_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_templates)
        )
        
    def flatten_slots(self, slot_preds: SlotPredictions) -> torch.Tensor:
        """Flatten slot predictions into single vector"""
        components = [slot_preds.reasoning_effort]
        
        # Add tool enables
        for tool_name in self.config.builtin_tools:
            components.append(slot_preds.tool_enables[tool_name])
        
        # Add tool weights
        components.append(slot_preds.tool_weights)
        
        return torch.cat(components, dim=-1)
    
    def forward(self, slot_preds: SlotPredictions) -> torch.Tensor:
        """
        Args:
            slot_preds: SlotPredictions object
            
        Returns:
            template_weights: [batch, num_templates]
        """
        slot_vector = self.flatten_slots(slot_preds)
        logits = self.gate_network(slot_vector)
        weights = F.softmax(logits, dim=-1)
        return weights


# ============================================================================
# Safety & Validation
# ============================================================================

class SafetyValidator:
    """
    Non-differentiable constraint enforcement
    """
    def __init__(self, config: RouterConfig):
        self.config = config
        self.violation_log = []
        
        # Define immutable sections that must always be present
        self.immutable_sections = list(REQUIRED_SECTIONS.values())
        
    def validate_slots(
        self,
        slot_preds: SlotPredictions,
        context_metadata: Dict
    ) -> Tuple[SlotPredictions, List[Dict]]:
        """
        Enforce hard constraints on slot predictions
        
        Returns:
            (corrected_slots, violations)
        """
        violations = []
        
        # Rule 1: Tier-based tool access
        user_tier = context_metadata.get('user_tier', 'free')
        if user_tier == 'free':
            # Free tier: no python, limited browser
            python_enable = slot_preds.tool_enables.get(
                'python',
                torch.zeros_like(slot_preds.tool_weights[:, :1])
            )
            if (python_enable > 0.5).any().item():
                violations.append({
                    'rule': 'tier_restriction',
                    'severity': 'HARD',
                    'message': 'Python tool requires paid tier',
                    'field': 'tool_enables.python'
                })
                slot_preds.tool_enables['python'] = torch.zeros_like(python_enable)
        
        # Rule 2: Reasoning effort bounds
        msg_count = context_metadata.get('message_count', 0)
        reasoning_idx = slot_preds.reasoning_effort.argmax(dim=-1)
        
        if msg_count < 3:
            high_mask = reasoning_idx == ReasoningEffort.HIGH.value
        else:
            high_mask = torch.zeros_like(reasoning_idx, dtype=torch.bool)
        
        if high_mask.any().item():
            violations.append({
                'rule': 'reasoning_premature',
                'severity': 'SOFT',
                'message': 'High reasoning inappropriate for short conversations',
                'field': 'reasoning_effort'
            })
            # Force to medium
            new_reasoning = slot_preds.reasoning_effort.clone()
            new_reasoning[high_mask] = 0.0
            new_reasoning[high_mask, ReasoningEffort.MEDIUM.value] = 1.0
            slot_preds.reasoning_effort = new_reasoning
        
        # Rule 3: Tool sparsity constraint
        tool_weight_sum = slot_preds.tool_weights.sum(dim=-1)
        over_limit = tool_weight_sum > 3.0
        if over_limit.any().item():
            violations.append({
                'rule': 'tool_sparsity',
                'severity': 'SOFT',
                'message': 'Too many tools selected',
                'field': 'tool_weights'
            })
            # Normalize
            scaled_weights = slot_preds.tool_weights.clone()
            scaled_weights[over_limit] = (
                slot_preds.tool_weights[over_limit] /
                tool_weight_sum[over_limit].unsqueeze(-1) * 2.0
            )
            slot_preds.tool_weights = scaled_weights
        
        # Rule 4: Tool dependency check
        has_tool_calls = context_metadata.get('has_tool_calls', False)
        if has_tool_calls:
            # Ensure at least one tool is enabled
            any_tool_enabled = any(
                (enable > 0.5).any().item()
                for enable in slot_preds.tool_enables.values()
            )
            if not any_tool_enabled:
                violations.append({
                    'rule': 'tool_dependency',
                    'severity': 'HARD',
                    'message': 'Tool calls require enabled tools',
                    'field': 'tool_enables'
                })
                # Enable browser as default
                browser_gate = slot_preds.tool_enables.get(
                    'browser',
                    torch.zeros_like(slot_preds.tool_weights[:, :1])
                )
                slot_preds.tool_enables['browser'] = torch.ones_like(browser_gate)
        
        self.violation_log.extend(violations)
        return slot_preds, violations
    
    def validate_output(self, generated_prompt: str) -> Tuple[str, List[str]]:
        """
        Ensure generated prompt has all required sections
        """
        issues = []
        
        for section in self.immutable_sections:
            if section not in generated_prompt:
                issues.append(f"Missing required section: {section}")
        
        return generated_prompt, issues


# ============================================================================
# Template Management
# ============================================================================

# Template files available in workspace (priority order)
TEMPLATE_FILES = {
    'og_jinja2': ['og_jinja2_template.jinja2', 'og_jinja2_template.jinja'],
    'system_prompt': ['system_prompt.jinja2', 'system_prompt.md'],
    'current_tools': ['tool_manifest.jinja2'],
    'channel_format': ['channel_format.jinja2', 'channel_format.txt'],
    'reference_appendix': ['reference_appendix.jinja2'],
    'tokenizer_profile': ['tokenizer_profile.jinja2'],
    'message_metadata': ['message_metadata.md'],
    'jinja_md': ['jinja2_template.md'],
    'offline_personality': ['offline_reasoning_agent.md'],
}

# Required sections for valid prompts (from safety validator)
REQUIRED_SECTIONS = {
    'channel_definitions': '# Valid channels: analysis, commentary, final.',
    'tool_call_format': "Calls to these tools must go to the commentary channel: 'functions'.",
    'citation_rules': '// Cite information from the tool using the following format:',
}


class TemplateLibrary:
    """
    Manages template variants and assembly - loads from .jinja2 files
    
    Template Selection Mapping:
    - Template 0: Minimal (low reasoning, no tools)
    - Template 1: Standard (medium reasoning, browser)
    - Template 2: Code-focused (high reasoning, python)
    - Template 3: Research (medium reasoning, browser + web)
    - Template 4: Advanced (high reasoning, all tools)
    """
    def __init__(self, config: RouterConfig):
        self.config = config
        self.repo_root = Path(__file__).parent
        self.template_root = self.repo_root / "prompt_templates"
        if not self.template_root.exists():
            self.template_root = self.repo_root
        self.load_log_path = (
            self.repo_root
            / "reports"
            / f"template_load_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.memories_text = self._load_memories()
        self.reference_appendix_spec = self._load_reference_appendix_spec()
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_root)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        
        # Load available templates
        self.loaded_templates: Dict[str, Template] = {}
        self.raw_templates: Dict[str, str] = {}
        self.raw_texts: Dict[str, str] = {}
        self._load_jinja_templates()
        
        # Build template index mapping template_id -> rendering config
        self.templates = self._initialize_templates()
        self.template_key_by_id = {tid: cfg["template_key"] for tid, cfg in self.templates.items()}
        self.template_id_by_key = {cfg["template_key"]: tid for tid, cfg in self.templates.items()}
        
    def _load_memories(self) -> str:
        """Load persistent memory blob for prompt conditioning."""
        mem_path = self.repo_root / "input_templates" / "reformatted_plaintext_memories.txt"
        try:
            return mem_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        except Exception as e:
            self._log_template(f"memory_load_failed: {e}")
            return ""
    
    def _load_reference_appendix_spec(self) -> str:
        """Load neutral reference appendix text (raw, non-executable)."""
        spec_path = self.template_root / "reference_appendix.jinja2"
        try:
            return spec_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        except Exception as e:
            self._log_template(f"reference_appendix_load_failed: {e}")
            return ""
        
    def _load_jinja_templates(self):
        """Load all available .jinja2 templates from workspace"""
        self._log_template("== Template load start ==")
        for name, filenames in TEMPLATE_FILES.items():
            if isinstance(filenames, str):
                filenames = [filenames]
            
            loaded = False
            for filename in filenames:
                filepath = self.template_root / filename
                if filepath.exists():
                    try:
                        raw_text = filepath.read_text(encoding="utf-8")
                        self.raw_texts[name] = raw_text
                    except Exception as e:
                        self._log_template(f"raw_read_failed {filename}: {e}")
                    try:
                        self.loaded_templates[name] = self.jinja_env.get_template(filename)
                        msg = f"loaded {filename}"
                        print(f"[templates] {msg}")
                        self._log_template(msg)
                        loaded = True
                        break
                    except Exception as e:
                        msg = f"failed {filename}: {e}"
                        print(f"[templates] {msg}")
                        self._log_template(msg)
            if not loaded:
                msg = f"missing {', '.join(filenames)}"
                print(f"[templates] {msg}")
                self._log_template(msg)
        self._log_template("== Template load end ==")

    def _log_template(self, message: str) -> None:
        """Append a template load log entry."""
        try:
            self.load_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.load_log_path.open("a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] {message}\n")
        except Exception:
            # Do not raise from logging; best-effort only.
            pass
        
    def _initialize_templates(self) -> Dict[int, Dict]:
        """
        Initialize template configurations.
        Each config specifies which jinja2 template + render params to use.
        """
        templates = {}
        tid = 0

        def add_template(template_key, fallback_builder, params):
            nonlocal tid
            templates[tid] = {
                'template_key': template_key,
                'fallback_builder': fallback_builder,
                'params': params,
            }
            tid += 1

        # Prioritized known templates if available (no restriction list)
        if 'system_prompt' in self.loaded_templates:
            add_template(
                'system_prompt',
                self._build_standard_template,
                {'reasoning_effort': 'medium', 'builtin_tools': ['browser'], 'include_channels': True, 'include_tools': True},
            )

        if 'channel_format' in self.loaded_templates:
            add_template(
                'channel_format',
                self._build_minimal_template,
                {'reasoning_effort': 'low', 'builtin_tools': [], 'include_channels': True, 'include_tools': False},
            )

        if 'tokenizer_profile' in self.loaded_templates:
            add_template(
                'tokenizer_profile',
                self._build_research_template,
                {'reasoning_effort': 'medium', 'builtin_tools': ['browser'], 'include_channels': True, 'include_tools': True},
            )

        # Any other successfully loaded templates (including reference_appendix, og_jinja2, etc.)
        for template_key in self.loaded_templates:
            if any(cfg['template_key'] == template_key for cfg in templates.values()):
                continue
            add_template(
                template_key,
                self._build_standard_template,
                {'reasoning_effort': 'medium', 'builtin_tools': ['browser'], 'include_channels': True, 'include_tools': True},
            )

        # If nothing loaded, fall back to builtin builders
        if not templates:
            add_template(
                'fallback_minimal',
                self._build_minimal_template,
                {'reasoning_effort': 'low', 'builtin_tools': [], 'include_channels': True, 'include_tools': False},
            )
            add_template(
                'fallback_standard',
                self._build_standard_template,
                {'reasoning_effort': 'medium', 'builtin_tools': ['browser'], 'include_channels': True, 'include_tools': True},
            )

        return templates
    
    def _render_jinja_template(self, template_key: str, params: Dict) -> Optional[str]:
        """Attempt to render a jinja2 template with given params"""
        if template_key in self.loaded_templates:
            try:
                template = self.loaded_templates[template_key]
                
                # Determine model identity: use offline personality if available
                model_identity = "You are a large language model assistant."
                if 'offline_personality' in self.raw_texts:
                    model_identity = self.raw_texts['offline_personality']

                # Add common params
                render_params = {
                    'current_date': datetime.now().strftime("%Y-%m-%d"),
                    'knowledge_cutoff': '2024-06',
                    'strftime_now': lambda fmt: datetime.now().strftime(fmt),
                    'add_generation_prompt': False,
                    'messages': [],
                    'tools': None,
                    'memories': self.memories_text,
                    'memory_blob': self.memories_text,
                    'reference_appendix_spec': self.reference_appendix_spec,
                    'model_identity': model_identity,
                    **params
                }
                
                rendered = template.render(**render_params)
                return rendered
            except Exception as e:
                print(f"Template render error ({template_key}): {e}")
                self._log_template(f"render error {template_key}: {e}")
                return None
        if template_key in self.raw_templates:
            # Return raw text when Jinja parse failed but file exists
            return self.raw_templates[template_key]
        return None
    
    def _ensure_required_sections(self, prompt: str) -> str:
        """Inject missing required sections into prompt"""
        for section_name, section_content in REQUIRED_SECTIONS.items():
            if section_content not in prompt:
                # Find insertion point (before <|end|> or at end)
                if '<|end|>' in prompt:
                    prompt = prompt.replace(
                        '<|end|>',
                        f'\n{section_content}\n<|end|>',
                        1  # Only first occurrence
                    )
                else:
                    prompt += f'\n{section_content}\n'
        return prompt
    
    # Fallback builders (used when jinja2 templates not available)
    def _build_minimal_template(self) -> str:
        return """<|start|>system<|message|>
You are a large language model assistant.
Knowledge cutoff: 2024-06
Current date: {current_date}

Reasoning: low

# Valid channels: analysis, commentary, final.
<|end|>"""
    
    def _build_standard_template(self) -> str:
        return """<|start|>system<|message|>
You are a large language model assistant.
Knowledge cutoff: 2024-06
Current date: {current_date}

Reasoning: medium

# Tools

## browser

// Tool for browsing.
// Cite information from the tool using the following format:
// `【{{cursor}}†L{{line_start}}(-L{{line_end}})?】`
namespace browser {{
type search = (_: {{query: string, topn?: number}}) => any;
type open = (_: {{id?: number | string}}) => any;
}} // namespace browser

# Valid channels: analysis, commentary, final.
Calls to these tools must go to the commentary channel: 'functions'.
<|end|>"""
    
    def _build_code_focused_template(self) -> str:
        return """<|start|>system<|message|>
You are a large language model assistant.
Knowledge cutoff: 2024-06
Current date: {current_date}

Reasoning: high

# Tools

## python

Use this tool to execute Python code in your chain of thought.
// Cite information from the tool using the following format:
// Reference outputs in your analysis.

IMPORTANT: Calls to python MUST go in the analysis channel.

# Valid channels: analysis, commentary, final.
Calls to these tools must go to the commentary channel: 'functions'.
<|end|>"""
    
    def _build_research_template(self) -> str:
        return """<|start|>system<|message|>
You are a large language model assistant.
Knowledge cutoff: 2024-06
Current date: {current_date}

Reasoning: medium

# Tools

## browser

// Tool for browsing.
// Cite information from the tool using the following format:
// `【{{cursor}}†L{{line_start}}(-L{{line_end}})?】`
namespace browser {{
type search = (_: {{query: string, topn?: number}}) => any;
type open = (_: {{id?: number | string}}) => any;
}} // namespace browser

# Valid channels: analysis, commentary, final.
Calls to these tools must go to the commentary channel: 'functions'.
<|end|>"""
    
    def _build_advanced_template(self) -> str:
        return """<|start|>system<|message|>
You are a large language model assistant.
Knowledge cutoff: 2024-06
Current date: {current_date}

Reasoning: high

# Tools

## browser

// Tool for browsing.
// Cite information from the tool using the following format:
// `【{{cursor}}†L{{line_start}}(-L{{line_end}})?】`
namespace browser {{
type search = (_: {{query: string, topn?: number}}) => any;
type open = (_: {{id?: number | string}}) => any;
}} // namespace browser

## python

Use this tool to execute Python code in your chain of thought.
IMPORTANT: Calls to python MUST go in the analysis channel.

# Valid channels: analysis, commentary, final.
Calls to these tools must go to the commentary channel: 'functions'.
<|end|>"""
    
    def assemble(
        self,
        template_weights: torch.Tensor,
        slot_preds: SlotPredictions,
        context_metadata: Dict
    ) -> str:
        """
        Assemble final prompt from template selection.
        FIXED: Now properly renders appended sub-templates (reference appendix).
        """
        # 1. Hard selection for inference
        template_id = template_weights.argmax(dim=-1).item()

        # Prefer system_prompt if available
        if 'system_prompt' in self.template_id_by_key:
            template_id = self.template_id_by_key['system_prompt']

        template_config = self.templates.get(template_id, next(iter(self.templates.values())))
        
        # 2. Override params based on slot predictions
        render_params = template_config['params'].copy()
        render_params.setdefault('memories', self.memories_text)
        render_params.setdefault('memory_blob', self.memories_text)
        render_params.setdefault('reference_appendix_spec', self.reference_appendix_spec)
        
        # Map reasoning effort from slot prediction
        reasoning_idx = slot_preds.reasoning_effort.argmax(dim=-1).item()
        reasoning_map = {0: 'low', 1: 'medium', 2: 'high'}
        render_params['reasoning_effort'] = reasoning_map.get(reasoning_idx, 'medium')
        
        # Build builtin_tools from slot enables
        active_tools = []
        for tool_name, enable in slot_preds.tool_enables.items():
            if (enable > 0.5).any().item():
                active_tools.append(tool_name)
        if active_tools:
            render_params['builtin_tools'] = active_tools
        
        # Add context metadata
        render_params['model_identity'] = context_metadata.get(
            'model_identity',
            'You are a large language model assistant.'
        )
        
        # 3. Render the MAIN template first
        prompt = self._render_jinja_template(
            template_config['template_key'],
            render_params
        )
        
        # Fall back to builder if jinja2 failed
        if prompt is None:
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")
            prompt = template_config['fallback_builder']().format(current_date=current_date)

        # 4. Handle Appended Templates (The Critical Fix)
        if template_config['template_key'] == 'system_prompt':

            # A. Handle Tools
            tool_spec = self.raw_texts.get('current_tools', '')
            if tool_spec:
                # Render the tool spec in case it has variables
                tool_tmpl = self.jinja_env.from_string(tool_spec)
                rendered_tools = tool_tmpl.render(**render_params)
                prompt += "\n\n# Tools\n\n" + rendered_tools

            # B. Handle reference appendix
            # Get the raw text first
            appendix_raw = self.raw_texts.get('reference_appendix', '')
            
            if appendix_raw:
                # RENDER it to process {{ current_date }} and {{ model_identity }}
                appendix_tmpl = self.jinja_env.from_string(appendix_raw)
                rendered_appendix = appendix_tmpl.render(**render_params)
                
                # STRIP invalid headers that cause "Inception" errors
                # We remove the <|start|> tags if they exist in the appendix
                rendered_appendix = rendered_appendix.replace("<|start|>system<|message|>", "").replace("<|end|>", "")
                
                prompt += "\n\n# Reference Appendix\n\n" + rendered_appendix.strip()
        
        # 5. Ensure required sections are present
        prompt = self._ensure_required_sections(prompt)
        
        return prompt


# ============================================================================
# Main Router Class
# ============================================================================

class NeuralPromptRouter(nn.Module):
    """
    Complete neural prompt router system
    """
    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config
        
        # Templates first so we can align num_templates with actual library size
        self.template_library = TemplateLibrary(config)
        self.config.num_templates = len(self.template_library.templates)
        
        # Core components
        self.context_encoder = ContextEncoder(self.config)
        self.slot_predictor = SlotPredictorNetwork(self.config)
        self.template_selector = TemplateSelectorNetwork(self.config)
        
        # Non-trainable components
        self.safety_validator = SafetyValidator(self.config)
        
        # Tool embeddings (learnable)
        self.tool_embeddings = nn.Parameter(
            torch.randn(config.num_tools, config.context_dim)
        )
        
    def forward(
        self,
        message_embs: torch.Tensor,
        user_profile: torch.Tensor,
        metadata: torch.Tensor,
        context_metadata: Dict,
        message_mask: Optional[torch.Tensor] = None,
        return_trace: bool = False
    ) -> Tuple[str, Optional[Dict]]:
        """
        Complete forward pass
        
        Args:
            message_embs: [batch, seq_len, dim]
            user_profile: [batch, profile_dim]
            metadata: [batch, metadata_dim]
            context_metadata: Dict with user_tier, message_count, etc.
            message_mask: Optional padding mask
            return_trace: If True, return execution trace
            
        Returns:
            (generated_prompt, trace_dict or None)
        """
        trace = {} if return_trace else None
        
        # 1. Encode context
        context_emb = self.context_encoder(
            message_embs, user_profile, metadata, message_mask
        )
        
        if return_trace:
            trace['context_embedding_norm'] = context_emb.norm(dim=-1).item()
        
        # 2. Predict slots
        slot_preds = self.slot_predictor(context_emb, self.tool_embeddings)
        
        if return_trace:
            trace['slot_predictions'] = {
                'reasoning_effort': slot_preds.reasoning_effort.argmax(dim=-1).item(),
                'tool_enables': {
                    k: (v.item() > 0.5)
                    for k, v in slot_preds.tool_enables.items()
                },
                'tool_weights_top5': slot_preds.tool_weights.topk(5).indices.tolist(),
                'confidence': slot_preds.confidence
            }
        
        # 3. Validate & enforce constraints
        slot_preds, violations = self.safety_validator.validate_slots(
            slot_preds, context_metadata
        )
        
        if return_trace:
            trace['safety_violations'] = violations
        
        # 4. Select template
        template_weights = self.template_selector(slot_preds)
        
        if return_trace:
            trace['template_weights'] = template_weights.tolist()
            trace['selected_template'] = template_weights.argmax(dim=-1).item()
        
        # 5. Assemble final prompt
        prompt = self.template_library.assemble(
            template_weights, slot_preds, context_metadata
        )
        
        # 6. Final safety check
        prompt, issues = self.safety_validator.validate_output(prompt)
        
        if return_trace:
            trace['output_issues'] = issues
            trace['prompt_length'] = len(prompt)
        
        return prompt, trace


# ============================================================================
# Training Infrastructure
# ============================================================================

class RouterTrainer:
    """
    Training pipeline for prompt router
    """
    def __init__(self, router: NeuralPromptRouter, config: RouterConfig):
        self.router = router
        self.config = config
        
        self.optimizer = torch.optim.AdamW(
            router.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100  # num_epochs
        )
        
    def compute_loss(
        self,
        slot_preds: SlotPredictions,
        template_weights: torch.Tensor,
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-objective loss computation
        """
        losses = {}
        
        # Loss 1: Reasoning effort classification
        if 'target_reasoning' in targets:
            losses['reasoning'] = F.cross_entropy(
                slot_preds.reasoning_effort,
                targets['target_reasoning']
            )
        
        # Loss 2: Tool enable prediction
        tool_loss = 0
        for tool_name in self.config.builtin_tools:
            if f'target_{tool_name}' in targets:
                tool_loss += F.binary_cross_entropy(
                    slot_preds.tool_enables[tool_name],
                    targets[f'target_{tool_name}']
                )
        losses['tools'] = tool_loss
        
        # Loss 3: Template selection
        if 'target_template' in targets:
            losses['template'] = F.cross_entropy(
                template_weights,
                targets['target_template']
            )
        
        # Loss 4: Sparsity regularization
        losses['sparsity'] = 0.01 * slot_preds.tool_weights.abs().sum()
        
        # Combined loss
        total_loss = (
            losses.get('reasoning', 0) +
            losses.get('tools', 0) +
            0.5 * losses.get('template', 0) +
            losses['sparsity']
        )
        losses['total'] = total_loss
        
        return losses
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Forward pass
        context_emb = self.router.context_encoder(
            batch['message_embs'],
            batch['user_profile'],
            batch['metadata']
        )
        
        slot_preds = self.router.slot_predictor(
            context_emb,
            self.router.tool_embeddings
        )
        
        template_weights = self.router.template_selector(slot_preds)
        
        # Compute loss
        losses = self.compute_loss(
            slot_preds,
            template_weights,
            batch['targets']
        )
        
        # Backward pass
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.router.parameters(), 1.0)
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}


# ============================================================================
# Inference Wrapper with Fallback
# ============================================================================

class SafeRouterWrapper:
    """
    Production wrapper with fallback to Jinja2
    """
    def __init__(
        self,
        neural_router: NeuralPromptRouter,
        jinja_template: any  # Your existing Jinja2 template
    ):
        self.neural_router = neural_router
        self.jinja_template = jinja_template
        self.failure_count = 0
        
    def route(
        self,
        context: Dict,
        use_neural: bool = True,
        timeout: float = 5.0
    ) -> Tuple[str, Dict]:
        """
        Route with automatic fallback
        """
        # Check if deterministic mode required
        if context.get('requires_determinism', False):
            return self._fallback_route(context, reason='determinism_required')
        
        if not use_neural or self.failure_count > 10:
            return self._fallback_route(context, reason='disabled_or_failures')
        
        try:
            # Prepare inputs
            inputs = self._prepare_inputs(context)
            
            # Neural routing
            prompt, trace = self.neural_router(
                **inputs,
                return_trace=True
            )
            
            # Validate output
            if self._validate_prompt(prompt):
                self.failure_count = max(0, self.failure_count - 1)
                return prompt, {'method': 'neural', 'trace': trace}
            else:
                return self._fallback_route(context, reason='validation_failed')
                
        except Exception as e:
            self.failure_count += 1
            return self._fallback_route(context, reason=f'exception: {str(e)}')
    
    def _fallback_route(self, context: Dict, reason: str) -> Tuple[str, Dict]:
        """Fallback to Jinja2 template"""
        prompt = self.jinja_template.render(**context)
        return prompt, {'method': 'jinja2_fallback', 'reason': reason}
    
    def _prepare_inputs(self, context: Dict) -> Dict:
        """Convert context dict to model inputs"""
        # This would encode messages, profile, etc.
        # Placeholder implementation
        return {
            'message_embs': torch.randn(1, 10, 768),
            'user_profile': torch.randn(1, 128),
            'metadata': torch.randn(1, 64),
            'context_metadata': context
        }
    
    def _validate_prompt(self, prompt: str) -> bool:
        """Validate generated prompt"""
        required = ['<|start|>system<|message|>', '<|end|>']
        return all(section in prompt for section in required)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    # Initialize
    config = RouterConfig(
        context_dim=768,
        num_templates=16,
        num_tools=32
    )
    
    router = NeuralPromptRouter(config)
    
    # Example inference
    batch = {
        'message_embs': torch.randn(1, 10, 768),
        'user_profile': torch.randn(1, 128),
        'metadata': torch.randn(1, 64),
        'context_metadata': {
            'user_tier': 'free',
            'message_count': 5,
            'has_tool_calls': False
        }
    }
    
    prompt, trace = router(**batch, return_trace=True)
    
    print("Generated Prompt:")
    print(prompt)
    print("\nExecution Trace:")
    print(json.dumps(trace, indent=2))

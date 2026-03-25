"""
Memory & Context Injection System

Hybrid memory system with summary-based persistence, semantic retrieval,
scoped isolation, explicit edits, background synthesis, and on-demand
RAG archive search.

Architecture:
    Layer 1 — Summary-based persistent memory (always-on, XML injection)
    Layer 2 — Atomic memory items (semantic retrieval on demand)
    Layer 3 — Full conversation archive with RAG search (always available)
    Layer 4 — Profile preferences and styles (survives incognito)
    Layer 5 — Control, audit, deletion propagation, enterprise infra
"""

from typing import Dict, List, Optional, Tuple, Set, Union, Any, Callable, Protocol, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
import json
import numpy as np
from collections import defaultdict
import contextvars
import hashlib
import logging
import re
import uuid
import asyncio
import math
import heapq
import time
import secrets
import hmac as hmac_mod
import base64

logger = logging.getLogger(__name__)


# ============================================================================
# Shared Protocols
# ============================================================================


@runtime_checkable
class EmbeddingInterface(Protocol):
    """Shared protocol for embedding backends.
    
    Both the router's HashTextEncoder and memory's PipelineEmbeddingBackend
    satisfy this protocol, ensuring consistent embedding contracts across
    the system.
    """

    def embed(self, text: str, **kwargs: Any) -> np.ndarray:
        """Produce a fixed-size embedding vector from text."""
        ...

    @property
    def dim(self) -> int:
        """Return the dimensionality of the produced embeddings."""
        ...


# ============================================================================
# Enums and Constants
# ============================================================================


class MemoryScope(Enum):
    """Memory scope boundaries — determines isolation and persistence rules."""

    GLOBAL = "global"  # Non-project standalone chats
    PROJECT = "project"  # Per-project isolated memory
    INCOGNITO = "incognito"  # Ephemeral, excluded from all memory


class MemoryState(Enum):
    """Memory system state machine."""

    OFF = "off"  # No synthesis, no injection
    ON = "on"  # Full operation
    PAUSED = "paused"  # Existing memory preserved, no new synthesis


class UserTier(Enum):
    """User subscription tier - gates certain features."""

    FREE = "free"
    PRO = "pro"
    MAX = "max"
    TEAM = "team"
    ENTERPRISE = "enterprise"


class MemoryType(Enum):
    """Legacy atomic memory types - used for semantic retrieval layer."""

    FACT = "fact"
    PREFERENCE = "preference"
    ENTITY = "entity"
    TOPIC = "topic"
    CONVERSATION = "conversation"
    TOOL_USAGE = "tool_usage"


class MemoryEvent(Enum):
    """Observable memory system lifecycle events for enterprise monitoring."""

    SYNTHESIS_STARTED = "synthesis_started"
    SYNTHESIS_COMPLETED = "synthesis_completed"
    SYNTHESIS_FAILED = "synthesis_failed"
    SYNTHESIS_SKIPPED = "synthesis_skipped"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    MEMORY_STORED = "memory_stored"
    MEMORY_DELETED = "memory_deleted"
    CONVERSATION_STORED = "conversation_stored"
    CONVERSATION_DELETED = "conversation_deleted"
    EXPLICIT_EDIT_APPLIED = "explicit_edit_applied"
    IMPORT_COMPLETED = "import_completed"
    EXPORT_COMPLETED = "export_completed"
    STATE_CHANGED = "state_changed"
    RESET_COMPLETED = "reset_completed"
    EVICTION_RUN = "eviction_run"
    CIRCUIT_OPENED = "circuit_opened"
    CIRCUIT_CLOSED = "circuit_closed"
    RATE_LIMITED = "rate_limited"
    HEALTH_CHECK = "health_check"
    PROMPT_ASSEMBLED = "prompt_assembled"


class CircuitState(Enum):
    """Circuit breaker FSM states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class StructuredSynthesisOutput:
    """
    Structured synthesis payload emitted by the local synthesis backend.
    """

    summary: str = ""
    role: str = ""
    projects: List[str] = field(default_factory=list)
    tech_stack: List[str] = field(default_factory=list)
    preferences: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    confidence: float = 0.0
    coverage_notes: str = ""
    raw_json: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SummaryClaim:
    """Single structured claim extracted from a synthesized summary."""

    text: str
    category: str
    confidence: float
    source_conv_ids: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryScore:
    """Composite ranking score for a memory item."""

    memory_id: str
    semantic_score: float
    recency_score: float
    confidence_score: float
    salience_score: float
    explicit_bonus: float
    final_score: float


@dataclass
class UnifiedMemoryResult:
    """Unified retrieval result across summary, atomic, and archive layers."""

    content: str
    source_layer: str
    source_id: str
    relevance_score: float
    confidence: float
    memory_type: Optional["MemoryType"] = None
    citation_url: Optional[str] = None


@dataclass
class HybridRetrievalResult:
    """Budgeted unified retrieval payload."""

    items: List[UnifiedMemoryResult] = field(default_factory=list)
    layers_used: List[str] = field(default_factory=list)
    budget_used: int = 0


@dataclass
class PromptContext:
    """Structured prompt-context payload returned by the unified assembler."""

    user_memories_xml: str = ""
    profile_preferences_xml: str = ""
    project_instructions_xml: str = ""
    styles_xml: str = ""
    semantic_memories_xml: str = ""
    archive_references_xml: str = ""
    token_budget_used: Dict[str, int] = field(default_factory=dict)
    layers_suppressed: List[str] = field(default_factory=list)

    def total_tokens_used(self) -> int:
        """Return the total estimated token usage across all included layers."""
        return sum(self.token_budget_used.values())

    def combined_context(self) -> str:
        """Combine all non-empty context sections into a single developer block."""
        sections = [
            self.user_memories_xml,
            self.profile_preferences_xml,
            self.project_instructions_xml,
            self.styles_xml,
            self.semantic_memories_xml,
            self.archive_references_xml,
        ]
        return "\n\n".join(section for section in sections if section)

    def as_dict(self) -> Dict[str, str]:
        """Return the context in the legacy dictionary shape."""
        return {
            "userMemories": self.user_memories_xml,
            "profilePreferences": self.profile_preferences_xml,
            "projectInstructions": self.project_instructions_xml,
            "styles": self.styles_xml,
            "semanticMemories": self.semantic_memories_xml,
            "archiveReferences": self.archive_references_xml,
        }

    def to_messages(self, system_prompt: str = "") -> List[Dict[str, str]]:
        """Serialize the prompt context into system/developer messages."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        combined = self.combined_context()
        if combined:
            messages.append({"role": "developer", "content": combined})
        return messages


@dataclass
class GlobalMemorySummary:
    """
    Global memory summary — the primary persistent storage layer.
    Injected as <userMemories> XML into prompts.
    """

    user_id: str
    summary_text: str = ""
    last_synthesized: Optional[datetime] = None
    last_modified: datetime = field(default_factory=datetime.now)
    source_conversation_ids: Set[str] = field(default_factory=set)
    explicit_edits: List[Dict] = field(default_factory=list)
    confidence_score: float = 1.0
    structured_data: Optional[StructuredSynthesisOutput] = None
    claims: List[SummaryClaim] = field(default_factory=list)

    def to_xml(self) -> str:
        """Convert to XML format for prompt injection."""
        if not self.summary_text:
            return ""
        escaped_text = self._escape_xml(self.summary_text)
        return f"<userMemories>\n{escaped_text}\n</userMemories>"

    @staticmethod
    def _escape_xml(text: str) -> str:
        """Escape XML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )


@dataclass
class ProjectMemorySummary:
    """
    Per-project memory summary - isolated from global and other projects.
    Each project has its own independent memory that doesn't leak.
    """

    project_id: str
    user_id: str
    summary_text: str = ""
    last_synthesized: Optional[datetime] = None
    last_modified: datetime = field(default_factory=datetime.now)
    source_conversation_ids: Set[str] = field(default_factory=set)
    explicit_edits: List[Dict] = field(default_factory=list)
    confidence_score: float = 1.0
    structured_data: Optional[StructuredSynthesisOutput] = None
    claims: List[SummaryClaim] = field(default_factory=list)

    def to_xml(self) -> str:
        """Convert to XML format for prompt injection."""
        if not self.summary_text:
            return ""
        escaped_text = GlobalMemorySummary._escape_xml(self.summary_text)
        return f"<projectMemories>\n{escaped_text}\n</projectMemories>"


@dataclass
class ProfilePreferences:
    """
    Profile preferences — separate from memory, accessible in incognito.
    Preferences persist across all scope modes including incognito.
    """

    user_id: str
    communication_style: str = "balanced"
    technical_expertise: str = "intermediate"
    response_length: str = "medium"
    code_style: str = "clean"
    preferred_languages: List[str] = field(default_factory=list)
    custom_instructions: List[str] = field(default_factory=list)
    last_modified: datetime = field(default_factory=datetime.now)

    def to_xml(self) -> str:
        """Convert to XML format for prompt injection."""
        parts = ["<profilePreferences>"]

        if self.communication_style:
            parts.append(
                f"  <communicationStyle>{self.communication_style}</communicationStyle>"
            )
        if self.technical_expertise:
            parts.append(
                f"  <technicalExpertise>{self.technical_expertise}</technicalExpertise>"
            )
        if self.response_length:
            parts.append(f"  <responseLength>{self.response_length}</responseLength>")
        if self.code_style:
            parts.append(f"  <codeStyle>{self.code_style}</codeStyle>")
        if self.preferred_languages:
            langs = ", ".join(self.preferred_languages)
            parts.append(f"  <preferredLanguages>{langs}</preferredLanguages>")
        if self.custom_instructions:
            parts.append("  <customInstructions>")
            for instruction in self.custom_instructions:
                parts.append(
                    f"    <instruction>{self._escape_xml(instruction)}</instruction>"
                )
            parts.append("  </customInstructions>")

        parts.append("</profilePreferences>")
        return "\n".join(parts)

    @staticmethod
    def _escape_xml(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


@dataclass
class UserStyle:
    """User communication style preference."""

    style_id: str
    user_id: str
    name: str
    description: str
    system_prompt_additions: str = ""
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SourceConversationRef:
    """Reference to a source conversation - used in archive search results."""

    conversation_id: str
    chat_url: str
    title: str
    updated_at: datetime
    human_excerpt: str
    assistant_excerpt: str
    relevance_score: float = 0.0


@dataclass
class ConversationMessage:
    """Single message in a conversation."""

    message_id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_calls: Optional[List[Dict]] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class Conversation:
    """Full conversation record for archive storage."""

    conversation_id: str
    user_id: str
    project_id: Optional[str] = None
    is_incognito: bool = False
    scope: MemoryScope = MemoryScope.GLOBAL
    title: str = ""
    messages: List[ConversationMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_deleted: bool = False

    def __post_init__(self) -> None:
        """Derive scope from project/incognito flags."""
        if self.is_incognito:
            self.scope = MemoryScope.INCOGNITO
        elif self.project_id:
            self.scope = MemoryScope.PROJECT
        else:
            self.scope = MemoryScope.GLOBAL


@dataclass
class MemoryItem:
    """
    Legacy atomic memory item - used for semantic retrieval layer.
    Kept for backward compatibility and hybrid retrieval.
    """

    id: str
    user_id: str
    memory_type: MemoryType
    content: str
    embedding: Optional[np.ndarray] = None

    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    confidence: float = 1.0
    source_conversation_id: Optional[str] = None

    related_memories: Set[str] = field(default_factory=set)
    supersedes: Optional[str] = None

    tags: Set[str] = field(default_factory=set)
    importance_score: float = 0.5

    def to_dict(self) -> Dict:
        """Serialize the memory item for snapshot export."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "confidence": self.confidence,
            "importance_score": self.importance_score,
            "source_conversation_id": self.source_conversation_id,
            "related_memories": sorted(self.related_memories),
            "supersedes": self.supersedes,
            "tags": list(self.tags),
        }


@dataclass
class ConversationContext:
    """Current conversation state."""

    conversation_id: str
    user_id: str
    project_id: Optional[str] = None
    is_incognito: bool = False
    scope: MemoryScope = MemoryScope.GLOBAL
    messages: List[Dict] = field(default_factory=list)
    current_topic: Optional[str] = None
    active_tools: Set[str] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Derive scope automatically so callers cannot drift routing manually."""
        if self.is_incognito:
            self.scope = MemoryScope.INCOGNITO
        elif self.project_id:
            self.scope = MemoryScope.PROJECT
        else:
            self.scope = MemoryScope.GLOBAL


@dataclass
class MemoryRetrievalResult:
    """Result of memory retrieval."""

    memories: List[MemoryItem]
    relevance_scores: List[float]
    retrieval_method: str
    query_embedding: Optional[np.ndarray] = None


@dataclass
class ExplicitMemoryEdit:
    """Explicit memory edit from user command."""

    edit_id: str
    user_id: str
    scope: MemoryScope  # GLOBAL or PROJECT
    edit_type: str  # "add", "remove", "update"
    content: str
    project_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    applied: bool = False

    def to_dict(self) -> Dict:
        return {
            "edit_id": self.edit_id,
            "user_id": self.user_id,
            "scope": self.scope.value,
            "project_id": self.project_id,
            "edit_type": self.edit_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "applied": self.applied,
        }


@dataclass
class SynthesisAuditEntry:
    """Audit log entry for synthesis runs."""

    synthesis_id: str
    timestamp: datetime
    scope: MemoryScope
    project_id: Optional[str]
    conversations_processed: int
    new_content: str
    source_conversation_ids: List[str] = field(default_factory=list)
    backend_name: str = "deterministic"
    delta_mode: bool = False
    confidence_score: float = 0.0
    structured_output: Dict[str, Any] = field(default_factory=dict)
    changes_made: List[str] = field(default_factory=list)


@dataclass
class ImportLogEntry:
    """Audit log entry for structured imports."""

    import_id: str
    user_id: str
    scope: MemoryScope
    project_id: Optional[str]
    source: str
    imported_at: datetime = field(default_factory=datetime.now)
    edits_created: int = 0
    notes: str = ""


@dataclass
class MemoryControlSettings:
    """User-facing memory control settings."""

    user_id: str
    state: MemoryState = MemoryState.ON
    synthesis_interval_hours: int = 24
    enable_chat_search: bool = False  # User-controlled setting
    last_reset: Optional[datetime] = None
    total_synthesized_count: int = 0


# ============================================================================
# Embedding Backend Stages
# ============================================================================


class EmbeddingStage(Protocol):
    """Single embedding pipeline stage."""

    stage_name: str

    def embed(
        self,
        normalized_text: str,
        tokens: List[str],
        embedding_dim: int,
    ) -> np.ndarray:
        """Return a stage vector for the provided normalized text."""


class HashEmbeddingStage:
    """Token-hash embedding stage for deterministic sparse semantic recall."""

    stage_name = "hash"

    def embed(
        self,
        normalized_text: str,
        tokens: List[str],
        embedding_dim: int,
    ) -> np.ndarray:
        vec = np.zeros(embedding_dim, dtype=np.float32)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx_a = int.from_bytes(digest[:4], "big") % embedding_dim
            idx_b = int.from_bytes(digest[4:8], "big") % embedding_dim
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vec[idx_a] += sign
            vec[idx_b] += 0.5 * sign
        return self._normalize(vec)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector


class TokenGraphEmbeddingStage:
    """Co-occurrence graph stage that captures adjacency and ordering structure."""

    stage_name = "graph"

    def embed(
        self,
        normalized_text: str,
        tokens: List[str],
        embedding_dim: int,
    ) -> np.ndarray:
        vec = np.zeros(embedding_dim, dtype=np.float32)
        for left, right in zip(tokens, tokens[1:]):
            edge = f"{left}->{right}"
            digest = hashlib.sha256(edge.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % embedding_dim
            weight = 1.0 + (digest[4] / 255.0)
            vec[idx] += weight
        return self._normalize(vec)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector


class SemanticHintEmbeddingStage:
    """Keyword-bucket stage for stable domain-specific semantic cues."""

    stage_name = "semantic"

    KEYWORD_BUCKETS = {
        "memory": {
            "memory",
            "recall",
            "retrieve",
            "embedding",
            "context",
            "summary",
        },
        "project": {"project", "build", "working", "implement", "ship", "design"},
        "code": {"python", "torch", "jinja2", "numpy", "async", "module", "api"},
        "preference": {"prefer", "style", "format", "length", "tone"},
    }

    def embed(
        self,
        normalized_text: str,
        tokens: List[str],
        embedding_dim: int,
    ) -> np.ndarray:
        vec = np.zeros(embedding_dim, dtype=np.float32)
        token_set = set(tokens)
        for bucket, keywords in self.KEYWORD_BUCKETS.items():
            overlap = token_set & keywords
            if not overlap:
                continue
            digest = hashlib.sha256(bucket.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % embedding_dim
            vec[idx] += float(len(overlap))
        return self._normalize(vec)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector


class SpacyEmbeddingStage:
    """Optional spaCy-powered enrichment stage with graceful fallback."""

    stage_name = "spacy"

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self._nlp = None
        try:
            import spacy  # type: ignore

            self._nlp = spacy.load(model_name)
        except Exception:
            self._nlp = None

    def embed(
        self,
        normalized_text: str,
        tokens: List[str],
        embedding_dim: int,
    ) -> np.ndarray:
        if self._nlp is None or not normalized_text:
            return np.zeros(embedding_dim, dtype=np.float32)

        vec = np.zeros(embedding_dim, dtype=np.float32)
        doc = self._nlp(normalized_text)
        for token in doc:
            lemma = token.lemma_.strip().lower()
            if not lemma:
                continue
            digest = hashlib.sha256(f"spacy:{lemma}".encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % embedding_dim
            vec[idx] += 1.0 + (0.5 if token.pos_ in {"NOUN", "PROPN"} else 0.0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


class PipelineEmbeddingBackend:
    """Weighted local embedding pipeline composed from multiple stages."""

    DEFAULT_STAGE_ORDER = ["hash", "graph", "semantic"]
    DEFAULT_STAGE_WEIGHTS = {"hash": 0.5, "graph": 0.25, "semantic": 0.25, "spacy": 0.35}

    def __init__(
        self,
        stage_names: Optional[List[str]] = None,
        stage_weights: Optional[Dict[str, float]] = None,
        target_dim: int = 3072,
    ):
        self.stage_names = stage_names or list(self.DEFAULT_STAGE_ORDER)
        self.stage_weights = dict(self.DEFAULT_STAGE_WEIGHTS)
        self.target_dim = target_dim
        if stage_weights:
            self.stage_weights.update(stage_weights)
        self._stage_registry: Dict[str, EmbeddingStage] = {
            "hash": HashEmbeddingStage(),
            "graph": TokenGraphEmbeddingStage(),
            "semantic": SemanticHintEmbeddingStage(),
            "spacy": SpacyEmbeddingStage(),
        }

    @property
    def dim(self) -> int:
        """Embedding dimensionality. Satisfies EmbeddingInterface."""
        return self.target_dim

    def embed(
        self,
        normalized_text: str,
        tokens: Optional[List[str]] = None,
        embedding_dim: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Run the configured embedding stages and merge them into one vector."""
        if tokens is None:
            tokens = normalized_text.lower().split()
        if embedding_dim is None:
            embedding_dim = self.target_dim
        combined = np.zeros(embedding_dim, dtype=np.float32)
        total_weight = 0.0
        for stage_name in self.stage_names:
            stage = self._stage_registry.get(stage_name)
            if stage is None:
                continue
            weight = float(self.stage_weights.get(stage_name, 0.0))
            if weight <= 0:
                continue
            stage_vector = stage.embed(normalized_text, tokens, embedding_dim)
            combined += stage_vector * weight
            total_weight += weight

        if total_weight > 0:
            combined = combined / total_weight

        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        return combined


# ============================================================================
# Embedding & Semantic Search
# ============================================================================


class MemoryEmbedder:
    """
    Generates embeddings for memory items and queries.
    Uses deterministic local embedding with caching.
    """

    def __init__(
        self,
        embedding_model: str = "local-pipeline",
        embedding_dim: int = 3072,
        backend: Optional[PipelineEmbeddingBackend] = None,
        stage_names: Optional[List[str]] = None,
        stage_weights: Optional[Dict[str, float]] = None,
    ):
        self.embedding_model = embedding_model
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        self.backend = backend or PipelineEmbeddingBackend(
            stage_names=stage_names,
            stage_weights=stage_weights,
        )
        self._cache: Dict[str, np.ndarray] = {}
        self._token_pattern = re.compile(r"[A-Za-z0-9_]+")

    async def embed_memory(self, memory: MemoryItem) -> np.ndarray:
        """Generate embedding for a memory item."""
        prefix = self._get_contextual_prefix(memory.memory_type)
        text = f"{prefix}{memory.content}"
        embedding = await self._call_embedding_api(text)
        memory.embedding = embedding
        return embedding

    async def embed_query(
        self, query: str, query_context: Optional[Dict] = None
    ) -> np.ndarray:
        """Generate query embedding with optional context."""
        if query_context:
            context_str = self._format_query_context(query_context)
            text = f"{context_str}\n{query}"
        else:
            text = query
        return await self._call_embedding_api(text)

    def _get_contextual_prefix(self, memory_type: MemoryType) -> str:
        """Contextual prefix improves retrieval accuracy."""
        prefixes = {
            MemoryType.FACT: "User fact: ",
            MemoryType.PREFERENCE: "User preference: ",
            MemoryType.ENTITY: "Related entity: ",
            MemoryType.TOPIC: "Interest area: ",
            MemoryType.TOOL_USAGE: "Tool usage pattern: ",
        }
        return prefixes.get(memory_type, "")

    def _format_query_context(self, context: Dict) -> str:
        """Format query context for embedding."""
        parts = []
        if "current_topic" in context:
            parts.append(f"Topic: {context['current_topic']}")
        if "active_tools" in context:
            parts.append(f"Tools: {', '.join(context['active_tools'])}")
        return " | ".join(parts)

    async def _call_embedding_api(self, text: str) -> np.ndarray:
        """Deterministic local embedding pipeline (no external API dependency)."""
        normalized = self._normalize_text(text)
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached

        tokens = self._token_pattern.findall(normalized)
        if not tokens:
            embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            self._cache[normalized] = embedding
            return embedding

        vec = self.backend.embed(
            normalized_text=normalized,
            tokens=tokens,
            embedding_dim=self.embedding_dim,
        )

        self._cache[normalized] = vec
        return vec

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())


# ============================================================================
# Memory Storage & Retrieval
# ============================================================================


class MemoryStore:
    """
    Persistent memory storage with semantic search.
    Uses vector database for efficient retrieval.
    """

    def __init__(
        self,
        storage_backend: str = "in_memory",
        embedder: Optional[MemoryEmbedder] = None,
    ):
        if storage_backend != "in_memory":
            raise ValueError(
                "Unsupported storage_backend. Inject a custom MemoryStore instance "
                "for non-in-memory backends."
            )
        self.storage_backend = storage_backend
        self.embedder = embedder or MemoryEmbedder()

        self.memories: Dict[str, MemoryItem] = {}
        self.user_memories: Dict[str, Set[str]] = defaultdict(set)
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_ids: List[str] = []
        self._mutation_lock = asyncio.Lock()

    def _normalize_memory_type(
        self, memory_type: Optional[Union[MemoryType, str]]
    ) -> Optional[MemoryType]:
        if memory_type is None:
            return None
        if isinstance(memory_type, MemoryType):
            return memory_type
        if isinstance(memory_type, str):
            try:
                return MemoryType(memory_type.lower())
            except ValueError as exc:
                raise ValueError(f"Unknown memory type: {memory_type}") from exc
        raise TypeError("memory_type must be MemoryType, str, or None")

    def _normalize_memory_types(
        self, memory_types: Optional[List[Union[MemoryType, str]]]
    ) -> Optional[Set[MemoryType]]:
        if memory_types is None:
            return None
        normalized: Set[MemoryType] = set()
        for memory_type in memory_types:
            normalized_type = self._normalize_memory_type(memory_type)
            if normalized_type is not None:
                normalized.add(normalized_type)
        return normalized

    async def store_memory(self, memory: MemoryItem) -> str:
        """Store a memory item."""
        if not memory or not memory.user_id or not memory.content:
            raise ValueError("Memory must include user_id and content")

        if memory.embedding is None:
            await self.embedder.embed_memory(memory)
        elif memory.embedding.shape[0] != self.embedder.embedding_dim:
            raise ValueError("Memory embedding dimension mismatch")

        memory = await self._deduplicate(memory)

        async with self._mutation_lock:
            self.memories[memory.id] = memory
            self.user_memories[memory.user_id].add(memory.id)
            self._update_embedding_index(memory)

        return memory.id

    async def retrieve_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        memory_types: Optional[List[Union[MemoryType, str]]] = None,
        min_importance: float = 0.0,
    ) -> MemoryRetrievalResult:
        """Retrieve relevant memories for a query."""
        if not user_id:
            raise ValueError("user_id is required")
        if not query:
            return MemoryRetrievalResult(
                memories=[], relevance_scores=[], retrieval_method="empty_query"
            )
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if min_importance < 0.0 or min_importance > 1.0:
            raise ValueError("min_importance must be between 0.0 and 1.0")

        user_memory_ids = self.user_memories.get(user_id, set())
        if not user_memory_ids:
            return MemoryRetrievalResult(
                memories=[], relevance_scores=[], retrieval_method="empty"
            )

        query_embedding = await self.embedder.embed_query(query)

        normalized_types = self._normalize_memory_types(memory_types)
        candidate_memories = [
            self.memories[mid]
            for mid in user_memory_ids
            if (
                normalized_types is None
                or self.memories[mid].memory_type in normalized_types
            )
            and self.memories[mid].importance_score >= min_importance
        ]

        if not candidate_memories:
            return MemoryRetrievalResult(
                memories=[], relevance_scores=[], retrieval_method="filtered_empty"
            )

        for memory in candidate_memories:
            if memory.embedding is None:
                await self.embedder.embed_memory(memory)

        candidate_embeddings = np.stack([m.embedding for m in candidate_memories])

        similarities = self._compute_similarity(query_embedding, candidate_embeddings)
        scorer = MemoryScorer()
        scored_results = [
            (
                memory,
                scorer.score(
                    memory=memory,
                    semantic_similarity=float(similarity),
                    query_time=datetime.now(),
                ),
            )
            for memory, similarity in zip(candidate_memories, similarities)
        ]
        scored_results.sort(key=lambda item: item[1].final_score, reverse=True)
        scored_results = scored_results[:top_k]

        top_memories = [memory for memory, _score in scored_results]
        top_scores = [score.final_score for _memory, score in scored_results]

        for memory in top_memories:
            memory.last_accessed = datetime.now()
            memory.access_count += 1

        return MemoryRetrievalResult(
            memories=top_memories,
            relevance_scores=top_scores,
            retrieval_method="semantic_search",
            query_embedding=query_embedding,
        )

    async def retrieve_by_type(
        self,
        user_id: str,
        memory_type: Optional[Union[MemoryType, str]],
        limit: int = 20,
    ) -> List[MemoryItem]:
        """Retrieve memories by type (no semantic search)."""
        if not user_id:
            raise ValueError("user_id is required")
        if limit <= 0:
            raise ValueError("limit must be positive")

        user_memory_ids = self.user_memories.get(user_id, set())
        normalized_type = self._normalize_memory_type(memory_type)

        type_memories = [
            self.memories[mid]
            for mid in user_memory_ids
            if normalized_type is None
            or self.memories[mid].memory_type == normalized_type
        ]

        type_memories.sort(
            key=lambda m: (m.importance_score, m.created_at), reverse=True
        )

        return type_memories[:limit]

    async def update_memory(self, memory_id: str, updates: Dict) -> MemoryItem:
        """Update an existing memory."""
        async with self._mutation_lock:
            if memory_id not in self.memories:
                raise ValueError(f"Memory {memory_id} not found")

            memory = self.memories[memory_id]

            if "content" in updates:
                memory.content = updates["content"]
                await self.embedder.embed_memory(memory)
                self._update_embedding_index(memory)

            if "importance_score" in updates:
                memory.importance_score = updates["importance_score"]

            if "confidence" in updates:
                memory.confidence = updates["confidence"]

            return memory

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        async with self._mutation_lock:
            if memory_id not in self.memories:
                return False

            memory = self.memories[memory_id]
            del self.memories[memory_id]
            self.user_memories[memory.user_id].discard(memory_id)

            self._remove_from_embedding_index(memory_id)

            return True

    async def _deduplicate(self, new_memory: MemoryItem) -> MemoryItem:
        """Check for duplicate or conflicting memories."""
        if new_memory.embedding is None:
            await self.embedder.embed_memory(new_memory)

        user_memories = [
            self.memories[mid]
            for mid in self.user_memories.get(new_memory.user_id, set())
            if self.memories[mid].memory_type == new_memory.memory_type
        ]

        if not user_memories:
            return new_memory

        for memory in user_memories:
            if memory.embedding is None:
                await self.embedder.embed_memory(memory)

        similarities = self._compute_similarity(
            new_memory.embedding, np.stack([m.embedding for m in user_memories])
        )

        max_sim_idx = np.argmax(similarities)
        max_similarity = similarities[max_sim_idx]

        if max_similarity > 0.95:
            existing = user_memories[max_sim_idx]
            existing.confidence = max(existing.confidence, new_memory.confidence)
            existing.importance_score = max(
                existing.importance_score, new_memory.importance_score
            )
            return existing

        elif max_similarity > 0.8:
            existing = user_memories[max_sim_idx]
            new_memory.related_memories.add(existing.id)
            existing.related_memories.add(new_memory.id)

        return new_memory

    def _compute_similarity(
        self, query_emb: np.ndarray, candidate_embs: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity."""
        if query_emb.ndim != 1:
            raise ValueError("query_emb must be a 1D array")
        if candidate_embs.ndim != 2:
            raise ValueError("candidate_embs must be a 2D array")
        if candidate_embs.shape[1] != query_emb.shape[0]:
            raise ValueError("Embedding dimension mismatch")

        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        candidate_norms = candidate_embs / (
            np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-8
        )

        similarities = candidate_norms @ query_norm
        return similarities

    def _update_embedding_index(self, memory: MemoryItem):
        """Update the embedding index with new memory."""
        if memory.embedding is None:
            raise ValueError("Cannot update embedding index without embedding")
        if memory.id in self.embedding_ids:
            idx = self.embedding_ids.index(memory.id)
            if self.embeddings is not None:
                self.embeddings[idx] = memory.embedding
        else:
            self.embedding_ids.append(memory.id)
            if self.embeddings is None:
                self.embeddings = memory.embedding.reshape(1, -1)
            else:
                self.embeddings = np.vstack(
                    [self.embeddings, memory.embedding.reshape(1, -1)]
                )

    def _remove_from_embedding_index(self, memory_id: str):
        """Remove memory from embedding index."""
        if memory_id not in self.embedding_ids:
            return

        idx = self.embedding_ids.index(memory_id)
        self.embedding_ids.pop(idx)

        if self.embeddings is not None:
            self.embeddings = np.delete(self.embeddings, idx, axis=0)


# ============================================================================
# Retrieval Scoring
# ============================================================================


class MemoryScorer:
    """Multi-signal ranking for atomic memory retrieval."""

    DEFAULT_WEIGHTS = {
        "semantic": 0.45,
        "recency": 0.20,
        "confidence": 0.15,
        "salience": 0.15,
        "explicit_bonus": 0.05,
    }

    def score(
        self,
        memory: MemoryItem,
        semantic_similarity: float,
        query_time: datetime,
        weights: Optional[Dict[str, float]] = None,
    ) -> MemoryScore:
        """Score a memory item using semantic, temporal, and salience signals."""
        resolved_weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            resolved_weights.update(weights)

        recency_score = self._recency_score(memory, query_time)
        confidence_score = max(0.0, min(memory.confidence, 1.0))
        salience_score = self._salience_score(memory)
        explicit_bonus = self._explicit_bonus(memory)

        final_score = (
            resolved_weights["semantic"] * semantic_similarity
            + resolved_weights["recency"] * recency_score
            + resolved_weights["confidence"] * confidence_score
            + resolved_weights["salience"] * salience_score
            + resolved_weights["explicit_bonus"] * explicit_bonus
        )

        return MemoryScore(
            memory_id=memory.id,
            semantic_score=semantic_similarity,
            recency_score=recency_score,
            confidence_score=confidence_score,
            salience_score=salience_score,
            explicit_bonus=explicit_bonus,
            final_score=final_score,
        )

    def _recency_score(self, memory: MemoryItem, now: datetime) -> float:
        """Exponential decay with a 30-day half-life."""
        age_days = max((now - memory.last_accessed).days, 0)
        return math.exp(-age_days * math.log(2) / 30.0)

    def _salience_score(self, memory: MemoryItem) -> float:
        """Blend importance with log-scaled access count."""
        access_signal = math.log1p(max(memory.access_count, 0)) / 10.0
        return 0.7 * memory.importance_score + 0.3 * min(access_signal, 1.0)

    def _explicit_bonus(self, memory: MemoryItem) -> float:
        """Boost user-authored memories so they survive tie-breaks."""
        if memory.tags and "explicit" in memory.tags:
            return 1.0
        return 0.0


# ============================================================================
# Conversation Archive (Layer 3 - RAG)
# ============================================================================


class ConversationArchive:
    """
    Full conversation archive with RAG search capabilities.
    This is Layer 3 - on-demand retrieval distinct from always-on summary.
    Available to all users when chat search is enabled in settings.
    """

    def __init__(
        self,
        embedder: Optional[MemoryEmbedder] = None,
        vector_backend: str = "in_memory",
        chat_url_base: str = "/chat",
    ):
        if vector_backend != "in_memory":
            raise ValueError(
                "Unsupported vector_backend. Inject a custom ConversationArchive "
                "instance for non-in-memory backends."
            )
        self.embedder = embedder or MemoryEmbedder()
        self.vector_backend = vector_backend
        self.chat_url_base = chat_url_base.rstrip("/")

        self.conversations: Dict[str, Conversation] = {}
        self.user_conversations: Dict[str, Set[str]] = defaultdict(set)
        self.project_conversations: Dict[str, Set[str]] = defaultdict(set)

        self.conversation_embeddings: Dict[str, np.ndarray] = {}

        self._index_lock = asyncio.Lock()

    async def store_conversation(self, conversation: Conversation) -> str:
        """Store a full conversation."""
        self.conversations[conversation.conversation_id] = conversation
        self.user_conversations[conversation.user_id].add(conversation.conversation_id)

        if conversation.project_id:
            self.project_conversations[conversation.project_id].add(
                conversation.conversation_id
            )

        await self._index_conversation(conversation)

        return conversation.conversation_id

    async def _index_conversation(self, conversation: Conversation) -> None:
        """Index conversation for search."""
        full_text = self._conversation_to_text(conversation)
        if not full_text:
            return

        normalized = self.embedder._normalize_text(full_text)
        embedding = await self.embedder._call_embedding_api(normalized)
        async with self._index_lock:
            self.conversation_embeddings[conversation.conversation_id] = embedding

    def _conversation_to_text(self, conversation: Conversation) -> str:
        """Convert conversation to searchable text."""
        parts = [conversation.title]
        for msg in conversation.messages:
            role = msg.role.upper()
            content = msg.content[:500]
            parts.append(f"{role}: {content}")
        return " ".join(parts)

    async def search(
        self,
        user_id: str,
        query: str,
        max_results: int = 5,
        project_id: Optional[str] = None,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
    ) -> List[SourceConversationRef]:
        """Search conversations using RAG."""
        if max_results < 1 or max_results > 10:
            max_results = max(1, min(10, max_results))

        user_conv_ids = self.user_conversations.get(user_id, set())

        candidates = []
        for conv_id in user_conv_ids:
            conv = self.conversations.get(conv_id)
            if not conv or conv.is_deleted:
                continue

            if conv.is_incognito:
                continue

            if project_id and conv.project_id != project_id:
                continue

            if before and conv.updated_at > before:
                continue

            if after and conv.updated_at < after:
                continue

            candidates.append(conv)

        if not candidates:
            return []

        query_embedding = await self.embedder.embed_query(query)

        results = []
        for conv in candidates:
            emb = self.conversation_embeddings.get(conv.conversation_id)
            if emb is None:
                continue

            sim = self._cosine_similarity(query_embedding, emb)

            excerpt = self._extract_excerpt(conv, query)

            ref = SourceConversationRef(
                conversation_id=conv.conversation_id,
                chat_url=f"{self.chat_url_base}/{conv.conversation_id}",
                title=conv.title or "Untitled",
                updated_at=conv.updated_at,
                human_excerpt=excerpt["human"],
                assistant_excerpt=excerpt["assistant"],
                relevance_score=float(sim),
            )
            results.append(ref)

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:max_results]

    def get_recent(
        self,
        user_id: str,
        n: int = 3,
        sort_order: str = "desc",
        project_id: Optional[str] = None,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
    ) -> List[SourceConversationRef]:
        """Get recent conversations."""
        if n < 1 or n > 20:
            n = max(1, min(20, n))

        user_conv_ids = self.user_conversations.get(user_id, set())

        candidates = []
        for conv_id in user_conv_ids:
            conv = self.conversations.get(conv_id)
            if not conv or conv.is_deleted or conv.is_incognito:
                continue

            if project_id and conv.project_id != project_id:
                continue

            if before and conv.updated_at > before:
                continue

            if after and conv.updated_at < after:
                continue

            candidates.append(conv)

        if sort_order == "asc":
            candidates.sort(key=lambda c: c.updated_at)
        else:
            candidates.sort(key=lambda c: c.updated_at, reverse=True)

        results = []
        for conv in candidates[:n]:
            excerpt = self._extract_excerpt(conv, "")

            ref = SourceConversationRef(
                conversation_id=conv.conversation_id,
                chat_url=f"{self.chat_url_base}/{conv.conversation_id}",
                title=conv.title or "Untitled",
                updated_at=conv.updated_at,
                human_excerpt=excerpt["human"],
                assistant_excerpt=excerpt["assistant"],
                relevance_score=1.0,
            )
            results.append(ref)

        return results

    def _extract_excerpt(
        self, conversation: Conversation, query: str
    ) -> Dict[str, str]:
        """Extract relevant excerpts from conversation."""
        human_excerpt = ""
        assistant_excerpt = ""

        for msg in reversed(conversation.messages):
            if msg.role == "user" and not human_excerpt:
                human_excerpt = msg.content[:200]
            elif msg.role == "assistant" and not assistant_excerpt:
                assistant_excerpt = msg.content[:200]

            if human_excerpt and assistant_excerpt:
                break

        return {"human": human_excerpt, "assistant": assistant_excerpt}

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        return float(a_norm @ b_norm)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Mark conversation as deleted (for synthesis refresh)."""
        if conversation_id not in self.conversations:
            return False

        self.conversations[conversation_id].is_deleted = True
        self.conversation_embeddings.pop(conversation_id, None)
        return True

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)


# ============================================================================
# Summary Synthesis Engine
# ============================================================================


class MemoryCategoryFilter:
    """Configurable domain filter for deciding which conversations enter synthesis."""

    DOMAIN_KEYWORDS = {
        "work": {
            "project",
            "code",
            "bug",
            "feature",
            "api",
            "database",
            "server",
            "deployment",
            "architecture",
            "design",
            "implement",
            "module",
            "interface",
            "testing",
            "documentation",
            "refactor",
            "optimize",
            "performance",
            "python",
            "torch",
            "router",
            "memory",
            "system",
            "build",
            "graph",
            "embedding",
            "audit",
            "retrieval",
            "neural",
            "transformer",
            "model",
            "research",
        },
        "technical": {
            "python",
            "javascript",
            "torch",
            "jinja2",
            "numpy",
            "model",
            "embedding",
            "vector",
            "async",
            "architecture",
        },
        "personal": {
            "family",
            "home",
            "vacation",
            "hobby",
            "friend",
            "weekend",
            "birthday",
        },
    }

    def __init__(self, domain: str = "work", threshold: int = 2):
        self.domain = domain
        self.threshold = threshold

    def is_relevant(self, conversation: Conversation) -> bool:
        """Return True when a conversation matches the configured domain."""
        keywords = self.DOMAIN_KEYWORDS.get(self.domain, set())
        if not keywords:
            return True

        full_text = self._conversation_to_text(conversation).lower()
        score = sum(1 for keyword in keywords if keyword in full_text)
        return score >= self.threshold

    def _conversation_to_text(self, conversation: Conversation) -> str:
        parts = [conversation.title or ""]
        for message in conversation.messages:
            parts.append(message.content)
        return " ".join(parts)


class SynthesisBackend(Protocol):
    """Local/pluggable synthesis backend contract."""

    backend_name: str

    async def synthesize(
        self,
        conversations: List[Conversation],
        existing_output: Optional[StructuredSynthesisOutput] = None,
    ) -> StructuredSynthesisOutput:
        """Produce a structured synthesis payload from a set of conversations."""


class StructuredSynthesisFormatter:
    """Render structured synthesis outputs into summary text for XML injection."""

    def to_summary_text(self, output: StructuredSynthesisOutput) -> str:
        """Format structured synthesis output into stable plain text."""
        sections: List[str] = []
        if output.summary:
            sections.append(output.summary.strip())

        if output.role:
            sections.append(f"Role: {output.role}")
        if output.projects:
            sections.append("Projects:\n- " + "\n- ".join(output.projects))
        if output.tech_stack:
            sections.append("Tech Stack:\n- " + "\n- ".join(output.tech_stack))
        if output.preferences:
            sections.append("Preferences:\n- " + "\n- ".join(output.preferences))
        if output.goals:
            sections.append("Goals:\n- " + "\n- ".join(output.goals))
        if output.constraints:
            sections.append("Constraints:\n- " + "\n- ".join(output.constraints))
        if output.coverage_notes:
            sections.append(f"Coverage Notes: {output.coverage_notes}")

        return "\n\n".join(section for section in sections if section)


class SummaryClaimExtractor:
    """Extract fine-grained claims from a structured synthesis output."""

    def extract(
        self,
        output: StructuredSynthesisOutput,
        conversations: Optional[List[Conversation]] = None,
        source_conv_ids: Optional[Set[str]] = None,
    ) -> List[SummaryClaim]:
        """Produce claim objects for later trimming, auditing, and provenance."""
        fallback_source_ids = sorted(source_conv_ids or set())
        claims: List[SummaryClaim] = []

        def _append_claims(category: str, items: List[str]) -> None:
            for item in items:
                claims.append(
                    SummaryClaim(
                        text=item,
                        category=category,
                        confidence=output.confidence,
                        source_conv_ids=self._resolve_source_ids(
                            item,
                            conversations,
                            fallback_source_ids,
                        ),
                    )
                )

        if output.role:
            claims.append(
                SummaryClaim(
                    text=output.role,
                    category="role",
                    confidence=output.confidence,
                    source_conv_ids=self._resolve_source_ids(
                        output.role,
                        conversations,
                        fallback_source_ids,
                    ),
                )
            )

        _append_claims("projects", output.projects)
        _append_claims("tech_stack", output.tech_stack)
        _append_claims("preferences", output.preferences)
        _append_claims("goals", output.goals)
        _append_claims("constraints", output.constraints)

        return claims

    def _resolve_source_ids(
        self,
        claim_text: str,
        conversations: Optional[List[Conversation]],
        fallback_source_ids: List[str],
    ) -> List[str]:
        """Map a claim to the most plausible supporting conversation IDs."""
        if not claim_text:
            return list(fallback_source_ids)
        if not conversations:
            return list(fallback_source_ids)

        claim_tokens = self._normalize_tokens(claim_text)
        if not claim_tokens:
            return list(fallback_source_ids)

        matches: List[str] = []
        for conversation in conversations:
            conversation_tokens = self._conversation_tokens(conversation)
            if not conversation_tokens:
                continue
            overlap = claim_tokens & conversation_tokens
            overlap_ratio = len(overlap) / max(len(claim_tokens), 1)
            if overlap_ratio >= 0.5:
                matches.append(conversation.conversation_id)
                continue

            conversation_text = self._conversation_text(conversation)
            if claim_text.lower() in conversation_text:
                matches.append(conversation.conversation_id)

        return matches or list(fallback_source_ids)

    def _conversation_text(self, conversation: Conversation) -> str:
        parts = [conversation.title or ""]
        parts.extend(message.content for message in conversation.messages)
        return " ".join(parts).lower()

    def _conversation_tokens(self, conversation: Conversation) -> Set[str]:
        return self._normalize_tokens(self._conversation_text(conversation))

    def _normalize_tokens(self, text: str) -> Set[str]:
        tokens = {
            token
            for token in re.findall(r"[a-z0-9_]+", text.lower())
            if len(token) > 2 and token not in {"the", "and", "for", "with", "that"}
        }
        return tokens


class SummaryClaimScorer:
    """Rank summary claims so budget trimming keeps the most durable facts."""

    CATEGORY_WEIGHTS = {
        "role": 1.00,
        "projects": 0.95,
        "constraints": 0.90,
        "preferences": 0.85,
        "goals": 0.82,
        "tech_stack": 0.80,
    }

    def score(self, claim: SummaryClaim, now: Optional[datetime] = None) -> float:
        """Return a composite summary-claim retention score."""
        current_time = now or datetime.now()
        category_score = self.CATEGORY_WEIGHTS.get(claim.category, 0.70)
        confidence_score = max(0.0, min(claim.confidence, 1.0))
        age_days = max((current_time - claim.last_updated).days, 0)
        recency_score = math.exp(-age_days * math.log(2) / 45.0)
        provenance_bonus = min(0.15, 0.05 * len(claim.source_conv_ids))
        return (
            0.45 * confidence_score
            + 0.30 * category_score
            + 0.20 * recency_score
            + 0.05 * provenance_bonus
        )


class DeterministicSynthesisBackend:
    """Local structured synthesis backend with no network dependency."""

    ROLE_PATTERNS = [
        re.compile(r"\b(?:i am|i'm|i work as|my role is)\s+([^.,;!?]+)", re.I),
        re.compile(r"\b(?:we are building|i'm building|i am building)\s+([^.,;!?]+)", re.I),
    ]
    PROJECT_KEYWORDS = (
        "project",
        "working on",
        "building",
        "system",
        "router",
        "engine",
        "pipeline",
        "toolkit",
        "memory",
    )
    PREFERENCE_KEYWORDS = ("prefer", "like", "love", "hate", "dislike", "want")
    GOAL_KEYWORDS = ("goal", "need to", "trying to", "working on", "build", "implement")
    CONSTRAINT_KEYWORDS = ("must", "cannot", "can't", "do not", "never", "only")
    TECH_KEYWORDS = (
        "python",
        "numpy",
        "torch",
        "jinja2",
        "spacy",
        "onnx",
        "safetensors",
        "embedding",
        "vector",
        "graph",
        "asyncio",
        "transformer",
    )

    backend_name = "deterministic"

    async def synthesize(
        self,
        conversations: List[Conversation],
        existing_output: Optional[StructuredSynthesisOutput] = None,
    ) -> StructuredSynthesisOutput:
        """Synthesize structured memory from conversation content."""
        user_sentences = self._collect_user_sentences(conversations)
        role = self._extract_role(user_sentences, existing_output)
        projects = self._collect_matching_sentences(user_sentences, self.PROJECT_KEYWORDS)
        preferences = self._collect_matching_sentences(
            user_sentences, self.PREFERENCE_KEYWORDS
        )
        goals = self._collect_matching_sentences(user_sentences, self.GOAL_KEYWORDS)
        constraints = self._collect_matching_sentences(
            user_sentences, self.CONSTRAINT_KEYWORDS
        )
        tech_stack = self._extract_tech_stack(user_sentences, existing_output)

        coverage_gaps = []
        if not role:
            coverage_gaps.append("role")
        if not projects:
            coverage_gaps.append("projects")
        if not tech_stack:
            coverage_gaps.append("tech_stack")

        confidence = min(
            0.95,
            0.20
            + 0.12 * int(bool(role))
            + 0.10 * min(len(projects), 3)
            + 0.08 * min(len(tech_stack), 4)
            + 0.08 * min(len(preferences), 2)
            + 0.08 * min(len(goals), 2)
            + 0.08 * min(len(constraints), 2),
        )

        summary = self._compose_summary(
            role=role,
            projects=projects,
            tech_stack=tech_stack,
            preferences=preferences,
            goals=goals,
            constraints=constraints,
        )

        raw_json = {
            "summary": summary,
            "structured": {
                "role": role,
                "projects": projects,
                "tech_stack": tech_stack,
                "preferences": preferences,
                "goals": goals,
                "constraints": constraints,
            },
            "confidence": confidence,
            "coverage_notes": (
                "Could not infer: " + ", ".join(coverage_gaps)
                if coverage_gaps
                else "No major coverage gaps detected."
            ),
        }

        return StructuredSynthesisOutput(
            summary=summary,
            role=role,
            projects=projects,
            tech_stack=tech_stack,
            preferences=preferences,
            goals=goals,
            constraints=constraints,
            confidence=confidence,
            coverage_notes=raw_json["coverage_notes"],
            raw_json=raw_json,
        )

    def _collect_user_sentences(self, conversations: List[Conversation]) -> List[str]:
        """Collect normalized user statements from conversations."""
        sentences: List[str] = []
        for conversation in conversations:
            for message in conversation.messages:
                if message.role != "user":
                    continue
                for chunk in re.split(r"[.!?]+", message.content):
                    candidate = chunk.strip()
                    if len(candidate) >= 12:
                        sentences.append(candidate)
        return self._unique_preserving_order(sentences)

    def _extract_role(
        self,
        sentences: List[str],
        existing_output: Optional[StructuredSynthesisOutput],
    ) -> str:
        """Extract the best role-like statement from user text."""
        if existing_output and existing_output.role:
            fallback_role = existing_output.role
        else:
            fallback_role = ""

        for sentence in sentences:
            for pattern in self.ROLE_PATTERNS:
                match = pattern.search(sentence)
                if match:
                    return match.group(1).strip()

        return fallback_role

    def _collect_matching_sentences(
        self,
        sentences: List[str],
        keywords: Tuple[str, ...],
        limit: int = 5,
    ) -> List[str]:
        """Collect unique sentences that match a keyword set."""
        matched = [
            sentence
            for sentence in sentences
            if any(keyword in sentence.lower() for keyword in keywords)
        ]
        return self._unique_preserving_order(matched)[:limit]

    def _extract_tech_stack(
        self,
        sentences: List[str],
        existing_output: Optional[StructuredSynthesisOutput],
    ) -> List[str]:
        """Extract technical terms from user statements."""
        tech_terms = list(existing_output.tech_stack) if existing_output else []
        seen = {term.lower() for term in tech_terms}
        for sentence in sentences:
            lowered = sentence.lower()
            for keyword in self.TECH_KEYWORDS:
                if keyword in lowered and keyword not in seen:
                    tech_terms.append(keyword)
                    seen.add(keyword)
        return tech_terms[:8]

    def _compose_summary(
        self,
        role: str,
        projects: List[str],
        tech_stack: List[str],
        preferences: List[str],
        goals: List[str],
        constraints: List[str],
    ) -> str:
        """Compose a compact prose summary from structured fields."""
        summary_parts: List[str] = []
        if role:
            summary_parts.append(f"The user presents primarily as {role}.")
        if projects:
            summary_parts.append(
                "Active project context centers on "
                + "; ".join(projects[:3])
                + "."
            )
        if tech_stack:
            summary_parts.append(
                "The visible technical stack includes "
                + ", ".join(tech_stack[:6])
                + "."
            )
        if preferences:
            summary_parts.append(
                "Stated working or communication preferences include "
                + "; ".join(preferences[:3])
                + "."
            )
        if goals:
            summary_parts.append(
                "Current goals appear to include " + "; ".join(goals[:3]) + "."
            )
        if constraints:
            summary_parts.append(
                "Explicit constraints include " + "; ".join(constraints[:3]) + "."
            )

        if not summary_parts:
            return "No stable structured memory could be synthesized from the available conversations."

        return " ".join(summary_parts)

    def _unique_preserving_order(self, items: List[str]) -> List[str]:
        seen: Set[str] = set()
        result: List[str] = []
        for item in items:
            normalized = item.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(item)
        return result


class SpacySynthesisBackend:
    """
    Optional enrichment layer. Uses spaCy if present, otherwise becomes a no-op
    stage that simply returns the incoming structure.
    """

    backend_name = "spacy_optional"

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self._nlp = None
        try:
            import spacy  # type: ignore

            self._nlp = spacy.load(model_name)
        except Exception:
            self._nlp = None

    async def synthesize(
        self,
        conversations: List[Conversation],
        existing_output: Optional[StructuredSynthesisOutput] = None,
    ) -> StructuredSynthesisOutput:
        """Return a lightly enriched structure when spaCy is available."""
        base = existing_output or StructuredSynthesisOutput()
        if self._nlp is None:
            return base

        combined_text = "\n".join(
            message.content
            for conversation in conversations
            for message in conversation.messages
            if message.role == "user"
        )
        if not combined_text.strip():
            return base

        doc = self._nlp(combined_text)
        projects = list(base.projects)
        tech_stack = list(base.tech_stack)
        seen_projects = {item.lower() for item in projects}
        seen_tech = {item.lower() for item in tech_stack}

        for ent in doc.ents:
            if ent.label_ in {"ORG", "PRODUCT", "WORK_OF_ART"}:
                candidate = ent.text.strip()
                if candidate and candidate.lower() not in seen_projects:
                    projects.append(candidate)
                    seen_projects.add(candidate.lower())

        for token in doc:
            candidate = token.lemma_.strip().lower()
            if candidate in DeterministicSynthesisBackend.TECH_KEYWORDS and candidate not in seen_tech:
                tech_stack.append(candidate)
                seen_tech.add(candidate)

        return StructuredSynthesisOutput(
            summary=base.summary,
            role=base.role,
            projects=projects[:8],
            tech_stack=tech_stack[:8],
            preferences=list(base.preferences),
            goals=list(base.goals),
            constraints=list(base.constraints),
            confidence=base.confidence,
            coverage_notes=base.coverage_notes,
            raw_json=dict(base.raw_json),
        )


class PipelineSynthesisBackend:
    """Compose multiple local synthesis stages into one backend."""

    backend_name = "pipeline"

    def __init__(self, stages: Optional[List[SynthesisBackend]] = None):
        self.stages = stages or [DeterministicSynthesisBackend(), SpacySynthesisBackend()]

    async def synthesize(
        self,
        conversations: List[Conversation],
        existing_output: Optional[StructuredSynthesisOutput] = None,
    ) -> StructuredSynthesisOutput:
        """Run each stage sequentially, merging the structured output."""
        current = existing_output or StructuredSynthesisOutput()
        for stage in self.stages:
            stage_output = await stage.synthesize(conversations, current)
            current = self._merge_outputs(current, stage_output)
        return current

    def _merge_outputs(
        self,
        base: StructuredSynthesisOutput,
        update: StructuredSynthesisOutput,
    ) -> StructuredSynthesisOutput:
        """Merge two synthesis outputs while preserving stable order."""
        return StructuredSynthesisOutput(
            summary=update.summary or base.summary,
            role=update.role or base.role,
            projects=self._merge_lists(base.projects, update.projects),
            tech_stack=self._merge_lists(base.tech_stack, update.tech_stack),
            preferences=self._merge_lists(base.preferences, update.preferences),
            goals=self._merge_lists(base.goals, update.goals),
            constraints=self._merge_lists(base.constraints, update.constraints),
            confidence=max(base.confidence, update.confidence),
            coverage_notes=update.coverage_notes or base.coverage_notes,
            raw_json=update.raw_json or base.raw_json,
        )

    def _merge_lists(self, left: List[str], right: List[str]) -> List[str]:
        seen: Set[str] = set()
        merged: List[str] = []
        for value in left + right:
            normalized = value.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            merged.append(value)
        return merged


class SummaryRefreshSynthesizer:
    """
    Background synthesis engine with structured local synthesis, delta mode,
    work-category filtering, and an exposed audit trail.
    """

    def __init__(
        self,
        archive: ConversationArchive,
        synthesis_interval_hours: int = 24,
        work_focused_filter: bool = True,
        synthesis_backend: Optional[SynthesisBackend] = None,
        category_filter: Optional[MemoryCategoryFilter] = None,
        claim_extractor: Optional[SummaryClaimExtractor] = None,
        formatter: Optional[StructuredSynthesisFormatter] = None,
    ):
        self.archive = archive
        self.synthesis_interval = timedelta(hours=synthesis_interval_hours)
        self.work_focused_filter = work_focused_filter
        self.category_filter = category_filter or MemoryCategoryFilter(
            domain="work", threshold=2 if work_focused_filter else 0
        )
        self.synthesis_backend = synthesis_backend or PipelineSynthesisBackend()
        self.claim_extractor = claim_extractor or SummaryClaimExtractor()
        self.formatter = formatter or StructuredSynthesisFormatter()
        self.synthesis_history: List[SynthesisAuditEntry] = []

    async def synthesize(
        self,
        user_id: str,
        scope: MemoryScope,
        project_id: Optional[str] = None,
        existing_summary: Optional[Union[GlobalMemorySummary, ProjectMemorySummary]] = None,
        force_rebuild: bool = False,
    ) -> GlobalMemorySummary:
        """
        Synthesize a scope-appropriate summary. When an existing summary is
        provided, only conversations newer than the last synthesis are processed.
        """
        cutoff = None
        if (
            not force_rebuild
            and existing_summary
            and existing_summary.last_synthesized
        ):
            cutoff = existing_summary.last_synthesized
        conversations = self._get_recent_conversations(user_id, project_id, cutoff)

        if not conversations:
            if force_rebuild:
                return GlobalMemorySummary(
                    user_id=user_id,
                    summary_text="",
                    last_synthesized=datetime.now(),
                    source_conversation_ids=set(),
                    explicit_edits=list(getattr(existing_summary, "explicit_edits", [])),
                    confidence_score=0.0,
                    structured_data=None,
                    claims=[],
                )
            return GlobalMemorySummary(
                user_id=user_id,
                summary_text=getattr(existing_summary, "summary_text", ""),
                last_synthesized=getattr(existing_summary, "last_synthesized", None),
                source_conversation_ids=set(
                    getattr(existing_summary, "source_conversation_ids", set())
                ),
                explicit_edits=list(getattr(existing_summary, "explicit_edits", [])),
                confidence_score=getattr(existing_summary, "confidence_score", 0.0),
                structured_data=getattr(existing_summary, "structured_data", None),
                claims=list(getattr(existing_summary, "claims", [])),
            )

        existing_output = getattr(existing_summary, "structured_data", None)
        structured_output = await self.synthesis_backend.synthesize(
            conversations=conversations,
            existing_output=existing_output,
        )
        summary_text = self.formatter.to_summary_text(structured_output)
        source_ids = {
            *set(getattr(existing_summary, "source_conversation_ids", set())),
            *(conversation.conversation_id for conversation in conversations),
        }
        claims = self.claim_extractor.extract(
            structured_output,
            conversations=conversations,
            source_conv_ids=source_ids,
        )
        changes_made = self._describe_changes(existing_output, structured_output)

        audit_entry = SynthesisAuditEntry(
            synthesis_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            scope=scope,
            project_id=project_id,
            conversations_processed=len(conversations),
            new_content=summary_text[:500],
            source_conversation_ids=sorted(source_ids),
            backend_name=self.synthesis_backend.backend_name,
            delta_mode=(cutoff is not None and not force_rebuild),
            confidence_score=structured_output.confidence,
            structured_output=structured_output.raw_json,
            changes_made=changes_made,
        )
        self.synthesis_history.append(audit_entry)

        return GlobalMemorySummary(
            user_id=user_id,
            summary_text=summary_text,
            last_synthesized=datetime.now(),
            source_conversation_ids=source_ids,
            confidence_score=structured_output.confidence,
            structured_data=structured_output,
            claims=claims,
            explicit_edits=list(getattr(existing_summary, "explicit_edits", [])),
        )

    def get_audit_log(self) -> List[SynthesisAuditEntry]:
        """Return all synthesis audit entries."""
        return list(self.synthesis_history)

    def _describe_changes(
        self,
        previous: Optional[StructuredSynthesisOutput],
        current: StructuredSynthesisOutput,
    ) -> List[str]:
        """Describe the structured delta applied by the current synthesis run."""
        if previous is None:
            return ["initial_summary_created"]

        changes: List[str] = []
        if current.role != previous.role:
            changes.append(
                f"role:{previous.role or '<empty>'}->{current.role or '<empty>'}"
            )

        for field_name in (
            "projects",
            "tech_stack",
            "preferences",
            "goals",
            "constraints",
        ):
            previous_values = set(getattr(previous, field_name, []))
            current_values = set(getattr(current, field_name, []))
            added = sorted(current_values - previous_values)
            removed = sorted(previous_values - current_values)
            if added:
                changes.append(f"{field_name}:added:{', '.join(added[:5])}")
            if removed:
                changes.append(f"{field_name}:removed:{', '.join(removed[:5])}")

        if current.summary != previous.summary:
            changes.append("summary_text_updated")

        return changes or ["no_material_change"]

    def _get_recent_conversations(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        cutoff: Optional[datetime] = None,
    ) -> List[Conversation]:
        """Get conversations that should contribute to synthesis."""
        user_conv_ids = self.archive.user_conversations.get(user_id, set())
        conversations: List[Conversation] = []

        for conv_id in user_conv_ids:
            conv = self.archive.conversations.get(conv_id)
            if not conv:
                continue
            if conv.is_deleted or conv.is_incognito:
                continue
            if cutoff is not None and conv.updated_at < cutoff:
                continue
            if project_id and conv.project_id != project_id:
                continue
            if not project_id and conv.project_id:
                continue
            if self.work_focused_filter and not self.category_filter.is_relevant(conv):
                continue
            conversations.append(conv)

        conversations.sort(key=lambda conversation: conversation.updated_at)
        return conversations


class ExplicitMemoryEditProcessor:
    """
    Synchronous immediate memory edit processor.
    Handles "remember X" / "forget Y" commands - applies immediately.
    """

    def __init__(self, embedder: Optional[MemoryEmbedder] = None):
        self.embedder = embedder or MemoryEmbedder()
        self.pending_edits: Dict[str, List[ExplicitMemoryEdit]] = defaultdict(list)
        self.edit_history: Dict[str, List[ExplicitMemoryEdit]] = defaultdict(list)

    def process_edit(
        self,
        user_id: str,
        scope: MemoryScope,
        project_id: Optional[str],
        edit_type: str,
        content: str,
    ) -> ExplicitMemoryEdit:
        """
        Process explicit memory edit immediately.
        This bypasses the 24h synthesis cycle.
        """
        edit = ExplicitMemoryEdit(
            edit_id=str(uuid.uuid4()),
            user_id=user_id,
            scope=scope,
            project_id=project_id,
            edit_type=edit_type,
            content=content,
            timestamp=datetime.now(),
            applied=False,
        )

        edit.applied = True

        self.pending_edits[user_id].append(edit)
        self.edit_history[user_id].append(edit)

        return edit

    def apply_explicit_edit_to_summary(
        self,
        summary: Union[GlobalMemorySummary, ProjectMemorySummary],
        edit: ExplicitMemoryEdit,
    ) -> Union[GlobalMemorySummary, ProjectMemorySummary]:
        """Apply explicit edit to summary text."""
        if edit.edit_type == "add":
            new_content = f"{summary.summary_text}\n\n# Explicit Memory\n{edit.content}"
            summary.summary_text = new_content
            summary.explicit_edits.append(edit.to_dict())

        elif edit.edit_type == "remove":
            summary.summary_text = summary.summary_text.replace(edit.content, "")
            summary.explicit_edits.append(edit.to_dict())

        elif edit.edit_type == "update":
            summary.summary_text = edit.content
            summary.explicit_edits.append(edit.to_dict())

        summary.last_modified = datetime.now()

        return summary

    def get_pending_edits(self, user_id: str) -> List[ExplicitMemoryEdit]:
        """Get all pending edits for a user."""
        return self.pending_edits.get(user_id, [])

    def get_edit_history(self, user_id: str) -> List[ExplicitMemoryEdit]:
        """Return immutable explicit edit history for audit inspection."""
        return list(self.edit_history.get(user_id, []))


# ============================================================================
# Profile and Style Storage
# ============================================================================


class ProfilePreferencesStore:
    """
    Separate profile preferences store — accessible in all scope modes.
    Preferences are never suppressed, even in incognito.
    """

    def __init__(self):
        self.preferences: Dict[str, ProfilePreferences] = {}

    def get(self, user_id: str) -> ProfilePreferences:
        """Get user preferences, creating default if not exists."""
        if user_id not in self.preferences:
            self.preferences[user_id] = ProfilePreferences(user_id=user_id)
        return self.preferences[user_id]

    def update(self, user_id: str, updates: Dict) -> ProfilePreferences:
        """Update user preferences."""
        prefs = self.get(user_id)

        if "communication_style" in updates:
            prefs.communication_style = updates["communication_style"]
        if "technical_expertise" in updates:
            prefs.technical_expertise = updates["technical_expertise"]
        if "response_length" in updates:
            prefs.response_length = updates["response_length"]
        if "code_style" in updates:
            prefs.code_style = updates["code_style"]
        if "preferred_languages" in updates:
            prefs.preferred_languages = updates["preferred_languages"]
        if "custom_instructions" in updates:
            prefs.custom_instructions = updates["custom_instructions"]

        prefs.last_modified = datetime.now()

        return prefs

    def to_xml(self, user_id: str) -> str:
        """Get preferences as XML."""
        prefs = self.get(user_id)
        return prefs.to_xml()


class StylesStore:
    """User communication styles."""

    def __init__(self):
        self.styles: Dict[str, UserStyle] = {}
        self.user_styles: Dict[str, Set[str]] = defaultdict(set)

    def create_style(
        self,
        user_id: str,
        name: str,
        description: str,
        system_prompt_additions: str = "",
    ) -> UserStyle:
        """Create a new user style."""
        style = UserStyle(
            style_id=str(uuid.uuid4()),
            user_id=user_id,
            name=name,
            description=description,
            system_prompt_additions=system_prompt_additions,
            created_at=datetime.now(),
        )

        self.styles[style.style_id] = style
        self.user_styles[user_id].add(style.style_id)

        return style

    def get_styles(self, user_id: str) -> List[UserStyle]:
        """Get all styles for a user."""
        style_ids = self.user_styles.get(user_id, set())
        return [self.styles[sid] for sid in style_ids if sid in self.styles]

    def add_style(self, user_id: str, style: UserStyle) -> None:
        """Add a pre-constructed style for a user."""
        self.styles[style.style_id] = style
        self.user_styles[user_id].add(style.style_id)


# ============================================================================
# Memory Control Panel
# ============================================================================


class MemoryControlPanel:
    """
    User-facing memory controls: enable/disable/pause/reset.
    Maps to Settings > Capabilities UI.
    """

    def __init__(self):
        self.user_settings: Dict[str, MemoryControlSettings] = {}

    def get_settings(self, user_id: str) -> MemoryControlSettings:
        """Get user memory settings."""
        if user_id not in self.user_settings:
            self.user_settings[user_id] = MemoryControlSettings(user_id=user_id)
        return self.user_settings[user_id]

    def enable(self, user_id: str) -> MemoryControlSettings:
        """Enable memory."""
        settings = self.get_settings(user_id)
        settings.state = MemoryState.ON
        return settings

    def disable(self, user_id: str) -> MemoryControlSettings:
        """Disable memory (pause)."""
        settings = self.get_settings(user_id)
        settings.state = MemoryState.PAUSED
        return settings

    def reset(self, user_id: str) -> MemoryControlSettings:
        """Reset all memory for user."""
        settings = self.get_settings(user_id)
        settings.state = MemoryState.OFF
        settings.last_reset = datetime.now()
        return settings

    def set_chat_search_enabled(
        self, user_id: str, enabled: bool
    ) -> MemoryControlSettings:
        """Enable/disable chat search."""
        settings = self.get_settings(user_id)
        settings.enable_chat_search = enabled
        return settings

    def can_use_chat_search(self, user_id: str, **kwargs: Any) -> bool:
        """Check if user has chat search enabled in their settings."""
        settings = self.get_settings(user_id)
        return settings.enable_chat_search


# ============================================================================
# Scope Validation
# ============================================================================


class MemoryScopeValidator:
    """Mechanical scope validator for all memory reads and writes."""

    @staticmethod
    def validate_write(scope: MemoryScope, is_incognito: bool) -> bool:
        """Return True when a write is allowed for the provided scope."""
        if is_incognito:
            return False
        if scope == MemoryScope.INCOGNITO:
            return False
        return True

    @staticmethod
    def assert_write_allowed(scope: MemoryScope, is_incognito: bool) -> None:
        """Raise when a caller attempts to persist memory from an incognito scope."""
        if not MemoryScopeValidator.validate_write(scope, is_incognito):
            raise PermissionError(
                f"Memory write blocked: scope={scope.value}, is_incognito={is_incognito}"
            )


# ============================================================================
# Deletion Propagation Service
# ============================================================================


class DeletionPropagationService:
    """
    Handles conversation deletion -> synthesis refresh propagation.
    When a conversation is deleted, triggers synthesis refresh.
    Includes timestamp-based dedup to coalesce rapid multi-delete bursts.
    """

    DEDUP_WINDOW_SECONDS: float = 2.0

    def __init__(
        self, archive: ConversationArchive, synthesizer: SummaryRefreshSynthesizer
    ):
        self.archive = archive
        self.synthesizer = synthesizer
        self.pending_refreshes: Dict[str, Set[str]] = defaultdict(set)
        self.pending_project_refreshes: Dict[str, Set[str]] = defaultdict(set)
        self._deletion_timestamps: Dict[str, float] = {}

    def mark_deleted(self, conversation_id: str) -> bool:
        """Mark conversation as deleted and schedule refresh with dedup guard."""
        now = time.monotonic()
        last_ts = self._deletion_timestamps.get(conversation_id)
        if last_ts is not None and (now - last_ts) < self.DEDUP_WINDOW_SECONDS:
            return False

        if not self.archive.delete_conversation(conversation_id):
            return False

        self._deletion_timestamps[conversation_id] = now
        self._prune_stale_timestamps(now)

        conv = self.archive.get_conversation(conversation_id)
        if conv:
            self.pending_refreshes[conv.user_id].add(conversation_id)
            if conv.project_id:
                self.pending_project_refreshes[conv.user_id].add(conv.project_id)

        return True

    def _prune_stale_timestamps(self, now: float) -> None:
        """Remove deletion timestamps older than 10x the dedup window."""
        cutoff = now - (self.DEDUP_WINDOW_SECONDS * 10)
        stale_keys = [
            k for k, ts in self._deletion_timestamps.items() if ts < cutoff
        ]
        for k in stale_keys:
            del self._deletion_timestamps[k]

    def should_refresh(self, user_id: str) -> bool:
        """Check if user needs synthesis refresh."""
        return bool(self.pending_refreshes.get(user_id))

    def clear_pending(self, user_id: str) -> None:
        """Clear pending refresh after synthesis completes."""
        self.pending_refreshes.pop(user_id, None)
        self.pending_project_refreshes.pop(user_id, None)

    def consume_refresh_requests(
        self, user_id: str
    ) -> List[Tuple[MemoryScope, Optional[str]]]:
        """
        Consume pending refresh requests and return the exact scopes that need
        targeted re-synthesis.
        """
        if not self.should_refresh(user_id):
            return []

        project_ids = sorted(self.pending_project_refreshes.get(user_id, set()))
        requests: List[Tuple[MemoryScope, Optional[str]]] = [
            (MemoryScope.PROJECT, project_id) for project_id in project_ids
        ]

        requests.append((MemoryScope.GLOBAL, None))
        self.clear_pending(user_id)
        return requests


# ============================================================================
# Import Normalizer
# ============================================================================


class ImportNormalizer:
    """
    Decomposes pasted text from other AI providers into memory edits.
    Experimental as of Mar 2026.
    """

    def __init__(self, edit_processor: ExplicitMemoryEditProcessor):
        self.edit_processor = edit_processor

    async def import_from_text(
        self,
        user_id: str,
        raw_text: str,
        scope: MemoryScope = MemoryScope.GLOBAL,
        project_id: Optional[str] = None,
    ) -> List[ExplicitMemoryEdit]:
        """
        Import memory from external text (e.g., from a third-party AI export).
        Prefers structured JSON payloads and falls back to heuristic category extraction.
        """
        MemoryScopeValidator.assert_write_allowed(
            scope=scope,
            is_incognito=(scope == MemoryScope.INCOGNITO),
        )

        structured_payload = self._try_parse_structured_payload(raw_text)
        if structured_payload is not None:
            return self._import_from_payload(
                user_id=user_id,
                payload=structured_payload,
                scope=scope,
                project_id=project_id,
            )

        categories = self._extract_categories(raw_text)

        edits = []
        for category, content in categories.items():
            edit = self.edit_processor.process_edit(
                user_id=user_id,
                scope=scope,
                project_id=project_id,
                edit_type="add",
                content=f"[{category}] {content}",
            )
            edits.append(edit)

        return edits

    def _import_from_payload(
        self,
        user_id: str,
        payload: Dict[str, Any],
        scope: MemoryScope,
        project_id: Optional[str],
    ) -> List[ExplicitMemoryEdit]:
        """Convert a structured payload into explicit memory edits."""
        edits: List[ExplicitMemoryEdit] = []
        structured = payload.get("structured", {})
        import_sections: Dict[str, List[str]] = {
            "Summary": [payload.get("summary", "")],
            "Role": [structured.get("role", "")],
            "Projects": structured.get("projects", []),
            "Tech Stack": structured.get("tech_stack", []),
            "Preferences": structured.get("preferences", []),
            "Goals": structured.get("goals", []),
            "Constraints": structured.get("constraints", []),
        }

        for category, items in import_sections.items():
            for item in items:
                cleaned = str(item).strip()
                if not cleaned:
                    continue
                edits.append(
                    self.edit_processor.process_edit(
                        user_id=user_id,
                        scope=scope,
                        project_id=project_id,
                        edit_type="add",
                        content=f"[{category}] {cleaned}",
                    )
                )

        return edits

    def _try_parse_structured_payload(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Parse structured JSON imports, tolerating fenced code blocks."""
        stripped = raw_text.strip()
        if not stripped:
            return None

        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
            stripped = re.sub(r"```$", "", stripped).strip()

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None

        if not isinstance(parsed, dict):
            return None

        if "summary" not in parsed and "structured" not in parsed:
            return None

        return parsed

    def _extract_categories(self, text: str) -> Dict[str, str]:
        """Extract memory categories from text."""
        categories = {
            "Instructions": [],
            "Identity": [],
            "Projects": [],
            "Technical Preferences": [],
            "Other Context": [],
        }

        current_category = "Other Context"

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            line_lower = line.lower()

            if any(
                kw in line_lower
                for kw in ["instruction", "tone", "format", "style", "always", "never"]
            ):
                current_category = "Instructions"
            elif any(
                kw in line_lower
                for kw in ["name", "location", "job", "family", "personal"]
            ):
                current_category = "Identity"
            elif any(kw in line_lower for kw in ["project", "goal", "working on"]):
                current_category = "Projects"
            elif any(
                kw in line_lower
                for kw in ["python", "javascript", "language", "framework", "tool"]
            ):
                current_category = "Technical Preferences"

            if line.startswith("-") or line.startswith("*"):
                categories[current_category].append(line[1:].strip())

        result = {}
        for cat, items in categories.items():
            if items:
                result[cat] = " ".join(items[:5])

        return result


# ============================================================================
# Enterprise Infrastructure — Observability, Resilience, Security
# ============================================================================


# --- Correlation Context ---

_correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


@contextmanager
def correlation_scope(correlation_id: Optional[str] = None):
    """Context manager that binds a correlation ID for the current scope."""
    cid = correlation_id or f"cor_{uuid.uuid4().hex[:12]}"
    token = _correlation_id_var.set(cid)
    try:
        yield cid
    finally:
        _correlation_id_var.reset(token)


def get_correlation_id() -> str:
    """Return the active correlation ID or generate an ephemeral one."""
    cid = _correlation_id_var.get("")
    return cid if cid else f"ephemeral_{uuid.uuid4().hex[:8]}"


# --- Structured Logger ---


class StructuredLogger:
    """JSON-structured logger with automatic correlation ID injection."""

    def __init__(self, name: str, level: int = logging.INFO):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self.name = name

    def _emit(self, level: str, event: str, **kwargs: Any) -> None:
        entry: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "logger": self.name,
            "correlation_id": get_correlation_id(),
            "event": event,
        }
        entry.update({k: v for k, v in kwargs.items() if v is not None})
        for sensitive_key in ("api_key", "token", "secret", "password", "credential"):
            if sensitive_key in entry:
                entry[sensitive_key] = "***REDACTED***"
        self._logger.log(
            getattr(logging, level.upper(), logging.INFO),
            json.dumps(entry, default=str),
        )

    def info(self, event: str, **kwargs: Any) -> None:
        self._emit("INFO", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._emit("WARNING", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._emit("ERROR", event, **kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._emit("DEBUG", event, **kwargs)


# --- Memory Event Bus ---

MemoryEventHandler = Callable[[MemoryEvent, Dict[str, Any]], None]


class MemoryEventBus:
    """In-process event bus for memory system observability and integration."""

    def __init__(self) -> None:
        self._handlers: Dict[MemoryEvent, List[MemoryEventHandler]] = defaultdict(list)
        self._global_handlers: List[MemoryEventHandler] = []

    def subscribe(self, event: MemoryEvent, handler: MemoryEventHandler) -> None:
        """Subscribe to a specific event type."""
        self._handlers[event].append(handler)

    def subscribe_all(self, handler: MemoryEventHandler) -> None:
        """Subscribe to all events (global listener)."""
        self._global_handlers.append(handler)

    def emit(self, event: MemoryEvent, data: Optional[Dict[str, Any]] = None) -> None:
        """Emit an event to all registered handlers. Handlers must not raise."""
        payload = data or {}
        payload["correlation_id"] = get_correlation_id()
        payload["timestamp"] = datetime.utcnow().isoformat()
        for handler in self._handlers.get(event, []):
            try:
                handler(event, payload)
            except Exception:
                pass
        for handler in self._global_handlers:
            try:
                handler(event, payload)
            except Exception:
                pass


# --- Metrics ---


class MemoryMetrics:
    """Atomic counters and histogram summaries for memory system telemetry."""

    def __init__(self) -> None:
        self._counters: Dict[str, int] = defaultdict(int)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def increment(self, metric: str, delta: int = 1) -> None:
        """Increment a counter metric."""
        self._counters[metric] += delta

    def record_duration(self, metric: str, duration_ms: float) -> None:
        """Record a duration sample in a histogram."""
        self._histograms[metric].append(duration_ms)
        if len(self._histograms[metric]) > 10000:
            self._histograms[metric] = self._histograms[metric][-5000:]

    def get_counter(self, metric: str) -> int:
        """Return the current counter value."""
        return self._counters.get(metric, 0)

    def get_histogram_summary(self, metric: str) -> Dict[str, float]:
        """Return p50/p95/p99/mean for a histogram metric."""
        values = self._histograms.get(metric, [])
        if not values:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        return {
            "count": n,
            "mean": sum(sorted_vals) / n,
            "p50": sorted_vals[n // 2],
            "p95": sorted_vals[min(int(n * 0.95), n - 1)],
            "p99": sorted_vals[min(int(n * 0.99), n - 1)],
        }

    def snapshot(self) -> Dict[str, Any]:
        """Return a full metrics snapshot for health endpoints."""
        return {
            "counters": dict(self._counters),
            "histograms": {
                k: self.get_histogram_summary(k) for k in self._histograms
            },
        }


# --- Health Check ---


@dataclass
class ComponentHealth:
    """Health status of a single memory subsystem component."""

    name: str
    healthy: bool
    message: str = ""
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class HealthCheck:
    """Composite health check aggregator for the memory system."""

    def __init__(
        self,
        memory_store: MemoryStore,
        archive: ConversationArchive,
        control_panel: MemoryControlPanel,
        metrics: Optional[MemoryMetrics] = None,
    ):
        self.memory_store = memory_store
        self.archive = archive
        self.control_panel = control_panel
        self.metrics = metrics

    def check(self) -> Dict[str, Any]:
        """Run all component checks and return aggregate health status."""
        components = [
            self._check_memory_store(),
            self._check_archive(),
            self._check_control_panel(),
        ]
        overall = all(c.healthy for c in components)
        result = {
            "status": "healthy" if overall else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "components": [
                {
                    "name": c.name,
                    "healthy": c.healthy,
                    "message": c.message,
                    "latency_ms": c.latency_ms,
                    "details": c.details,
                }
                for c in components
            ],
        }
        if self.metrics:
            result["metrics_snapshot"] = self.metrics.snapshot()
        return result

    def _check_memory_store(self) -> ComponentHealth:
        start = time.monotonic()
        try:
            count = len(self.memory_store.memories)
            return ComponentHealth(
                name="memory_store",
                healthy=True,
                message=f"{count} items stored",
                latency_ms=(time.monotonic() - start) * 1000,
                details={
                    "item_count": count,
                    "backend": self.memory_store.storage_backend,
                },
            )
        except Exception as exc:
            return ComponentHealth(
                name="memory_store",
                healthy=False,
                message=str(exc),
                latency_ms=(time.monotonic() - start) * 1000,
            )

    def _check_archive(self) -> ComponentHealth:
        start = time.monotonic()
        try:
            count = len(self.archive.conversations)
            return ComponentHealth(
                name="conversation_archive",
                healthy=True,
                message=f"{count} conversations archived",
                latency_ms=(time.monotonic() - start) * 1000,
                details={
                    "conversation_count": count,
                    "embedding_count": len(self.archive.conversation_embeddings),
                    "backend": self.archive.vector_backend,
                },
            )
        except Exception as exc:
            return ComponentHealth(
                name="conversation_archive",
                healthy=False,
                message=str(exc),
                latency_ms=(time.monotonic() - start) * 1000,
            )

    def _check_control_panel(self) -> ComponentHealth:
        start = time.monotonic()
        try:
            user_count = len(self.control_panel.user_settings)
            return ComponentHealth(
                name="control_panel",
                healthy=True,
                message=f"{user_count} configured users",
                latency_ms=(time.monotonic() - start) * 1000,
                details={"user_count": user_count},
            )
        except Exception as exc:
            return ComponentHealth(
                name="control_panel",
                healthy=False,
                message=str(exc),
                latency_ms=(time.monotonic() - start) * 1000,
            )


# --- Circuit Breaker ---


class CircuitBreaker:
    """
    Circuit breaker for external/pluggable backend calls.
    Prevents cascading failures when an upstream backend degrades.

    States:
      CLOSED  → normal operation; failures counted
      OPEN    → calls blocked; waits for recovery_timeout
      HALF_OPEN → allows limited probe calls to test recovery
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
        half_open_max_calls: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Return the resolved state, promoting OPEN → HALF_OPEN after timeout."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute a function through the circuit breaker gate."""
        async with self._lock:
            current_state = self.state
            if current_state == CircuitState.OPEN:
                raise RuntimeError(
                    f"CircuitBreaker '{self.name}' is OPEN — backend unavailable"
                )
            if (
                current_state == CircuitState.HALF_OPEN
                and self._half_open_calls >= self.half_open_max_calls
            ):
                raise RuntimeError(
                    f"CircuitBreaker '{self.name}' is HALF_OPEN — max probe calls reached"
                )
            if current_state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as exc:
            await self._record_failure()
            raise exc

    async def _record_success(self) -> None:
        async with self._lock:
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED

    async def _record_failure(self) -> None:
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN


# --- Rate Limiter ---


class TokenBucketRateLimiter:
    """Token-bucket rate limiter for API entry points."""

    def __init__(
        self,
        max_tokens: int = 100,
        refill_rate: float = 10.0,
    ):
        self._max_tokens = max_tokens
        self._refill_rate = refill_rate
        self._tokens = float(max_tokens)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens. Returns True if allowed."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._max_tokens,
                self._tokens + elapsed * self._refill_rate,
            )
            self._last_refill = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def reset(self) -> None:
        """Reset to full capacity."""
        self._tokens = float(self._max_tokens)
        self._last_refill = time.monotonic()


# --- Encryption at Rest ---


class EncryptionProvider(Protocol):
    """Protocol for at-rest encryption of memory content."""

    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext and return a ciphertext string."""
        ...

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt ciphertext and return the original plaintext."""
        ...


class NoOpEncryptionProvider:
    """Passthrough provider — no encryption. Default for backward compatibility."""

    def encrypt(self, plaintext: str) -> str:
        return plaintext

    def decrypt(self, ciphertext: str) -> str:
        return ciphertext


class HMACEncryptionProvider:
    """
    HMAC-authenticated XOR stream cipher for local at-rest protection.
    Uses HMAC-SHA256 for integrity verification and a derived keystream
    for confidentiality. No external dependencies required.

    Not suitable as a replacement for AES/Fernet in high-security contexts;
    intended for local-only memory protection against casual inspection.
    """

    def __init__(self, key: Optional[bytes] = None):
        self._key = key or secrets.token_bytes(32)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt and return URL-safe base64 with integrity tag."""
        nonce = secrets.token_bytes(16)
        plaintext_bytes = plaintext.encode("utf-8")
        stream = self._derive_stream(nonce, len(plaintext_bytes))
        ciphertext = bytes(a ^ b for a, b in zip(plaintext_bytes, stream))
        mac = hmac_mod.new(self._key, nonce + ciphertext, hashlib.sha256).digest()
        combined = nonce + ciphertext + mac
        return base64.urlsafe_b64encode(combined).decode("ascii")

    def decrypt(self, ciphertext_b64: str) -> str:
        """Decrypt and verify integrity. Raises ValueError on tamper."""
        combined = base64.urlsafe_b64decode(ciphertext_b64.encode("ascii"))
        if len(combined) < 48:
            raise ValueError("Invalid ciphertext: too short")
        nonce = combined[:16]
        mac = combined[-32:]
        ciphertext = combined[16:-32]
        expected_mac = hmac_mod.new(
            self._key, nonce + ciphertext, hashlib.sha256
        ).digest()
        if not hmac_mod.compare_digest(mac, expected_mac):
            raise ValueError("Decryption failed: integrity check failure")
        stream = self._derive_stream(nonce, len(ciphertext))
        plaintext_bytes = bytes(a ^ b for a, b in zip(ciphertext, stream))
        return plaintext_bytes.decode("utf-8")

    def _derive_stream(self, nonce: bytes, length: int) -> bytes:
        """Derive a pseudo-random byte stream from nonce and key via counter mode."""
        blocks: List[bytes] = []
        counter = 0
        while len(b"".join(blocks)) < length:
            block_input = self._key + nonce + counter.to_bytes(4, "big")
            blocks.append(hashlib.sha256(block_input).digest())
            counter += 1
        return b"".join(blocks)[:length]


# --- TTL / Eviction / Lifecycle ---


@dataclass
class EvictionPolicy:
    """Configuration for memory eviction and lifecycle management."""

    max_memories_per_user: int = 10000
    max_conversations_per_user: int = 5000
    memory_ttl_days: int = 30
    conversation_ttl_days: int = 90
    low_confidence_threshold: float = 0.15
    low_access_threshold: int = 0
    eviction_batch_size: int = 100


class MemoryLifecycleManager:
    """
    Manages memory TTL, eviction, and bounded resource usage.
    Runs eviction sweeps to keep memory stores within capacity limits.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        archive: ConversationArchive,
        policy: Optional[EvictionPolicy] = None,
        metrics: Optional[MemoryMetrics] = None,
        event_bus: Optional[MemoryEventBus] = None,
        slog: Optional[StructuredLogger] = None,
    ):
        self.memory_store = memory_store
        self.archive = archive
        self.policy = policy or EvictionPolicy()
        self.metrics = metrics
        self.event_bus = event_bus
        self.slog = slog or StructuredLogger("memory.lifecycle")

    async def run_eviction(self, user_id: str) -> Dict[str, int]:
        """Evict expired and low-value memories for a user."""
        evicted_memories = 0
        evicted_conversations = 0
        now = datetime.now()

        memory_ids = list(self.memory_store.user_memories.get(user_id, set()))
        expired_memory_ids: List[str] = []
        for mid in memory_ids:
            memory = self.memory_store.memories.get(mid)
            if not memory:
                continue
            age_days = (now - memory.created_at).days
            if age_days > self.policy.memory_ttl_days:
                expired_memory_ids.append(mid)
            elif (
                memory.confidence < self.policy.low_confidence_threshold
                and memory.access_count <= self.policy.low_access_threshold
                and age_days > 30
            ):
                expired_memory_ids.append(mid)

        for mid in expired_memory_ids[: self.policy.eviction_batch_size]:
            await self.memory_store.delete_memory(mid)
            evicted_memories += 1

        remaining_ids = sorted(
            self.memory_store.user_memories.get(user_id, set()),
            key=lambda mid: (
                self.memory_store.memories[mid].last_accessed
                if mid in self.memory_store.memories
                else datetime.min
            ),
        )
        if len(remaining_ids) > self.policy.max_memories_per_user:
            overshoot = len(remaining_ids) - self.policy.max_memories_per_user
            for mid in remaining_ids[:overshoot]:
                await self.memory_store.delete_memory(mid)
                evicted_memories += 1

        conv_ids = list(self.archive.user_conversations.get(user_id, set()))
        for cid in conv_ids:
            conv = self.archive.conversations.get(cid)
            if not conv:
                continue
            age_days = (now - conv.updated_at).days
            if age_days > self.policy.conversation_ttl_days:
                self.archive.delete_conversation(cid)
                evicted_conversations += 1

        remaining_convs = sorted(
            self.archive.user_conversations.get(user_id, set()),
            key=lambda cid: (
                self.archive.conversations[cid].updated_at
                if cid in self.archive.conversations
                else datetime.min
            ),
        )
        if len(remaining_convs) > self.policy.max_conversations_per_user:
            overshoot = len(remaining_convs) - self.policy.max_conversations_per_user
            for cid in remaining_convs[:overshoot]:
                self.archive.delete_conversation(cid)
                evicted_conversations += 1

        result = {
            "memories_evicted": evicted_memories,
            "conversations_evicted": evicted_conversations,
        }

        self.slog.info(
            "eviction_completed",
            user_id=user_id,
            memories_evicted=evicted_memories,
            conversations_evicted=evicted_conversations,
        )
        return result


# --- Tool Definitions (function-calling compatible schemas) ---


class MemoryToolDefinitions:
    """
    Function-calling tool definitions for memory system operations.
    Schemas injected into the system prompt for the assistant to invoke.
    """

    @staticmethod
    def past_chat_search_tool() -> Dict[str, Any]:
        """Tool schema for searching past conversations."""
        return {
            "type": "function",
            "function": {
                "name": "search_past_chats",
                "description": (
                    "Search the user's past conversation history for relevant "
                    "context. Returns excerpts from previous chats that match "
                    "the query."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "The search query to find relevant past conversations."
                            ),
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results to return (1-10).",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    @staticmethod
    def recent_chats_tool() -> Dict[str, Any]:
        """Tool schema for getting recent conversations."""
        return {
            "type": "function",
            "function": {
                "name": "get_recent_chats",
                "description": (
                    "Get the user's most recent conversation titles and excerpts. "
                    "Useful for context continuity across sessions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "count": {
                            "type": "integer",
                            "description": "Number of recent chats (1-20).",
                            "default": 3,
                            "minimum": 1,
                            "maximum": 20,
                        },
                        "sort_order": {
                            "type": "string",
                            "enum": ["asc", "desc"],
                            "default": "desc",
                            "description": "Sort order by date.",
                        },
                    },
                    "required": [],
                },
            },
        }

    @staticmethod
    def memory_manage_tool() -> Dict[str, Any]:
        """Tool schema for explicit memory management (remember/forget)."""
        return {
            "type": "function",
            "function": {
                "name": "manage_memory",
                "description": (
                    "Add, update, or remove a specific memory about the user. "
                    "Use when the user explicitly asks to remember or forget something."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add", "update", "remove"],
                            "description": "The memory operation to perform.",
                        },
                        "content": {
                            "type": "string",
                            "description": "The memory content to add/update/remove.",
                        },
                    },
                    "required": ["action", "content"],
                },
            },
        }

    @classmethod
    def all_tools(cls, **kwargs: Any) -> List[Dict[str, Any]]:
        """Return all available tool schemas. All capabilities are always enabled."""
        return [
            cls.memory_manage_tool(),
            cls.past_chat_search_tool(),
            cls.recent_chats_tool(),
        ]


# ============================================================================
# Prompt Context Assembly and Hybrid Retrieval
# ============================================================================


class ProjectInstructionStore:
    """Static per-project instruction store."""

    def __init__(self):
        self.instructions: Dict[str, str] = {}

    def set(self, project_id: str, content: str) -> None:
        """Store instruction text for a project."""
        self.instructions[project_id] = content

    def get(self, project_id: str) -> str:
        """Return project instruction text if present."""
        return self.instructions.get(project_id, "")

    def delete(self, project_id: str) -> None:
        """Delete stored project instruction text."""
        self.instructions.pop(project_id, None)


class PlanGate:
    """Capability gates — all features unconditionally enabled."""

    PAID_TIERS = {UserTier.PRO, UserTier.MAX, UserTier.TEAM, UserTier.ENTERPRISE}  # Legacy ref, not used for gating

    @classmethod
    def can_search(cls, user_tier: UserTier) -> bool:
        """All users can search archived conversations."""
        return True

    @classmethod
    def allows_archive_injection(
        cls,
        user_tier: UserTier,
        chat_search_enabled: bool,
    ) -> bool:
        """Archive injection requires settings opt-in only."""
        return chat_search_enabled


class ContextBudgetManager:
    """Per-layer context budgeting with deterministic token estimation."""

    DEFAULT_BUDGETS = {
        "summary": 800,
        "preferences": 300,
        "project_instructions": 300,
        "styles": 200,
        "semantic": 600,
        "archive": 400,
    }

    def __init__(self, layer_budgets: Optional[Dict[str, int]] = None):
        self.layer_budgets = dict(self.DEFAULT_BUDGETS)
        if layer_budgets:
            self.layer_budgets.update(layer_budgets)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using a stable chars-per-token heuristic."""
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4))

    def apply_budget(self, layer: str, text: str, budget: Optional[int] = None) -> Tuple[str, int]:
        """Trim a layer to its configured or supplied budget."""
        resolved_budget = self.layer_budgets.get(layer, 0) if budget is None else budget
        if not text or resolved_budget <= 0:
            return "", 0

        current_tokens = self.estimate_tokens(text)
        if current_tokens <= resolved_budget:
            return text, current_tokens

        return self._truncate_text(text, resolved_budget)

    def allocate_layer_budgets(
        self,
        layer_texts: Dict[str, str],
        total_budget_tokens: Optional[int] = None,
    ) -> Dict[str, int]:
        """Allocate budgets proportionally across only the active layers."""
        active_layers = {
            layer: text
            for layer, text in layer_texts.items()
            if text and self.layer_budgets.get(layer, 0) > 0
        }
        if not active_layers:
            return {}

        if total_budget_tokens is None:
            total_budget_tokens = sum(self.layer_budgets.values())

        requested = {
            layer: min(self.estimate_tokens(text), self.layer_budgets.get(layer, 0))
            for layer, text in active_layers.items()
        }
        requested_total = sum(requested.values())
        if requested_total <= total_budget_tokens:
            return requested

        floors = {
            layer: min(requested[layer], max(12, math.floor(self.layer_budgets.get(layer, 0) * 0.20)))
            for layer in requested
        }
        allocations = dict(floors)
        remaining_budget = max(total_budget_tokens - sum(allocations.values()), 0)

        extra_need = {
            layer: max(requested[layer] - allocations[layer], 0)
            for layer in requested
        }
        total_extra_need = sum(extra_need.values())
        if total_extra_need <= 0:
            return allocations

        for layer, need in sorted(
            extra_need.items(),
            key=lambda item: self.layer_budgets.get(item[0], 0),
            reverse=True,
        ):
            if remaining_budget <= 0:
                break
            share = math.floor((need / total_extra_need) * remaining_budget)
            granted = min(need, share if share > 0 else 1, remaining_budget)
            allocations[layer] += granted
            remaining_budget -= granted

        if remaining_budget > 0:
            for layer, need in sorted(
                extra_need.items(),
                key=lambda item: self.layer_budgets.get(item[0], 0),
                reverse=True,
            ):
                if remaining_budget <= 0:
                    break
                outstanding = requested[layer] - allocations[layer]
                if outstanding <= 0:
                    continue
                granted = min(outstanding, remaining_budget)
                allocations[layer] += granted
                remaining_budget -= granted

        return allocations

    def apply_global_budget(
        self,
        layer_texts: Dict[str, str],
        total_budget_tokens: Optional[int] = None,
    ) -> Tuple[Dict[str, str], Dict[str, int]]:
        """Apply a whole-context budget with proportional layer rebalancing."""
        allocations = self.allocate_layer_budgets(
            layer_texts=layer_texts,
            total_budget_tokens=total_budget_tokens,
        )
        trimmed: Dict[str, str] = {}
        used_tokens: Dict[str, int] = {}
        for layer, text in layer_texts.items():
            budget = allocations.get(layer, 0)
            trimmed[layer], used_tokens[layer] = self.apply_budget(layer, text, budget=budget)
        return trimmed, used_tokens

    def _truncate_text(self, text: str, budget: int) -> Tuple[str, int]:
        """Truncate plain text or XML while preserving wrapper tags."""
        xml_match = re.match(
            r"^\s*(<(?P<tag>[A-Za-z][A-Za-z0-9]*)>)\s*(?P<body>.*)\s*(</(?P=tag)>)\s*$",
            text,
            re.DOTALL,
        )
        if xml_match:
            opening = xml_match.group(1)
            closing = xml_match.group(4)
            body = xml_match.group("body")
            body_budget = max(
                budget - self.estimate_tokens(opening) - self.estimate_tokens(closing) - 3,
                1,
            )
            truncated_body, _ = self._truncate_plain_text(body, body_budget)
            wrapped = f"{opening}\n{truncated_body}\n{closing}"
            return wrapped, self.estimate_tokens(wrapped)

        return self._truncate_plain_text(text, budget)

    def _truncate_plain_text(self, text: str, budget: int) -> Tuple[str, int]:
        """Trim plain text without splitting mid-sentence when avoidable."""
        max_chars = max(1, budget * 4)
        truncated = text[:max_chars].rstrip()
        sentence_break = max(
            truncated.rfind(". "),
            truncated.rfind("\n"),
            truncated.rfind("; "),
            truncated.rfind(", "),
        )
        if sentence_break > 0:
            truncated = truncated[: sentence_break + 1].rstrip()
        if truncated != text:
            truncated = truncated.rstrip() + "\n... (truncated)"
        return truncated, self.estimate_tokens(truncated)


class AtomicMemoryRetriever:
    """
    Legacy semantic retrieval layer, demoted from primary injector to subordinate
    data source used by the unified prompt assembler.
    """

    def __init__(self, memory_store: MemoryStore, max_context_tokens: int = 2000):
        self.memory_store = memory_store
        self.max_context_tokens = max_context_tokens

    async def retrieve(
        self,
        user_id: str,
        current_query: str,
        conversation_context: Optional[ConversationContext] = None,
    ) -> MemoryRetrievalResult:
        """Return atomic retrieval results without directly injecting them."""
        return await self.memory_store.retrieve_memories(
            user_id=user_id,
            query=current_query,
            top_k=15,
            min_importance=0.3,
        )

    async def build_context_block(
        self,
        user_id: str,
        current_query: str,
        conversation_context: Optional[ConversationContext] = None,
    ) -> str:
        """Build memory context block for injection."""
        if not user_id or not current_query:
            return ""

        retrieval_result = await self.retrieve(
            user_id=user_id,
            current_query=current_query,
            conversation_context=conversation_context,
        )

        if not retrieval_result.memories:
            return ""

        memories_by_type = defaultdict(list)
        for memory, score in zip(
            retrieval_result.memories, retrieval_result.relevance_scores
        ):
            memories_by_type[memory.memory_type].append((memory, score))

        context_parts = ["<semanticMemories>"]

        if MemoryType.PREFERENCE in memories_by_type:
            context_parts.append("\n## Preferences")
            for memory, score in memories_by_type[MemoryType.PREFERENCE][:3]:
                context_parts.append(f"- {memory.content}")

        if MemoryType.FACT in memories_by_type:
            context_parts.append("\n## Facts")
            for memory, score in memories_by_type[MemoryType.FACT][:5]:
                context_parts.append(f"- {memory.content}")

        if MemoryType.ENTITY in memories_by_type:
            context_parts.append("\n## Entities")
            for memory, score in memories_by_type[MemoryType.ENTITY][:3]:
                context_parts.append(f"- {memory.content}")

        if MemoryType.TOPIC in memories_by_type:
            topics = [m.content for m, s in memories_by_type[MemoryType.TOPIC][:5]]
            context_parts.append(f"\n## Topics: {', '.join(topics)}")

        context_parts.append("\n</semanticMemories>")

        context_block = "\n".join(context_parts)

        context_block = self._truncate_to_token_limit(
            context_block, self.max_context_tokens
        )

        return context_block

    def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to token limit."""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]
        return truncated + "\n... (truncated)\n</semanticMemories>"


class HybridMemoryRetriever:
    """Unified retrieval across summary, atomic memory, and archive layers."""

    def __init__(
        self,
        memory_store: MemoryStore,
        atomic_retriever: AtomicMemoryRetriever,
        archive: ConversationArchive,
        budget_manager: Optional[ContextBudgetManager] = None,
    ):
        self.memory_store = memory_store
        self.atomic_retriever = atomic_retriever
        self.archive = archive
        self.budget_manager = budget_manager or ContextBudgetManager()

    async def retrieve(
        self,
        user_id: str,
        query: str,
        scope: MemoryScope,
        project_id: Optional[str],
        global_summary: Optional[GlobalMemorySummary],
        project_summary: Optional[ProjectMemorySummary],
        total_budget_tokens: int = 1800,
        **kwargs: Any,
    ) -> HybridRetrievalResult:
        """Return unified retrieval results with deduplication and budget control."""
        if scope == MemoryScope.INCOGNITO:
            return HybridRetrievalResult()

        summary = project_summary if scope == MemoryScope.PROJECT else global_summary
        results: List[UnifiedMemoryResult] = []

        if summary and summary.summary_text:
            results.append(
                UnifiedMemoryResult(
                    content=summary.summary_text,
                    source_layer="summary",
                    source_id=f"summary:{project_id or user_id}",
                    relevance_score=1.0,
                    confidence=summary.confidence_score,
                )
            )

        atomic_result = await self.atomic_retriever.retrieve(user_id, query)
        for memory, score in zip(atomic_result.memories, atomic_result.relevance_scores):
            results.append(
                UnifiedMemoryResult(
                    content=memory.content,
                    source_layer="atomic",
                    source_id=memory.id,
                    relevance_score=score,
                    confidence=memory.confidence,
                    memory_type=memory.memory_type,
                )
            )

        # Archive search — always available (no tier gating)
        archive_results = await self.archive.search(
            user_id=user_id,
            query=query,
            max_results=5,
            project_id=project_id,
        )
        for reference in archive_results:
            results.append(
                UnifiedMemoryResult(
                    content=f"{reference.human_excerpt}\n{reference.assistant_excerpt}",
                    source_layer="archive",
                    source_id=reference.conversation_id,
                    relevance_score=reference.relevance_score,
                    confidence=1.0,
                    citation_url=reference.chat_url,
                )
            )

        deduped = self._deduplicate(results)
        budgeted = self._apply_budget(deduped, total_budget_tokens)
        budgeted.layers_used = self._collect_layers(budgeted.items)
        return budgeted

    def _deduplicate(self, results: List[UnifiedMemoryResult]) -> List[UnifiedMemoryResult]:
        """Deduplicate by normalized content hash while preserving priority ordering."""
        seen: Set[str] = set()
        deduped: List[UnifiedMemoryResult] = []
        for result in sorted(
            results,
            key=lambda item: (
                self._layer_priority(item.source_layer),
                -item.relevance_score,
                -item.confidence,
            ),
        ):
            normalized = re.sub(r"\s+", " ", result.content.strip().lower())
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            deduped.append(result)
        return deduped

    def _apply_budget(
        self,
        results: List[UnifiedMemoryResult],
        budget_tokens: int,
    ) -> HybridRetrievalResult:
        """Apply total budget with layer priority ordering."""
        used = 0
        selected: List[UnifiedMemoryResult] = []
        for result in results:
            item_tokens = self.budget_manager.estimate_tokens(result.content)
            if used + item_tokens > budget_tokens:
                remaining = max(budget_tokens - used, 0)
                if remaining <= 0:
                    continue
                truncated_content, truncated_tokens = self.budget_manager.apply_budget(
                    result.source_layer,
                    result.content,
                    budget=remaining,
                )
                if not truncated_content:
                    continue
                selected.append(
                    UnifiedMemoryResult(
                        content=truncated_content,
                        source_layer=result.source_layer,
                        source_id=result.source_id,
                        relevance_score=result.relevance_score,
                        confidence=result.confidence,
                        memory_type=result.memory_type,
                        citation_url=result.citation_url,
                    )
                )
                used += truncated_tokens
                continue
            selected.append(result)
            used += item_tokens
        return HybridRetrievalResult(items=selected, budget_used=used)

    def _layer_priority(self, layer: str) -> int:
        priorities = {"summary": 0, "atomic": 1, "archive": 2}
        return priorities.get(layer, 3)

    def _collect_layers(self, results: List[UnifiedMemoryResult]) -> List[str]:
        layers: List[str] = []
        for result in results:
            if result.source_layer not in layers:
                layers.append(result.source_layer)
        return layers


class PromptContextAssembler:
    """Single entry point for all prompt context construction."""

    def __init__(
        self,
        preferences_store: ProfilePreferencesStore,
        project_store: ProjectInstructionStore,
        hybrid_retriever: HybridMemoryRetriever,
        budget_manager: Optional[ContextBudgetManager] = None,
        claim_scorer: Optional[SummaryClaimScorer] = None,
        styles_store: Optional[StylesStore] = None,
        control_panel: Optional[MemoryControlPanel] = None,
    ):
        self.preferences_store = preferences_store
        self.project_store = project_store
        self.hybrid_retriever = hybrid_retriever
        self.budget_manager = budget_manager or ContextBudgetManager()
        self.claim_scorer = claim_scorer or SummaryClaimScorer()
        self.styles_store = styles_store
        self.control_panel = control_panel

    async def assemble(
        self,
        user_id: str,
        scope: MemoryScope,
        project_id: Optional[str],
        global_summary: Optional[GlobalMemorySummary],
        project_summary: Optional[ProjectMemorySummary],
        current_query: str,
        include_semantic: bool = True,
        memory_state: MemoryState = MemoryState.ON,
        **kwargs: Any,
    ) -> PromptContext:
        """Assemble the complete prompt context in one scoped pass."""
        # --- MemoryState enforcement ---
        # OFF: suppress all memory layers; only preferences survive.
        # PAUSED: existing memories injected, no new synthesis (caller-side).
        # ON: full operation.
        effective_off = memory_state == MemoryState.OFF

        if scope == MemoryScope.INCOGNITO or effective_off:
            prefs_xml = self.preferences_store.to_xml(user_id)
            styles_xml = self._build_styles_layer(user_id)
            trimmed_layers, used_tokens = self.budget_manager.apply_global_budget(
                {"preferences": prefs_xml, "styles": styles_xml},
                total_budget_tokens=sum(self.budget_manager.layer_budgets.values()),
            )
            suppressed = [
                "user_memories",
                "project_instructions",
                "semantic_memories",
                "archive_references",
            ]
            if effective_off:
                suppressed.append("state_off")
            return PromptContext(
                profile_preferences_xml=trimmed_layers.get("preferences", ""),
                styles_xml=trimmed_layers.get("styles", ""),
                token_budget_used={
                    "preferences": used_tokens.get("preferences", 0),
                    "styles": used_tokens.get("styles", 0),
                },
                layers_suppressed=suppressed,
            )

        if scope == MemoryScope.PROJECT and project_id:
            summary = project_summary
        else:
            summary = global_summary

        summary_xml = self._build_summary_layer(summary, scope)
        prefs_xml = self.preferences_store.to_xml(user_id)
        project_instructions_xml = ""
        if scope == MemoryScope.PROJECT and project_id:
            instructions = self.project_store.get(project_id)
            if instructions:
                project_instructions_xml = (
                    "<projectInstructions>\n"
                    + GlobalMemorySummary._escape_xml(instructions)
                    + "\n</projectInstructions>"
                )

        semantic_xml = ""
        archive_xml = ""
        styles_xml = self._build_styles_layer(user_id)
        layers_suppressed: List[str] = []

        if include_semantic and current_query:
            hybrid_result = await self.hybrid_retriever.retrieve(
                user_id=user_id,
                query=current_query,
                scope=scope,
                project_id=project_id,
                global_summary=global_summary,
                project_summary=project_summary,
                total_budget_tokens=sum(self.budget_manager.layer_budgets.values()),
            )
            semantic_items = [
                item for item in hybrid_result.items if item.source_layer == "atomic"
            ]
            archive_items = [
                item for item in hybrid_result.items if item.source_layer == "archive"
            ]
            if semantic_items:
                semantic_xml = "<semanticMemories>\n" + "\n".join(
                    f"- {GlobalMemorySummary._escape_xml(item.content)}"
                    for item in semantic_items
                ) + "\n</semanticMemories>"
            if archive_items:
                archive_xml = "<conversationReferences>\n" + "\n".join(
                    (
                        f"- {GlobalMemorySummary._escape_xml(item.content)}"
                        + (
                            f" ({item.citation_url})"
                            if item.citation_url
                            else ""
                        )
                    )
                    for item in archive_items
                ) + "\n</conversationReferences>"
        else:
            layers_suppressed.append("semantic_memories")

        raw_layers = {
            "summary": summary_xml,
            "preferences": prefs_xml,
            "project_instructions": project_instructions_xml,
            "styles": styles_xml,
            "semantic": semantic_xml,
            "archive": archive_xml,
        }
        trimmed_layers, used_tokens = self.budget_manager.apply_global_budget(
            raw_layers,
            total_budget_tokens=sum(self.budget_manager.layer_budgets.values()),
        )

        return PromptContext(
            user_memories_xml=trimmed_layers.get("summary", ""),
            profile_preferences_xml=trimmed_layers.get("preferences", ""),
            project_instructions_xml=trimmed_layers.get("project_instructions", ""),
            styles_xml=trimmed_layers.get("styles", ""),
            semantic_memories_xml=trimmed_layers.get("semantic", ""),
            archive_references_xml=trimmed_layers.get("archive", ""),
            token_budget_used={
                "summary": used_tokens.get("summary", 0),
                "preferences": used_tokens.get("preferences", 0),
                "project_instructions": used_tokens.get("project_instructions", 0),
                "styles": used_tokens.get("styles", 0),
                "semantic": used_tokens.get("semantic", 0),
                "archive": used_tokens.get("archive", 0),
            },
            layers_suppressed=layers_suppressed,
        )

    def _build_styles_layer(self, user_id: str) -> str:
        """Build the styles XML layer from the StylesStore."""
        if not self.styles_store:
            return ""
        user_styles = self.styles_store.get_styles(user_id)
        if not user_styles:
            return ""
        active_styles = [
            s for s in user_styles
            if s.enabled and s.system_prompt_additions
        ]
        if not active_styles:
            return ""
        lines = ["<userStyles>"]
        for style in active_styles:
            lines.append(
                f"  <style name=\"{GlobalMemorySummary._escape_xml(style.name)}\">"
            )
            lines.append(
                f"    {GlobalMemorySummary._escape_xml(style.system_prompt_additions)}"
            )
            lines.append("  </style>")
        lines.append("</userStyles>")
        return "\n".join(lines)

    def _build_summary_layer(
        self,
        summary: Optional[Union[GlobalMemorySummary, ProjectMemorySummary]],
        scope: MemoryScope,
    ) -> str:
        """Build a claim-aware summary layer before global context budgeting."""
        if summary is None:
            return ""

        summary_xml = summary.to_xml()
        budget = self.budget_manager.layer_budgets.get("summary", 0)
        if not summary_xml or self.budget_manager.estimate_tokens(summary_xml) <= budget:
            return summary_xml
        if not summary.claims:
            return summary_xml

        tag_name = "projectMemories" if scope == MemoryScope.PROJECT else "userMemories"
        return self._build_claim_budget_summary(summary, tag_name, budget)

    def _build_claim_budget_summary(
        self,
        summary: Union[GlobalMemorySummary, ProjectMemorySummary],
        tag_name: str,
        budget: int,
    ) -> str:
        """Select the highest-signal claims when a full summary would overflow."""
        ranked_claims = sorted(
            summary.claims,
            key=lambda claim: self.claim_scorer.score(claim),
            reverse=True,
        )
        lines = [f"<{tag_name}>"]

        prose_summary = ""
        if summary.structured_data and summary.structured_data.summary:
            prose_summary = summary.structured_data.summary.strip()
        elif summary.summary_text:
            prose_summary = summary.summary_text.strip().split("\n\n", 1)[0]
        if prose_summary:
            lines.append(GlobalMemorySummary._escape_xml(prose_summary))

        claims_added = 0
        for claim in ranked_claims:
            provenance = (
                f" [src:{','.join(claim.source_conv_ids[:3])}]"
                if claim.source_conv_ids
                else ""
            )
            candidate_line = (
                f"- [{claim.category}] "
                f"{GlobalMemorySummary._escape_xml(claim.text)}"
                f"{provenance}"
            )
            candidate_text = "\n".join(lines + [candidate_line, f"</{tag_name}>"])
            if (
                self.budget_manager.estimate_tokens(candidate_text) > budget
                and len(lines) > 1
            ):
                if claims_added == 0 and prose_summary:
                    lines = [f"<{tag_name}>"]
                    candidate_text = "\n".join(lines + [candidate_line, f"</{tag_name}>"])
                    if self.budget_manager.estimate_tokens(candidate_text) > budget:
                        break
                else:
                    break
            lines.append(candidate_line)
            claims_added += 1

        lines.append(f"</{tag_name}>")
        return "\n".join(lines)


class ContextInjectionManager(AtomicMemoryRetriever):
    """Compatibility shim over the atomic retriever for legacy callers."""

    async def inject_into_messages(
        self, messages: List[Dict], user_id: str, current_query: str
    ) -> List[Dict]:
        """Inject semantic memory context into a message list."""
        context_block = await self.build_context_block(user_id, current_query)

        if not context_block:
            return messages

        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = context_block + "\n\n" + messages[0]["content"]
            return messages

        developer_msg = {"role": "developer", "content": context_block}
        return [developer_msg] + messages


# ============================================================================
# Complete Memory Manager (Orchestrator)
# ============================================================================


class MemoryManager:
    """
    Complete memory system orchestrator.
    Combines summary-based persistent memory with semantic retrieval and RAG archive.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        embedder: Optional[MemoryEmbedder] = None,
        memory_store: Optional[MemoryStore] = None,
        archive: Optional[ConversationArchive] = None,
        synthesizer: Optional[SummaryRefreshSynthesizer] = None,
        edit_processor: Optional[ExplicitMemoryEditProcessor] = None,
        preferences_store: Optional[ProfilePreferencesStore] = None,
        styles_store: Optional[StylesStore] = None,
        project_store: Optional[ProjectInstructionStore] = None,
        control_panel: Optional[MemoryControlPanel] = None,
        budget_manager: Optional[ContextBudgetManager] = None,
    ):
        config = config or {}

        self.embedder = embedder or MemoryEmbedder(
            embedding_model=config.get("embedding_model", "local-pipeline"),
            embedding_dim=config.get("embedding_dim", 3072),
            stage_names=config.get("embedding_stage_names"),
            stage_weights=config.get("embedding_stage_weights"),
        )

        self.memory_store = memory_store or MemoryStore(
            storage_backend=config.get("storage_backend", "in_memory"),
            embedder=self.embedder,
        )

        self.archive = archive or ConversationArchive(
            embedder=self.embedder,
            vector_backend=config.get("vector_backend", "in_memory"),
        )

        self.synthesizer = synthesizer or SummaryRefreshSynthesizer(
            archive=self.archive,
            synthesis_interval_hours=config.get("synthesis_interval_hours", 24),
            work_focused_filter=config.get("work_focused_filter", True),
            synthesis_backend=config.get("synthesis_backend"),
            category_filter=config.get("category_filter"),
        )

        self.edit_processor = edit_processor or ExplicitMemoryEditProcessor(
            embedder=self.embedder
        )

        self.preferences_store = preferences_store or ProfilePreferencesStore()
        self.styles_store = styles_store or StylesStore()
        self.project_store = project_store or ProjectInstructionStore()
        self.control_panel = control_panel or MemoryControlPanel()
        self.deletion_service = DeletionPropagationService(
            archive=self.archive, synthesizer=self.synthesizer
        )
        self.import_normalizer = ImportNormalizer(edit_processor=self.edit_processor)

        self.budget_manager = budget_manager or ContextBudgetManager(
            layer_budgets=config.get("layer_budgets")
        )
        self.atomic_retriever = AtomicMemoryRetriever(
            memory_store=self.memory_store,
            max_context_tokens=config.get("max_context_tokens", 2000),
        )
        self.injector = ContextInjectionManager(
            memory_store=self.memory_store,
            max_context_tokens=config.get("max_context_tokens", 2000),
        )
        self.hybrid_retriever = HybridMemoryRetriever(
            memory_store=self.memory_store,
            atomic_retriever=self.atomic_retriever,
            archive=self.archive,
            budget_manager=self.budget_manager,
        )
        self.prompt_assembler = PromptContextAssembler(
            preferences_store=self.preferences_store,
            project_store=self.project_store,
            hybrid_retriever=self.hybrid_retriever,
            budget_manager=self.budget_manager,
            styles_store=self.styles_store,
            control_panel=self.control_panel,
        )

        self.global_summaries: Dict[str, GlobalMemorySummary] = {}
        self.project_summaries: Dict[Tuple[str, str], ProjectMemorySummary] = {}
        self.import_history: List[ImportLogEntry] = []

        self.auto_extract = config.get("auto_extract", True)
        self.extraction_interval = config.get("extraction_interval", 5)

        self.slog = StructuredLogger("memory.manager")
        self.metrics = MemoryMetrics()
        self.event_bus = MemoryEventBus()
        self.rate_limiter = TokenBucketRateLimiter(
            max_tokens=config.get("rate_limit_max_tokens", 100),
            refill_rate=config.get("rate_limit_refill_rate", 10.0),
        )
        self.encryption_provider: EncryptionProvider = NoOpEncryptionProvider()
        self.lifecycle_manager = MemoryLifecycleManager(
            memory_store=self.memory_store,
            archive=self.archive,
            policy=EvictionPolicy(
                max_memories_per_user=config.get("max_memories_per_user", 10000),
                max_conversations_per_user=config.get("max_conversations_per_user", 5000),
                memory_ttl_days=config.get("memory_ttl_days", 30),
                conversation_ttl_days=config.get("conversation_ttl_days", 90),
            ),
            slog=StructuredLogger("memory.lifecycle"),
        )
        self.health_check = HealthCheck(
            memory_store=self.memory_store,
            archive=self.archive,
            control_panel=self.control_panel,
            metrics=None,
        )

    def get_global_summary(self, user_id: str) -> GlobalMemorySummary:
        """Get global memory summary for user."""
        if user_id not in self.global_summaries:
            self.global_summaries[user_id] = GlobalMemorySummary(user_id=user_id)
        return self.global_summaries[user_id]

    def get_project_summary(
        self, user_id: str, project_id: str
    ) -> ProjectMemorySummary:
        """Get project memory summary."""
        key = (user_id, project_id)
        if key not in self.project_summaries:
            self.project_summaries[key] = ProjectMemorySummary(
                project_id=project_id, user_id=user_id
            )
        return self.project_summaries[key]

    async def synthesize_memory(
        self,
        user_id: str,
        scope: MemoryScope = MemoryScope.GLOBAL,
        project_id: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> Union[GlobalMemorySummary, ProjectMemorySummary]:
        """Trigger memory synthesis. Respects MemoryState enforcement."""
        settings = self.control_panel.get_settings(user_id)
        if settings.state in (MemoryState.OFF, MemoryState.PAUSED):
            self.slog.info(
                "synthesis_skipped",
                user_id=user_id,
                reason=f"memory_state={settings.state.value}",
            )
            if scope == MemoryScope.PROJECT and project_id:
                return self.get_project_summary(user_id, project_id)
            return self.get_global_summary(user_id)

        start_time = time.monotonic()

        try:
            if scope == MemoryScope.GLOBAL:
                existing_summary = self.get_global_summary(user_id)
                summary = await self.synthesizer.synthesize(
                    user_id,
                    scope,
                    existing_summary=existing_summary,
                    force_rebuild=force_rebuild,
                )
                self.global_summaries[user_id] = summary
                settings.total_synthesized_count += 1
                result = summary

            elif scope == MemoryScope.PROJECT and project_id:
                existing_project_summary = self.get_project_summary(user_id, project_id)
                summary = await self.synthesizer.synthesize(
                    user_id,
                    scope,
                    project_id,
                    existing_summary=existing_project_summary,
                    force_rebuild=force_rebuild,
                )
                project_summary = ProjectMemorySummary(
                    project_id=project_id,
                    user_id=user_id,
                    summary_text=summary.summary_text,
                    last_synthesized=summary.last_synthesized,
                    source_conversation_ids=summary.source_conversation_ids,
                    explicit_edits=summary.explicit_edits,
                    confidence_score=summary.confidence_score,
                    structured_data=summary.structured_data,
                    claims=list(summary.claims),
                )
                self.project_summaries[(user_id, project_id)] = project_summary
                settings.total_synthesized_count += 1
                result = project_summary

            else:
                raise ValueError(f"Invalid scope for synthesis: {scope}")

            duration_ms = (time.monotonic() - start_time) * 1000
            self.slog.info(
                "synthesis_completed",
                user_id=user_id,
                scope=scope.value,
                duration_ms=round(duration_ms, 2),
            )
            return result

        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            self.slog.error(
                "synthesis_failed",
                user_id=user_id,
                scope=scope.value,
                error=str(exc),
                duration_ms=round(duration_ms, 2),
            )
            raise

    def process_explicit_edit(
        self,
        user_id: str,
        edit_type: str,
        content: str,
        scope: MemoryScope = MemoryScope.GLOBAL,
        project_id: Optional[str] = None,
    ) -> ExplicitMemoryEdit:
        """Process explicit memory edit (remember/forget)."""
        MemoryScopeValidator.assert_write_allowed(
            scope=scope,
            is_incognito=(scope == MemoryScope.INCOGNITO),
        )
        edit = self.edit_processor.process_edit(
            user_id=user_id,
            scope=scope,
            project_id=project_id,
            edit_type=edit_type,
            content=content,
        )

        if scope == MemoryScope.GLOBAL:
            summary = self.get_global_summary(user_id)
            self.edit_processor.apply_explicit_edit_to_summary(summary, edit)

        elif scope == MemoryScope.PROJECT and project_id:
            summary = self.get_project_summary(user_id, project_id)
            self.edit_processor.apply_explicit_edit_to_summary(summary, edit)

        return edit

    async def search_conversations(
        self,
        user_id: str,
        query: str,
        max_results: int = 5,
        project_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[SourceConversationRef]:
        """Search past conversations. All users have full archive access."""
        if not self.control_panel.can_use_chat_search(user_id):
            return []

        return await self.archive.search(
            user_id=user_id, query=query, max_results=max_results, project_id=project_id
        )

    async def get_recent_chats(
        self,
        user_id: str,
        n: int = 3,
        sort_order: str = "desc",
        project_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[SourceConversationRef]:
        """Get recent chats. All users have full archive access."""
        if not self.control_panel.can_use_chat_search(user_id):
            return []

        return self.archive.get_recent(
            user_id=user_id, n=n, sort_order=sort_order, project_id=project_id
        )

    async def build_prompt_context(
        self,
        user_id: str,
        current_query: str,
        conversation_context: Optional[ConversationContext] = None,
        include_summary: bool = True,
        include_preferences: bool = True,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """
        Build complete prompt context with proper injection order.
        Returns dict with keys for each prompt layer.
        Respects MemoryState enforcement via the assembler.
        All capabilities (archive search, semantic retrieval) are always active.
        """
        settings = self.control_panel.get_settings(user_id)
        memory_state = settings.state

        scope = conversation_context.scope if conversation_context else MemoryScope.GLOBAL
        project_id = conversation_context.project_id if conversation_context else None
        global_summary = self.get_global_summary(user_id)
        project_summary = (
            self.get_project_summary(user_id, project_id) if project_id else None
        )

        prompt_context = await self.prompt_assembler.assemble(
            user_id=user_id,
            scope=scope,
            project_id=project_id,
            global_summary=global_summary,
            project_summary=project_summary,
            current_query=current_query,
            include_semantic=True,
            memory_state=memory_state,
        )
        if not include_summary:
            prompt_context.user_memories_xml = ""
            prompt_context.token_budget_used["summary"] = 0
        if not include_preferences:
            prompt_context.profile_preferences_xml = ""
            prompt_context.token_budget_used["preferences"] = 0

        return prompt_context.as_dict()

    async def store_conversation(self, conversation: Conversation) -> None:
        """Store conversation in archive."""
        MemoryScopeValidator.assert_write_allowed(
            scope=conversation.scope,
            is_incognito=conversation.is_incognito,
        )
        await self.archive.store_conversation(conversation)

        if self.deletion_service.should_refresh(conversation.user_id):
            refresh_requests = self.deletion_service.consume_refresh_requests(
                conversation.user_id
            )
            for refresh_scope, refresh_project_id in refresh_requests:
                await self.synthesize_memory(
                    conversation.user_id,
                    refresh_scope,
                    refresh_project_id,
                    force_rebuild=True,
                )

    async def process_conversation_turn(
        self, conversation: ConversationContext, should_extract: Optional[bool] = None
    ) -> Optional[List[MemoryItem]]:
        """Process conversation turn for extraction."""
        if conversation.scope == MemoryScope.INCOGNITO:
            return None
        if should_extract is None:
            should_extract = self.auto_extract and (
                len(conversation.messages) % self.extraction_interval == 0
            )

        if not should_extract:
            return None

        existing = await self.memory_store.retrieve_by_type(
            user_id=conversation.user_id, memory_type=None, limit=50
        )

        extractor = SimpleMemoryExtractor()

        new_memories = await extractor.extract(conversation, existing)

        for memory in new_memories:
            await self.memory_store.store_memory(memory)

        return new_memories

    async def prepare_prompt_with_memory(
        self,
        messages: List[Dict],
        user_id: str,
        conversation_context: Optional[ConversationContext] = None,
        **kwargs: Any,
    ) -> List[Dict]:
        """Prepare messages with injected memory context. Respects MemoryState.
        All capabilities (archive search, semantic retrieval) are always active."""
        settings = self.control_panel.get_settings(user_id)
        memory_state = settings.state

        current_query = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                current_query = msg["content"]
                break

        scope = conversation_context.scope if conversation_context else MemoryScope.GLOBAL
        project_id = conversation_context.project_id if conversation_context else None
        global_summary = self.get_global_summary(user_id)
        project_summary = (
            self.get_project_summary(user_id, project_id) if project_id else None
        )
        prompt_context = await self.prompt_assembler.assemble(
            user_id=user_id,
            scope=scope,
            project_id=project_id,
            global_summary=global_summary,
            project_summary=project_summary,
            current_query=current_query,
            include_semantic=True,
            memory_state=memory_state,
        )
        combined_context = prompt_context.combined_context()
        if not combined_context:
            return messages

        updated_messages = [dict(message) for message in messages]
        if updated_messages and updated_messages[0].get("role") == "system":
            updated_messages[0]["content"] = (
                combined_context + "\n\n" + updated_messages[0]["content"]
            )
            return updated_messages

        developer_msg = {"role": "developer", "content": combined_context}
        return [developer_msg] + updated_messages

    async def get_user_memory_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of user's memory bank."""
        summary = {
            "total_memories": len(self.memory_store.user_memories.get(user_id, set())),
            "by_type": {},
            "top_topics": [],
            "last_updated": None,
            "global_summary_exists": user_id in self.global_summaries,
            "control_state": self.control_panel.get_settings(user_id).state.value,
        }

        for memory_type in MemoryType:
            type_memories = await self.memory_store.retrieve_by_type(
                user_id, memory_type, limit=100
            )
            summary["by_type"][memory_type.value] = len(type_memories)

            if type_memories:
                latest = max(type_memories, key=lambda m: m.created_at)
                if (
                    summary["last_updated"] is None
                    or latest.created_at > summary["last_updated"]
                ):
                    summary["last_updated"] = latest.created_at

        topic_memories = await self.memory_store.retrieve_by_type(
            user_id, MemoryType.TOPIC, limit=10
        )
        summary["top_topics"] = [m.content for m in topic_memories]

        return summary

    def export_memory_snapshot(
        self,
        user_id: str,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export a structured memory snapshot for round-trip portability."""
        global_summary = self.global_summaries.get(user_id)
        project_summary = (
            self.project_summaries.get((user_id, project_id)) if project_id else None
        )
        preferences = self.preferences_store.get(user_id)
        control_settings = self.control_panel.get_settings(user_id)

        if project_id:
            project_summaries = (
                {project_id: self._serialize_summary(project_summary)}
                if project_summary
                else {}
            )
            relevant_instruction_ids = {project_id}
        else:
            project_summaries = {
                summary_project_id: self._serialize_summary(summary)
                for (summary_user_id, summary_project_id), summary in self.project_summaries.items()
                if summary_user_id == user_id
            }
            relevant_instruction_ids = set(self.project_store.instructions.keys()) | set(
                project_summaries.keys()
            )

        memory_items = [
            self.memory_store.memories[memory_id].to_dict()
            for memory_id in sorted(self.memory_store.user_memories.get(user_id, set()))
        ]

        archived_conversations = [
            self._serialize_conversation(conversation)
            for conversation_id in sorted(self.archive.user_conversations.get(user_id, set()))
            if (conversation := self.archive.conversations.get(conversation_id)) is not None
            and (project_id is None or conversation.project_id == project_id)
        ]

        project_instructions = {
            instruction_project_id: self.project_store.get(instruction_project_id)
            for instruction_project_id in sorted(relevant_instruction_ids)
            if self.project_store.get(instruction_project_id)
        }

        return {
            "user_id": user_id,
            "project_id": project_id,
            "global_summary": self._serialize_summary(global_summary),
            "project_summary": self._serialize_summary(project_summary),
            "project_summaries": project_summaries,
            "preferences": {
                "communication_style": preferences.communication_style,
                "technical_expertise": preferences.technical_expertise,
                "response_length": preferences.response_length,
                "code_style": preferences.code_style,
                "preferred_languages": list(preferences.preferred_languages),
                "custom_instructions": list(preferences.custom_instructions),
            },
            "control_settings": self._serialize_control_settings(control_settings),
            "project_instructions": project_instructions,
            "styles": [
                {
                    "style_id": style.style_id,
                    "name": style.name,
                    "description": style.description,
                    "system_prompt_additions": style.system_prompt_additions,
                }
                for style in self.styles_store.get_styles(user_id)
            ],
            "memories": memory_items,
            "archived_conversations": archived_conversations,
            "synthesis_audit": [
                self._serialize_audit_entry(entry)
                for entry in self.get_synthesis_audit_log()
                if entry.project_id == project_id or project_id is None
            ],
            "explicit_edits": [
                edit.to_dict() for edit in self.get_explicit_edit_log(user_id)
            ],
            "import_history": [
                self._serialize_import_entry(entry)
                for entry in self.get_import_log()
                if entry.user_id == user_id
                and (entry.project_id == project_id or project_id is None)
            ],
        }

    async def import_memory_snapshot(
        self,
        user_id: str,
        payload: Dict[str, Any],
        scope: MemoryScope = MemoryScope.GLOBAL,
        project_id: Optional[str] = None,
        source: str = "structured_snapshot",
        conflict_policy: str = "merge",
    ) -> Dict[str, int]:
        """Import a structured snapshot with validation and audit logging."""
        MemoryScopeValidator.assert_write_allowed(
            scope=scope,
            is_incognito=(scope == MemoryScope.INCOGNITO),
        )

        if conflict_policy not in {"merge", "overwrite", "skip"}:
            raise ValueError("conflict_policy must be 'merge', 'overwrite', or 'skip'")

        imported_memories = 0
        imported_edits = 0
        imported_conversations = 0

        preferences = payload.get("preferences", {})
        if (
            isinstance(preferences, dict)
            and preferences
            and conflict_policy != "skip"
        ):
            self.preferences_store.update(user_id, preferences)

        control_payload = payload.get("control_settings", {})
        if isinstance(control_payload, dict) and control_payload and conflict_policy != "skip":
            self._restore_control_settings(user_id, control_payload, conflict_policy)

        for imported_project_id, instructions in payload.get("project_instructions", {}).items():
            if not instructions:
                continue
            if conflict_policy == "skip" and self.project_store.get(imported_project_id):
                continue
            self.project_store.set(imported_project_id, instructions)

        summary_payload = payload.get(
            "project_summary" if scope == MemoryScope.PROJECT else "global_summary"
        )
        if isinstance(summary_payload, dict) and (
            conflict_policy != "skip"
            or (
                not self.project_summaries.get((user_id, project_id))
                if scope == MemoryScope.PROJECT and project_id
                else user_id not in self.global_summaries
            )
        ):
            restored_summary = self._restore_summary(
                user_id=user_id,
                project_id=project_id,
                payload=summary_payload,
                scope=scope,
            )
            if scope == MemoryScope.PROJECT and project_id:
                if (
                    conflict_policy == "merge"
                    and (user_id, project_id) in self.project_summaries
                ):
                    self.project_summaries[(user_id, project_id)] = self._merge_summary_objects(
                        self.project_summaries[(user_id, project_id)],
                        restored_summary,
                        project_id=project_id,
                    )
                else:
                    self.project_summaries[(user_id, project_id)] = restored_summary
            else:
                if conflict_policy == "merge" and user_id in self.global_summaries:
                    self.global_summaries[user_id] = self._merge_summary_objects(
                        self.global_summaries[user_id],
                        restored_summary,
                    )
                else:
                    self.global_summaries[user_id] = restored_summary

        if scope == MemoryScope.GLOBAL:
            for imported_project_id, summary_blob in payload.get("project_summaries", {}).items():
                if not isinstance(summary_blob, dict):
                    continue
                if conflict_policy == "skip" and (user_id, imported_project_id) in self.project_summaries:
                    continue
                restored_project_summary = self._restore_summary(
                    user_id=user_id,
                    project_id=imported_project_id,
                    payload=summary_blob,
                    scope=MemoryScope.PROJECT,
                )
                if conflict_policy == "merge" and (user_id, imported_project_id) in self.project_summaries:
                    self.project_summaries[(user_id, imported_project_id)] = self._merge_summary_objects(
                        self.project_summaries[(user_id, imported_project_id)],
                        restored_project_summary,
                        project_id=imported_project_id,
                    )
                else:
                    self.project_summaries[(user_id, imported_project_id)] = restored_project_summary

        for memory_payload in payload.get("memories", []):
            if conflict_policy == "skip" and memory_payload.get("id") in self.memory_store.memories:
                continue
            memory = self._restore_memory_item(user_id, memory_payload)
            await self.memory_store.store_memory(memory)
            imported_memories += 1

        for conversation_payload in payload.get("archived_conversations", []):
            conversation = self._restore_conversation(user_id, conversation_payload)
            if conversation is None:
                continue
            if (
                conflict_policy == "skip"
                and conversation.conversation_id in self.archive.conversations
            ):
                continue
            await self.archive.store_conversation(conversation)
            imported_conversations += 1

        # synthesis_history tracking removed (De-Tox)

        if "summary" in payload or "structured" in payload:
            edits = await self.import_normalizer.import_from_text(
                user_id=user_id,
                raw_text=json.dumps(payload),
                scope=scope,
                project_id=project_id,
            )
            imported_edits += len(edits)

        self.import_history.append(
            ImportLogEntry(
                import_id=str(uuid.uuid4()),
                user_id=user_id,
                scope=scope,
                project_id=project_id,
                source=source,
                edits_created=imported_edits,
                notes=(
                    f"Imported {imported_memories} memories and "
                    f"{imported_conversations} conversations "
                    f"from structured snapshot using {conflict_policy} policy."
                ),
            )
        )

        return {
            "memories_imported": imported_memories,
            "edits_created": imported_edits,
            "conversations_imported": imported_conversations,
        }

    async def import_conversations_json(
        self,
        user_id: str,
        file_path: str,
        scope: MemoryScope = MemoryScope.GLOBAL,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Universal importer for raw JSON/JSONL conversation exports (ChatGPT, Claude, etc).
        Auto-detects format, converts to Conversation objects, and stores them in the archive.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                while True:
                    first_char = f.read(1)
                    if not first_char or not first_char.isspace():
                        break
                f.seek(0)

                if first_char == "[":
                    data = json.load(f)
                else:
                    data = []
                    for line_number, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            raise ValueError(
                                f"Invalid JSONL at line {line_number} in {file_path}: {e}"
                            ) from e

        except Exception as e:
            raise ValueError(f"Failed to load JSON/JSONL file {file_path}: {e}") from e

        if not isinstance(data, list):
            raise ValueError("Expected a JSON list of conversations at the root level.")

        # Auto-detect flat message list (e.g. from custom CSV/JSON exports)
        if data and isinstance(data[0], dict) and "conversation_id" in data[0] and "role" in data[0]:
            grouped = {}
            for msg in data:
                cid = msg.get("conversation_id")
                if not cid:
                    continue
                if cid not in grouped:
                    grouped[cid] = {
                        "id": cid,
                        "title": msg.get("conversation_title", "Imported Conversation"),
                        "messages": []
                    }
                grouped[cid]["messages"].append({
                    "id": msg.get("message_id", str(uuid.uuid4())),
                    "role": msg.get("role", "user"),
                    "content": msg.get("text") or msg.get("content") or "",
                    "timestamp": msg.get("create_time")
                })
            data = list(grouped.values())

        conversations_imported = 0
        messages_processed = 0

        for item in data:
            if not isinstance(item, dict):
                continue
            
            conv = None
            if "chat_messages" in item:
                conv = self._parse_claude_export(user_id, item, scope, project_id)
            elif "mapping" in item:
                conv = self._parse_chatgpt_export(user_id, item, scope, project_id)
            elif "messages" in item:
                conv = self._parse_generic_export(user_id, item, scope, project_id)
                
            if conv and conv.messages:
                await self.store_conversation(conv)
                conversations_imported += 1
                messages_processed += len(conv.messages)

        return {
            "conversations_imported": conversations_imported,
            "messages_processed": messages_processed,
            "source_file": file_path
        }

    def _parse_claude_export(
        self, user_id: str, item: Dict[str, Any], scope: MemoryScope, project_id: Optional[str]
    ) -> Optional[Conversation]:
        """Convert a Claude-format export item into a Conversation."""
        conv_id = item.get("uuid", str(uuid.uuid4()))
        title = item.get("name") or item.get("summary") or "Imported Claude Conversation"
        created_at_str = item.get("created_at")
        timestamp = datetime.now()
        if created_at_str:
            try:
                # Basic ISO parse attempt
                timestamp = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            except ValueError:
                pass

        messages = []
        for msg_item in item.get("chat_messages", []):
            sender = msg_item.get("sender", "human").lower()
            role = "user" if sender == "human" else "assistant"
            text_parts = []
            
            # Text might be in 'text' directly, or inside 'content' arrays
            if "text" in msg_item and msg_item["text"]:
                text_parts.append(msg_item["text"])
            
            for content_block in msg_item.get("content", []):
                if isinstance(content_block, dict) and "text" in content_block and content_block["text"]:
                    text_parts.append(content_block["text"])
            
            content = "\n".join(text_parts).strip()
            if not content:
                continue
                
            messages.append(ConversationMessage(
                message_id=msg_item.get("uuid", str(uuid.uuid4())),
                role=role,
                content=content,
                timestamp=timestamp
            ))
            
        if not messages:
            return None
            
        return Conversation(
            conversation_id=conv_id,
            user_id=user_id,
            scope=scope,
            project_id=project_id,
            title=title,
            messages=messages,
            created_at=timestamp,
            updated_at=timestamp
        )

    def _parse_chatgpt_export(
        self, user_id: str, item: Dict[str, Any], scope: MemoryScope, project_id: Optional[str]
    ) -> Optional[Conversation]:
        """Convert a ChatGPT-format export item into a Conversation by walking the mapping graph."""
        conv_id = item.get("id") or item.get("conversation_id", str(uuid.uuid4()))
        title = item.get("title") or "Imported ChatGPT Conversation"
        mapping = item.get("mapping", {})
        
        nodes = []
        for node_id, node_data in mapping.items():
            if not isinstance(node_data, dict):
                continue
            message = node_data.get("message")
            if not message or not isinstance(message, dict):
                continue
                
            author = message.get("author", {})
            role = author.get("role")
            if role not in ("user", "assistant"):
                continue
                
            content_dict = message.get("content", {})
            parts = content_dict.get("parts", [])
            text = "\n".join(str(p) for p in parts if isinstance(p, str)).strip()
            if not text:
                continue
                
            create_time = message.get("create_time") or 0.0
            nodes.append({
                "id": node_id,
                "role": role,
                "text": text,
                "time": create_time
            })
            
        nodes.sort(key=lambda x: x["time"])
        
        messages = []
        for n in nodes:
            messages.append(ConversationMessage(
                message_id=n["id"],
                role=n["role"],
                content=n["text"],
                timestamp=datetime.fromtimestamp(n["time"]) if n["time"] > 0 else datetime.now()
            ))
            
        if not messages:
            return None
            
        return Conversation(
            conversation_id=conv_id,
            user_id=user_id,
            scope=scope,
            project_id=project_id,
            title=title,
            messages=messages,
            created_at=messages[0].timestamp if messages else datetime.now()
        )

    def _parse_generic_export(
        self, user_id: str, item: Dict[str, Any], scope: MemoryScope, project_id: Optional[str]
    ) -> Optional[Conversation]:
        """Fallback for simple {"messages": [{"role": "user", "content": "..."}]} formats."""
        conv_id = item.get("id", str(uuid.uuid4()))
        title = item.get("title", "Imported Conversation")
        
        messages = []
        for msg in item.get("messages", []):
            role = msg.get("role")
            content = msg.get("content")
            if role in ("user", "assistant") and content:
                timestamp_val = msg.get("timestamp")
                if timestamp_val:
                    if isinstance(timestamp_val, (int, float)):
                        timestamp = datetime.fromtimestamp(timestamp_val)
                    else:
                        timestamp = datetime.now()
                else:
                    timestamp = datetime.now()
                
                messages.append(ConversationMessage(
                    message_id=msg.get("id", str(uuid.uuid4())),
                    role=role,
                    content=content,
                    timestamp=timestamp
                ))
                
        if not messages:
            return None
            
        return Conversation(
            conversation_id=conv_id,
            user_id=user_id,
            scope=scope,
            project_id=project_id,
            title=title,
            messages=messages,
        )

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and immediately run targeted synthesis refreshes."""
        if not self.deletion_service.mark_deleted(conversation_id):
            return False

        conversation = self.archive.get_conversation(conversation_id)
        if conversation is None:
            return False

        for scope, project_id in self.deletion_service.consume_refresh_requests(
            conversation.user_id
        ):
            await self.synthesize_memory(
                conversation.user_id,
                scope,
                project_id,
                force_rebuild=True,
            )

        return True

    def get_synthesis_audit_log(self) -> List[SynthesisAuditEntry]:
        """Expose synthesis audit history through the manager."""
        return self.synthesizer.get_audit_log()

    def get_explicit_edit_log(self, user_id: str) -> List[ExplicitMemoryEdit]:
        """Expose explicit memory edit history for a user."""
        return self.edit_processor.get_edit_history(user_id)

    def get_import_log(self) -> List[ImportLogEntry]:
        """Expose structured import history."""
        return list(self.import_history)

    def set_project_instructions(self, project_id: str, content: str) -> None:
        """Store project-scoped instructions used by prompt assembly."""
        self.project_store.set(project_id, content)

    async def reset_memory(self, user_id: str) -> Dict[str, int]:
        """
        Full data purge for a user — clears all stores and sets state to OFF.
        Returns counts of purged items.
        """
        with correlation_scope() as cid:
            self.slog.info("reset_memory_started", user_id=user_id, correlation_id=cid)

            purged_memories = 0
            purged_conversations = 0

            memory_ids = list(self.memory_store.user_memories.get(user_id, set()))
            for mid in memory_ids:
                if mid in self.memory_store.memories:
                    del self.memory_store.memories[mid]
                    purged_memories += 1
            self.memory_store.user_memories.pop(user_id, None)

            conv_ids = list(self.archive.user_conversations.get(user_id, set()))
            for cid_conv in conv_ids:
                if cid_conv in self.archive.conversations:
                    del self.archive.conversations[cid_conv]
                    purged_conversations += 1
                self.archive.conversation_embeddings.pop(cid_conv, None)
            self.archive.user_conversations.pop(user_id, None)

            self.global_summaries.pop(user_id, None)

            project_keys_to_remove = [
                key for key in self.project_summaries if key[0] == user_id
            ]
            for key in project_keys_to_remove:
                del self.project_summaries[key]

            settings = self.control_panel.get_settings(user_id)
            settings.state = MemoryState.OFF
            settings.last_reset = datetime.now()

            result = {
                "memories_purged": purged_memories,
                "conversations_purged": purged_conversations,
                "project_summaries_purged": len(project_keys_to_remove),
            }
            self.slog.info("reset_memory_completed", user_id=user_id, **result)
            return result

    def get_health(self) -> Dict[str, Any]:
        """Return composite health check for the memory system."""
        return self.health_check.check()

    def get_metrics(self) -> Dict[str, Any]:
        """Return current metrics snapshot."""
        return self.metrics.snapshot()

    async def run_eviction(self, user_id: str) -> Dict[str, int]:
        """Run TTL-based eviction for a user. Delegates to lifecycle manager."""
        return await self.lifecycle_manager.run_eviction(user_id)

    def get_tool_definitions(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Return all function-calling tool schemas. All capabilities enabled."""
        return MemoryToolDefinitions.all_tools()

    def _serialize_summary(
        self,
        summary: Optional[Union[GlobalMemorySummary, ProjectMemorySummary]],
    ) -> Dict[str, Any]:
        """Serialize a summary dataclass for export."""
        if summary is None:
            return {}
        return {
            "summary_text": summary.summary_text,
            "last_synthesized": (
                summary.last_synthesized.isoformat()
                if summary.last_synthesized
                else None
            ),
            "last_modified": summary.last_modified.isoformat(),
            "source_conversation_ids": sorted(summary.source_conversation_ids),
            "explicit_edits": list(summary.explicit_edits),
            "confidence_score": summary.confidence_score,
            "structured_data": (
                summary.structured_data.raw_json if summary.structured_data else {}
            ),
            "claims": [
                {
                    "text": claim.text,
                    "category": claim.category,
                    "confidence": claim.confidence,
                    "source_conv_ids": list(claim.source_conv_ids),
                    "last_updated": claim.last_updated.isoformat(),
                }
                for claim in summary.claims
            ],
        }

    def _serialize_audit_entry(self, entry: SynthesisAuditEntry) -> Dict[str, Any]:
        """Serialize a synthesis audit entry."""
        return {
            "synthesis_id": entry.synthesis_id,
            "timestamp": entry.timestamp.isoformat(),
            "scope": entry.scope.value,
            "project_id": entry.project_id,
            "conversations_processed": entry.conversations_processed,
            "new_content": entry.new_content,
            "source_conversation_ids": list(entry.source_conversation_ids),
            "backend_name": entry.backend_name,
            "delta_mode": entry.delta_mode,
            "confidence_score": entry.confidence_score,
            "structured_output": dict(entry.structured_output),
            "changes_made": list(entry.changes_made),
        }

    def _serialize_import_entry(self, entry: ImportLogEntry) -> Dict[str, Any]:
        """Serialize an import audit entry."""
        return {
            "import_id": entry.import_id,
            "user_id": entry.user_id,
            "scope": entry.scope.value,
            "project_id": entry.project_id,
            "source": entry.source,
            "imported_at": entry.imported_at.isoformat(),
            "edits_created": entry.edits_created,
            "notes": entry.notes,
        }

    def _serialize_control_settings(self, settings: MemoryControlSettings) -> Dict[str, Any]:
        """Serialize control-panel settings for export."""
        return {
            "state": settings.state.value,
            "synthesis_interval_hours": settings.synthesis_interval_hours,
            "enable_chat_search": settings.enable_chat_search,
            "last_reset": settings.last_reset.isoformat() if settings.last_reset else None,
            "total_synthesized_count": settings.total_synthesized_count,
        }

    def _serialize_conversation(self, conversation: Conversation) -> Dict[str, Any]:
        """Serialize an archived conversation for snapshot export."""
        return {
            "conversation_id": conversation.conversation_id,
            "user_id": conversation.user_id,
            "project_id": conversation.project_id,
            "is_incognito": conversation.is_incognito,
            "scope": conversation.scope.value,
            "title": conversation.title,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "is_deleted": conversation.is_deleted,
            "messages": [
                {
                    "message_id": message.message_id,
                    "role": message.role,
                    "content": message.content,
                    "timestamp": message.timestamp.isoformat(),
                    "tool_calls": list(message.tool_calls or []),
                    "metadata": dict(message.metadata),
                }
                for message in conversation.messages
            ],
        }

    def _restore_summary(
        self,
        user_id: str,
        project_id: Optional[str],
        payload: Dict[str, Any],
        scope: MemoryScope,
    ) -> Union[GlobalMemorySummary, ProjectMemorySummary]:
        """Restore a summary payload from exported JSON."""
        structured_data_payload = payload.get("structured_data", {})
        structured_data = (
            StructuredSynthesisOutput(
                summary=structured_data_payload.get("summary", ""),
                role=structured_data_payload.get("structured", {}).get("role", ""),
                projects=list(
                    structured_data_payload.get("structured", {}).get("projects", [])
                ),
                tech_stack=list(
                    structured_data_payload.get("structured", {}).get("tech_stack", [])
                ),
                preferences=list(
                    structured_data_payload.get("structured", {}).get(
                        "preferences", []
                    )
                ),
                goals=list(
                    structured_data_payload.get("structured", {}).get("goals", [])
                ),
                constraints=list(
                    structured_data_payload.get("structured", {}).get(
                        "constraints", []
                    )
                ),
                confidence=structured_data_payload.get("confidence", 0.0),
                coverage_notes=structured_data_payload.get("coverage_notes", ""),
                raw_json=dict(structured_data_payload),
            )
            if structured_data_payload
            else None
        )
        claims = [
            SummaryClaim(
                text=claim.get("text", ""),
                category=claim.get("category", "unknown"),
                confidence=claim.get("confidence", 0.0),
                source_conv_ids=list(claim.get("source_conv_ids", [])),
                last_updated=datetime.fromisoformat(claim["last_updated"])
                if claim.get("last_updated")
                else datetime.now(),
            )
            for claim in payload.get("claims", [])
            if claim.get("text")
        ]

        summary_kwargs = {
            "user_id": user_id,
            "summary_text": payload.get("summary_text", ""),
            "last_synthesized": (
                datetime.fromisoformat(payload["last_synthesized"])
                if payload.get("last_synthesized")
                else None
            ),
            "last_modified": (
                datetime.fromisoformat(payload["last_modified"])
                if payload.get("last_modified")
                else datetime.now()
            ),
            "source_conversation_ids": set(payload.get("source_conversation_ids", [])),
            "explicit_edits": list(payload.get("explicit_edits", [])),
            "confidence_score": payload.get("confidence_score", 0.0),
            "structured_data": structured_data,
            "claims": claims,
        }

        if scope == MemoryScope.PROJECT and project_id:
            return ProjectMemorySummary(project_id=project_id, **summary_kwargs)
        return GlobalMemorySummary(**summary_kwargs)

    def _restore_memory_item(self, user_id: str, payload: Dict[str, Any]) -> MemoryItem:
        """Restore an exported atomic memory payload."""
        created_at = (
            datetime.fromisoformat(payload["created_at"])
            if payload.get("created_at")
            else datetime.now()
        )
        last_accessed = (
            datetime.fromisoformat(payload["last_accessed"])
            if payload.get("last_accessed")
            else created_at
        )
        return MemoryItem(
            id=payload.get("id", f"mem_{uuid.uuid4().hex[:16]}"),
            user_id=user_id,
            memory_type=MemoryType(payload.get("memory_type", MemoryType.FACT.value)),
            content=payload.get("content", ""),
            created_at=created_at,
            last_accessed=last_accessed,
            access_count=payload.get("access_count", 0),
            confidence=payload.get("confidence", 1.0),
            source_conversation_id=payload.get("source_conversation_id"),
            related_memories=set(payload.get("related_memories", [])),
            supersedes=payload.get("supersedes"),
            importance_score=payload.get("importance_score", 0.5),
            tags=set(payload.get("tags", [])),
        )

    def _restore_control_settings(
        self,
        user_id: str,
        payload: Dict[str, Any],
        conflict_policy: str,
    ) -> None:
        """Restore control settings with simple conflict handling semantics."""
        settings = self.control_panel.get_settings(user_id)
        if conflict_policy == "merge":
            if payload.get("state"):
                settings.state = MemoryState(payload["state"])
            settings.synthesis_interval_hours = payload.get(
                "synthesis_interval_hours",
                settings.synthesis_interval_hours,
            )
            settings.enable_chat_search = payload.get(
                "enable_chat_search",
                settings.enable_chat_search,
            )
            settings.total_synthesized_count = max(
                settings.total_synthesized_count,
                int(payload.get("total_synthesized_count", 0)),
            )
            if payload.get("last_reset"):
                settings.last_reset = datetime.fromisoformat(payload["last_reset"])
            return

        settings.state = MemoryState(payload.get("state", settings.state.value))
        settings.synthesis_interval_hours = payload.get(
            "synthesis_interval_hours",
            settings.synthesis_interval_hours,
        )
        settings.enable_chat_search = payload.get(
            "enable_chat_search",
            settings.enable_chat_search,
        )
        settings.total_synthesized_count = int(
            payload.get("total_synthesized_count", settings.total_synthesized_count)
        )
        settings.last_reset = (
            datetime.fromisoformat(payload["last_reset"])
            if payload.get("last_reset")
            else settings.last_reset
        )

    def _restore_conversation(
        self,
        user_id: str,
        payload: Dict[str, Any],
    ) -> Optional[Conversation]:
        """Restore an archived conversation payload."""
        if not isinstance(payload, dict) or not payload.get("conversation_id"):
            return None

        messages = [
            ConversationMessage(
                message_id=message.get("message_id", f"msg_{uuid.uuid4().hex[:8]}"),
                role=message.get("role", "user"),
                content=message.get("content", ""),
                timestamp=(
                    datetime.fromisoformat(message["timestamp"])
                    if message.get("timestamp")
                    else datetime.now()
                ),
                tool_calls=list(message.get("tool_calls") or []),
                metadata=dict(message.get("metadata") or {}),
            )
            for message in payload.get("messages", [])
        ]

        conversation = Conversation(
            conversation_id=payload["conversation_id"],
            user_id=user_id,
            project_id=payload.get("project_id"),
            is_incognito=payload.get("is_incognito", False),
            title=payload.get("title", ""),
            messages=messages,
            created_at=(
                datetime.fromisoformat(payload["created_at"])
                if payload.get("created_at")
                else datetime.now()
            ),
            updated_at=(
                datetime.fromisoformat(payload["updated_at"])
                if payload.get("updated_at")
                else datetime.now()
            ),
            is_deleted=payload.get("is_deleted", False),
        )
        if payload.get("scope"):
            conversation.scope = MemoryScope(payload["scope"])
        return conversation

    def _restore_audit_entry(
        self,
        payload: Dict[str, Any],
    ) -> Optional[SynthesisAuditEntry]:
        """Restore a serialized synthesis audit entry."""
        if not isinstance(payload, dict) or not payload.get("synthesis_id"):
            return None
        return SynthesisAuditEntry(
            synthesis_id=payload["synthesis_id"],
            timestamp=(
                datetime.fromisoformat(payload["timestamp"])
                if payload.get("timestamp")
                else datetime.now()
            ),
            scope=MemoryScope(payload.get("scope", MemoryScope.GLOBAL.value)),
            project_id=payload.get("project_id"),
            conversations_processed=int(payload.get("conversations_processed", 0)),
            new_content=payload.get("new_content", ""),
            source_conversation_ids=list(payload.get("source_conversation_ids", [])),
            backend_name=payload.get("backend_name", "deterministic"),
            delta_mode=bool(payload.get("delta_mode", False)),
            confidence_score=float(payload.get("confidence_score", 0.0)),
            structured_output=dict(payload.get("structured_output", {})),
            changes_made=list(payload.get("changes_made", [])),
        )

    def _merge_summary_objects(
        self,
        current: Union[GlobalMemorySummary, ProjectMemorySummary],
        incoming: Union[GlobalMemorySummary, ProjectMemorySummary],
        project_id: Optional[str] = None,
    ) -> Union[GlobalMemorySummary, ProjectMemorySummary]:
        """Merge summary objects without losing structured or provenance state."""
        merged_structured = incoming.structured_data or current.structured_data
        merged_claims = list(current.claims)
        seen_claims = {(claim.category, claim.text) for claim in merged_claims}
        for claim in incoming.claims:
            key = (claim.category, claim.text)
            if key not in seen_claims:
                merged_claims.append(claim)
                seen_claims.add(key)

        summary_kwargs = {
            "user_id": current.user_id,
            "summary_text": incoming.summary_text or current.summary_text,
            "last_synthesized": incoming.last_synthesized or current.last_synthesized,
            "last_modified": max(current.last_modified, incoming.last_modified),
            "source_conversation_ids": set(current.source_conversation_ids)
            | set(incoming.source_conversation_ids),
            "explicit_edits": list(current.explicit_edits) + [
                edit
                for edit in incoming.explicit_edits
                if edit not in current.explicit_edits
            ],
            "confidence_score": max(current.confidence_score, incoming.confidence_score),
            "structured_data": merged_structured,
            "claims": merged_claims,
        }

        if isinstance(current, ProjectMemorySummary) or project_id:
            return ProjectMemorySummary(
                project_id=project_id or current.project_id,
                **summary_kwargs,
            )
        return GlobalMemorySummary(**summary_kwargs)


# ============================================================================
# Simple Memory Extractor (Deterministic)
# ============================================================================


class SimpleMemoryExtractor:
    """
    Deterministic memory extraction without external LLM.
    Used as fallback when LLM is not available.
    """

    def __init__(self):
        self.max_sentence_chars = 280

    async def extract(
        self, conversation: ConversationContext, existing_memories: List[MemoryItem]
    ) -> List[MemoryItem]:
        """Extract memories from conversation."""
        if not conversation or not conversation.messages:
            return []

        existing_set = {m.content for m in existing_memories}

        candidates = self._split_user_sentences(conversation.messages)

        memories = []
        timestamp = datetime.now()

        for sentence in candidates:
            normalized = sentence.lower().strip()
            if not normalized or normalized in existing_set:
                continue

            memory_type = self._classify_sentence(sentence)
            if not memory_type:
                continue

            memory = MemoryItem(
                id=f"mem_{uuid.uuid4().hex[:16]}",
                user_id=conversation.user_id,
                memory_type=memory_type,
                content=sentence,
                created_at=timestamp,
                source_conversation_id=conversation.conversation_id,
                importance_score=self._get_importance(memory_type),
            )
            memories.append(memory)

        return memories

    def _split_user_sentences(self, messages: List[Dict]) -> List[str]:
        """Split user messages into candidate sentences."""
        sentences = []

        for msg in messages:
            if msg.get("role") != "user":
                continue

            content = msg.get("content", "")
            for chunk in re.split(r"[.!?]+", content):
                candidate = chunk.strip()
                if not candidate:
                    continue
                if len(candidate) > self.max_sentence_chars:
                    candidate = candidate[: self.max_sentence_chars].rstrip()
                if len(candidate) > 10:
                    sentences.append(candidate)

        return sentences

    def _classify_sentence(self, sentence: str) -> Optional[MemoryType]:
        """Classify sentence into memory type."""
        text = sentence.lower()

        if any(
            k in text for k in ["prefer", "like", "love", "dislike", "hate", "rather"]
        ):
            return MemoryType.PREFERENCE

        if any(
            k in text for k in ["i am", "i'm", "i work", "i live", "i study", "i built"]
        ):
            return MemoryType.FACT

        if any(
            k in text for k in ["interested in", "studying", "research", "learning"]
        ):
            return MemoryType.TOPIC

        keywords = ["project", "company", "system", "app", "engine"]
        if any(k in text for k in keywords):
            return MemoryType.ENTITY

        return None

    def _get_importance(self, memory_type: MemoryType) -> float:
        """Get importance score by type."""
        importance_map = {
            MemoryType.PREFERENCE: 0.8,
            MemoryType.FACT: 0.7,
            MemoryType.ENTITY: 0.6,
            MemoryType.TOPIC: 0.5,
            MemoryType.TOOL_USAGE: 0.4,
            MemoryType.CONVERSATION: 0.3,
        }
        return importance_map.get(memory_type, 0.5)


# ============================================================================
# Example Usage
# ============================================================================


async def example_usage():
    """Demonstrate complete memory system."""

    memory_manager = MemoryManager(
        config={
            "auto_extract": True,
            "extraction_interval": 5,
            "max_context_tokens": 2000,
            "synthesis_interval_hours": 24,
            "work_focused_filter": True,
        }
    )

    user_id = "user_456"

    conv = Conversation(
        conversation_id="conv_123",
        user_id=user_id,
        title="Neural Architecture Discussion",
        messages=[
            ConversationMessage(
                message_id="msg_1",
                role="user",
                content="I'm working on a frequency-based neural architecture",
            ),
            ConversationMessage(
                message_id="msg_2",
                role="assistant",
                content="That sounds fascinating! Tell me more about your approach.",
            ),
            ConversationMessage(
                message_id="msg_3",
                role="user",
                content="I'm merging concepts from 80s/90s neural nets with modern transformers",
            ),
        ],
    )

    await memory_manager.store_conversation(conv)

    print("Conversation stored in archive")

    summary = await memory_manager.synthesize_memory(user_id, MemoryScope.GLOBAL)
    print(f"\nSynthesized summary:\n{summary.to_xml()}")

    edit = memory_manager.process_explicit_edit(
        user_id=user_id,
        edit_type="add",
        content="User prefers detailed technical explanations",
        scope=MemoryScope.GLOBAL,
    )
    print(f"\nExplicit edit applied: {edit.edit_id}")

    prefs = memory_manager.preferences_store.get(user_id)
    prefs.communication_style = "technical"
    prefs.technical_expertise = "expert"
    memory_manager.preferences_store.update(
        user_id, {"communication_style": "technical", "technical_expertise": "expert"}
    )
    print(f"\nPreferences updated: {prefs.to_xml()}")

    context = await memory_manager.build_prompt_context(
        user_id=user_id,
        current_query="Help me implement this",
        include_summary=True,
        include_preferences=True,
    )
    print(f"\nPrompt context keys: {list(context.keys())}")

    summary_stats = await memory_manager.get_user_memory_summary(user_id)
    print(f"\nMemory Summary: {json.dumps(summary_stats, indent=2, default=str)}")


if __name__ == "__main__":
    asyncio.run(example_usage())

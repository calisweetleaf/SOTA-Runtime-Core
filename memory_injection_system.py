"""
Memory & Context Injection System

Provenance note:
- Built from open research and publicly available technical references.
- Uses public-provider memory patterns (including OpenAI as a leading public benchmark example) as implementation references only.
- Implemented independently as non-proprietary code in this repository.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np
from collections import defaultdict

# ============================================================================
# Data Structures
# ============================================================================

class MemoryType(Enum):
    FACT = "fact"  # "User is a Python developer"
    PREFERENCE = "preference"  # "Prefers concise explanations"
    ENTITY = "entity"  # "Works at CompanyX"
    TOPIC = "topic"  # "Interested in ML"
    CONVERSATION = "conversation"  # Full conversation summary
    TOOL_USAGE = "tool_usage"  # Patterns of tool usage


@dataclass
class MemoryItem:
    """Single memory entry"""
    id: str
    user_id: str
    memory_type: MemoryType
    content: str
    embedding: Optional[np.ndarray] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    confidence: float = 1.0
    source_conversation_id: Optional[str] = None
    
    # Relationships
    related_memories: Set[str] = field(default_factory=set)
    supersedes: Optional[str] = None  # For updates/corrections
    
    # Contextual metadata
    tags: Set[str] = field(default_factory=set)
    importance_score: float = 0.5
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'memory_type': self.memory_type.value,
            'content': self.content,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'confidence': self.confidence,
            'importance_score': self.importance_score,
            'tags': list(self.tags)
        }


@dataclass
class ConversationContext:
    """Current conversation state"""
    conversation_id: str
    user_id: str
    messages: List[Dict]  # [{"role": "user", "content": "..."}]
    current_topic: Optional[str] = None
    active_tools: Set[str] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)


@dataclass
class MemoryRetrievalResult:
    """Result of memory retrieval"""
    memories: List[MemoryItem]
    relevance_scores: List[float]
    retrieval_method: str
    query_embedding: Optional[np.ndarray] = None


# ============================================================================
# Memory Extraction (Structured Information Extraction)
# ============================================================================

class MemoryExtractor:
    """
    Extracts structured memories from conversation using LLM
    """
    def __init__(self, extraction_model="gpt-4o-mini"):
        self.extraction_model = extraction_model
        
        self.extraction_prompt = """Analyze the conversation and extract structured information:

1. FACTS: Concrete, verifiable information about the user
2. PREFERENCES: User's stated likes, dislikes, or working styles
3. ENTITIES: People, organizations, projects mentioned
4. TOPICS: Subject areas discussed
5. TOOL_USAGE: Patterns in how user employs tools

Return JSON:
{
  "facts": ["User is a 23-year-old developer", ...],
  "preferences": ["Prefers detailed technical explanations", ...],
  "entities": ["Working on frequency-based architecture project", ...],
  "topics": ["Neural networks", "ML research", ...],
  "tool_usage": ["Frequently uses code artifacts for implementations", ...]
}

Only include NEW information not already in existing memories.
Be conservative - only extract high-confidence information."""

    async def extract_from_conversation(
        self,
        conversation: ConversationContext,
        existing_memories: List[MemoryItem]
    ) -> List[MemoryItem]:
        """
        Extract new memories from conversation
        Uses LLM to perform structured extraction
        """
        # Format conversation for extraction
        conversation_text = self._format_conversation(conversation)
        existing_text = self._format_existing_memories(existing_memories)
        
        # Call extraction model (pseudo-code for API call)
        extracted = await self._call_extraction_api(
            conversation_text,
            existing_text
        )
        
        # Convert to MemoryItem objects
        memories = []
        timestamp = datetime.now()
        
        for fact in extracted.get('facts', []):
            memories.append(MemoryItem(
                id=self._generate_id(),
                user_id=conversation.user_id,
                memory_type=MemoryType.FACT,
                content=fact,
                created_at=timestamp,
                source_conversation_id=conversation.conversation_id,
                importance_score=0.7
            ))
        
        for pref in extracted.get('preferences', []):
            memories.append(MemoryItem(
                id=self._generate_id(),
                user_id=conversation.user_id,
                memory_type=MemoryType.PREFERENCE,
                content=pref,
                created_at=timestamp,
                source_conversation_id=conversation.conversation_id,
                importance_score=0.8  # Preferences are high importance
            ))
        
        for entity in extracted.get('entities', []):
            memories.append(MemoryItem(
                id=self._generate_id(),
                user_id=conversation.user_id,
                memory_type=MemoryType.ENTITY,
                content=entity,
                created_at=timestamp,
                source_conversation_id=conversation.conversation_id,
                importance_score=0.6
            ))
        
        for topic in extracted.get('topics', []):
            memories.append(MemoryItem(
                id=self._generate_id(),
                user_id=conversation.user_id,
                memory_type=MemoryType.TOPIC,
                content=topic,
                created_at=timestamp,
                source_conversation_id=conversation.conversation_id,
                importance_score=0.5
            ))
        
        return memories
    
    def _format_conversation(self, conversation: ConversationContext) -> str:
        """Format conversation for extraction"""
        lines = []
        for msg in conversation.messages[-10:]:  # Last 10 messages
            role = msg['role'].upper()
            content = msg['content']
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    
    def _format_existing_memories(self, memories: List[MemoryItem]) -> str:
        """Format existing memories to avoid duplication"""
        lines = [f"- {m.content}" for m in memories[:20]]  # Top 20
        return "\n".join(lines)
    
    async def _call_extraction_api(
        self,
        conversation: str,
        existing: str
    ) -> Dict:
        """
        Call LLM API for extraction
        Placeholder - implement with actual API
        """
        # In production, call a provider API (e.g., OpenAI/Anthropic) as a serving backend.
        # This implementation itself is independent and based on public references.
        # For now, return mock data
        return {
            "facts": [],
            "preferences": [],
            "entities": [],
            "topics": []
        }
    
    def _generate_id(self) -> str:
        """Generate unique memory ID"""
        import uuid
        return f"mem_{uuid.uuid4().hex[:16]}"


# ============================================================================
# Embedding & Semantic Search
# ============================================================================

class MemoryEmbedder:
    """
    Generates embeddings for memory items and queries
    """
    def __init__(self, embedding_model="text-embedding-3-large"):
        self.embedding_model = embedding_model
        self.embedding_dim = 3072  # text-embedding-3-large dimension
        
    async def embed_memory(self, memory: MemoryItem) -> np.ndarray:
        """
        Generate embedding for a memory item
        Includes type-specific contextual prefix
        """
        # Add contextual prefix based on memory type
        prefix = self._get_contextual_prefix(memory.memory_type)
        text = f"{prefix}{memory.content}"
        
        # Call embedding API
        embedding = await self._call_embedding_api(text)
        memory.embedding = embedding
        
        return embedding
    
    async def embed_query(
        self,
        query: str,
        query_context: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Generate query embedding with optional context
        """
        # Optionally add query context
        if query_context:
            context_str = self._format_query_context(query_context)
            text = f"{context_str}\n{query}"
        else:
            text = query
        
        return await self._call_embedding_api(text)
    
    def _get_contextual_prefix(self, memory_type: MemoryType) -> str:
        """
        Contextual prefix improves retrieval accuracy
        See: Anthropic's Contextual Retrieval blog post
        """
        prefixes = {
            MemoryType.FACT: "User fact: ",
            MemoryType.PREFERENCE: "User preference: ",
            MemoryType.ENTITY: "Related entity: ",
            MemoryType.TOPIC: "Interest area: ",
            MemoryType.TOOL_USAGE: "Tool usage pattern: "
        }
        return prefixes.get(memory_type, "")
    
    def _format_query_context(self, context: Dict) -> str:
        """Format query context for embedding"""
        parts = []
        if 'current_topic' in context:
            parts.append(f"Topic: {context['current_topic']}")
        if 'active_tools' in context:
            parts.append(f"Tools: {', '.join(context['active_tools'])}")
        return " | ".join(parts)
    
    async def _call_embedding_api(self, text: str) -> np.ndarray:
        """
        Call embedding API
        Placeholder - implement with actual API
        """
        # In production: call a provider embeddings API (e.g., OpenAI-compatible endpoint).
        # This implementation itself is independent and based on public references.
        # For now, return random embedding
        return np.random.randn(self.embedding_dim).astype(np.float32)


# ============================================================================
# Memory Storage & Retrieval
# ============================================================================

class MemoryStore:
    """
    Persistent memory storage with semantic search
    Uses vector database for efficient retrieval
    """
    def __init__(
        self,
        storage_backend="in_memory",  # Or "pinecone", "qdrant", etc.
        embedder: Optional[MemoryEmbedder] = None
    ):
        self.storage_backend = storage_backend
        self.embedder = embedder or MemoryEmbedder()
        
        # In-memory storage (replace with real DB in production)
        self.memories: Dict[str, MemoryItem] = {}
        self.user_memories: Dict[str, Set[str]] = defaultdict(set)
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_ids: List[str] = []
        
    async def store_memory(self, memory: MemoryItem) -> str:
        """
        Store a memory item
        """
        # Generate embedding if not present
        if memory.embedding is None:
            await self.embedder.embed_memory(memory)
        
        # Check for duplicates/conflicts
        memory = await self._deduplicate(memory)
        
        # Store
        self.memories[memory.id] = memory
        self.user_memories[memory.user_id].add(memory.id)
        
        # Update embedding index
        self._update_embedding_index(memory)
        
        return memory.id
    
    async def retrieve_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        memory_types: Optional[List[MemoryType]] = None,
        min_importance: float = 0.0
    ) -> MemoryRetrievalResult:
        """
        Retrieve relevant memories for a query
        """
        # Get user's memories
        user_memory_ids = self.user_memories.get(user_id, set())
        if not user_memory_ids:
            return MemoryRetrievalResult(
                memories=[],
                relevance_scores=[],
                retrieval_method="empty"
            )
        
        # Generate query embedding
        query_embedding = await self.embedder.embed_query(query)
        
        # Filter by type and importance
        candidate_memories = [
            self.memories[mid] for mid in user_memory_ids
            if (memory_types is None or self.memories[mid].memory_type in memory_types)
            and self.memories[mid].importance_score >= min_importance
        ]
        
        if not candidate_memories:
            return MemoryRetrievalResult(
                memories=[],
                relevance_scores=[],
                retrieval_method="filtered_empty"
            )
        
        # Compute similarities
        candidate_embeddings = np.stack([
            m.embedding for m in candidate_memories
        ])
        
        similarities = self._compute_similarity(
            query_embedding,
            candidate_embeddings
        )
        
        # Rank and select top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        top_memories = [candidate_memories[i] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]
        
        # Update access metadata
        for memory in top_memories:
            memory.last_accessed = datetime.now()
            memory.access_count += 1
        
        return MemoryRetrievalResult(
            memories=top_memories,
            relevance_scores=top_scores,
            retrieval_method="semantic_search",
            query_embedding=query_embedding
        )
    
    async def retrieve_by_type(
        self,
        user_id: str,
        memory_type: MemoryType,
        limit: int = 20
    ) -> List[MemoryItem]:
        """
        Retrieve memories by type (no semantic search)
        """
        user_memory_ids = self.user_memories.get(user_id, set())
        
        type_memories = [
            self.memories[mid] for mid in user_memory_ids
            if self.memories[mid].memory_type == memory_type
        ]
        
        # Sort by importance and recency
        type_memories.sort(
            key=lambda m: (m.importance_score, m.created_at),
            reverse=True
        )
        
        return type_memories[:limit]
    
    async def update_memory(
        self,
        memory_id: str,
        updates: Dict
    ) -> MemoryItem:
        """
        Update an existing memory
        """
        if memory_id not in self.memories:
            raise ValueError(f"Memory {memory_id} not found")
        
        memory = self.memories[memory_id]
        
        # Update fields
        if 'content' in updates:
            memory.content = updates['content']
            # Re-embed
            await self.embedder.embed_memory(memory)
            self._update_embedding_index(memory)
        
        if 'importance_score' in updates:
            memory.importance_score = updates['importance_score']
        
        if 'confidence' in updates:
            memory.confidence = updates['confidence']
        
        return memory
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory"""
        if memory_id not in self.memories:
            return False
        
        memory = self.memories[memory_id]
        del self.memories[memory_id]
        self.user_memories[memory.user_id].discard(memory_id)
        
        # Remove from embedding index
        self._remove_from_embedding_index(memory_id)
        
        return True
    
    async def _deduplicate(self, new_memory: MemoryItem) -> MemoryItem:
        """
        Check for duplicate or conflicting memories
        """
        user_memories = [
            self.memories[mid]
            for mid in self.user_memories.get(new_memory.user_id, set())
            if self.memories[mid].memory_type == new_memory.memory_type
        ]
        
        if not user_memories:
            return new_memory
        
        # Check for semantic duplicates
        similarities = self._compute_similarity(
            new_memory.embedding,
            np.stack([m.embedding for m in user_memories])
        )
        
        max_sim_idx = np.argmax(similarities)
        max_similarity = similarities[max_sim_idx]
        
        # If very similar (>0.95), it's likely a duplicate
        if max_similarity > 0.95:
            # Update existing memory instead of creating new
            existing = user_memories[max_sim_idx]
            existing.confidence = max(existing.confidence, new_memory.confidence)
            existing.importance_score = max(
                existing.importance_score,
                new_memory.importance_score
            )
            return existing
        
        # If moderately similar (0.8-0.95), mark as related
        elif max_similarity > 0.8:
            existing = user_memories[max_sim_idx]
            new_memory.related_memories.add(existing.id)
            existing.related_memories.add(new_memory.id)
        
        return new_memory
    
    def _compute_similarity(
        self,
        query_emb: np.ndarray,
        candidate_embs: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity
        """
        # Normalize
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        candidate_norms = candidate_embs / (
            np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-8
        )
        
        # Cosine similarity
        similarities = candidate_norms @ query_norm
        return similarities
    
    def _update_embedding_index(self, memory: MemoryItem):
        """Update the embedding index with new memory"""
        if memory.id in self.embedding_ids:
            # Update existing
            idx = self.embedding_ids.index(memory.id)
            self.embeddings[idx] = memory.embedding
        else:
            # Add new
            self.embedding_ids.append(memory.id)
            if self.embeddings is None:
                self.embeddings = memory.embedding.reshape(1, -1)
            else:
                self.embeddings = np.vstack([
                    self.embeddings,
                    memory.embedding.reshape(1, -1)
                ])
    
    def _remove_from_embedding_index(self, memory_id: str):
        """Remove memory from embedding index"""
        if memory_id not in self.embedding_ids:
            return
        
        idx = self.embedding_ids.index(memory_id)
        self.embedding_ids.pop(idx)
        
        if self.embeddings is not None:
            self.embeddings = np.delete(self.embeddings, idx, axis=0)


# ============================================================================
# Context Injection Manager
# ============================================================================

class ContextInjectionManager:
    """
    Manages injection of memories into prompts
    """
    def __init__(
        self,
        memory_store: MemoryStore,
        max_context_tokens: int = 2000
    ):
        self.memory_store = memory_store
        self.max_context_tokens = max_context_tokens
        
    async def build_context_block(
        self,
        user_id: str,
        current_query: str,
        conversation_context: Optional[ConversationContext] = None
    ) -> str:
        """
        Build memory context block to inject into prompt
        """
        # Retrieve relevant memories
        retrieval_result = await self.memory_store.retrieve_memories(
            user_id=user_id,
            query=current_query,
            top_k=15,
            min_importance=0.3
        )
        
        if not retrieval_result.memories:
            return ""
        
        # Group by type
        memories_by_type = defaultdict(list)
        for memory, score in zip(
            retrieval_result.memories,
            retrieval_result.relevance_scores
        ):
            memories_by_type[memory.memory_type].append((memory, score))
        
        # Build structured context
        context_parts = ["<memory>"]
        
        # Preferences (highest priority)
        if MemoryType.PREFERENCE in memories_by_type:
            context_parts.append("\n## User Preferences")
            for memory, score in memories_by_type[MemoryType.PREFERENCE][:3]:
                context_parts.append(f"- {memory.content}")
        
        # Facts
        if MemoryType.FACT in memories_by_type:
            context_parts.append("\n## User Context")
            for memory, score in memories_by_type[MemoryType.FACT][:5]:
                context_parts.append(f"- {memory.content}")
        
        # Entities
        if MemoryType.ENTITY in memories_by_type:
            context_parts.append("\n## Relevant Entities")
            for memory, score in memories_by_type[MemoryType.ENTITY][:3]:
                context_parts.append(f"- {memory.content}")
        
        # Topics
        if MemoryType.TOPIC in memories_by_type:
            topics = [m.content for m, s in memories_by_type[MemoryType.TOPIC][:5]]
            context_parts.append(f"\n## Interest Areas: {', '.join(topics)}")
        
        context_parts.append("\n</memory>")
        
        context_block = "\n".join(context_parts)
        
        # Truncate if too long
        context_block = self._truncate_to_token_limit(
            context_block,
            self.max_context_tokens
        )
        
        return context_block
    
    def _truncate_to_token_limit(
        self,
        text: str,
        max_tokens: int
    ) -> str:
        """
        Truncate text to token limit
        Rough approximation: 1 token ≈ 4 chars
        """
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        
        return text[:max_chars] + "\n... (truncated)\n</memory>"
    
    async def inject_into_messages(
        self,
        messages: List[Dict],
        user_id: str,
        current_query: str
    ) -> List[Dict]:
        """
        Inject memory context into message list
        
        Two strategies:
        1. Prepend to system message
        2. Insert as separate developer message
        """
        context_block = await self.build_context_block(
            user_id, current_query
        )
        
        if not context_block:
            return messages
        
        # Strategy 1: Prepend to existing system message
        if messages and messages[0].get('role') == 'system':
            messages[0]['content'] = (
                context_block + "\n\n" + messages[0]['content']
            )
            return messages
        
        # Strategy 2: Insert as developer message
        developer_msg = {
            'role': 'developer',
            'content': context_block
        }
        return [developer_msg] + messages


# ============================================================================
# Complete Memory Manager (Orchestrator)
# ============================================================================

class MemoryManager:
    """
    Complete memory system orchestrator
    Handles extraction, storage, retrieval, and injection
    """
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        
        self.embedder = MemoryEmbedder(
            embedding_model=config.get('embedding_model', 'text-embedding-3-large')
        )
        
        self.memory_store = MemoryStore(
            storage_backend=config.get('storage_backend', 'in_memory'),
            embedder=self.embedder
        )
        
        self.extractor = MemoryExtractor(
            extraction_model=config.get('extraction_model', 'gpt-4o-mini')
        )
        
        self.injector = ContextInjectionManager(
            memory_store=self.memory_store,
            max_context_tokens=config.get('max_context_tokens', 2000)
        )
        
        # Background processing config
        self.auto_extract = config.get('auto_extract', True)
        self.extraction_interval = config.get('extraction_interval', 5)  # messages
        
    async def process_conversation_turn(
        self,
        conversation: ConversationContext,
        should_extract: bool = None
    ) -> Optional[List[MemoryItem]]:
        """
        Process a conversation turn
        Optionally extract new memories
        """
        if should_extract is None:
            should_extract = self.auto_extract and (
                len(conversation.messages) % self.extraction_interval == 0
            )
        
        if not should_extract:
            return None
        
        # Get existing memories
        existing = await self.memory_store.retrieve_by_type(
            user_id=conversation.user_id,
            memory_type=None,  # All types
            limit=50
        )
        
        # Extract new memories
        new_memories = await self.extractor.extract_from_conversation(
            conversation, existing
        )
        
        # Store new memories
        for memory in new_memories:
            await self.memory_store.store_memory(memory)
        
        return new_memories
    
    async def prepare_prompt_with_memory(
        self,
        messages: List[Dict],
        user_id: str,
        conversation_context: Optional[ConversationContext] = None
    ) -> List[Dict]:
        """
        Prepare messages with injected memory context
        """
        # Extract current query
        current_query = ""
        for msg in reversed(messages):
            if msg['role'] == 'user':
                current_query = msg['content']
                break
        
        # Inject memory
        messages_with_memory = await self.injector.inject_into_messages(
            messages, user_id, current_query
        )
        
        return messages_with_memory
    
    async def get_user_memory_summary(
        self,
        user_id: str
    ) -> Dict[str, any]:
        """
        Get summary of user's memory bank
        """
        summary = {
            'total_memories': len(self.memory_store.user_memories.get(user_id, set())),
            'by_type': {},
            'top_topics': [],
            'last_updated': None
        }
        
        for memory_type in MemoryType:
            type_memories = await self.memory_store.retrieve_by_type(
                user_id, memory_type, limit=100
            )
            summary['by_type'][memory_type.value] = len(type_memories)
            
            if type_memories:
                latest = max(type_memories, key=lambda m: m.created_at)
                if summary['last_updated'] is None or latest.created_at > summary['last_updated']:
                    summary['last_updated'] = latest.created_at
        
        # Get top topics
        topic_memories = await self.memory_store.retrieve_by_type(
            user_id, MemoryType.TOPIC, limit=10
        )
        summary['top_topics'] = [m.content for m in topic_memories]
        
        return summary


# ============================================================================
# Example Usage
# ============================================================================

async def example_usage():
    """Demonstrate complete memory system"""
    
    # Initialize
    memory_manager = MemoryManager(config={
        'auto_extract': True,
        'extraction_interval': 5,
        'max_context_tokens': 2000
    })
    
    # Simulate conversation
    conversation = ConversationContext(
        conversation_id="conv_123",
        user_id="user_456",
        messages=[
            {
                "role": "user",
                "content": "I'm working on a frequency-based neural architecture"
            },
            {
                "role": "assistant",
                "content": "That sounds fascinating! Tell me more about your approach."
            },
            {
                "role": "user",
                "content": "I'm merging concepts from 80s/90s neural nets with modern transformers"
            }
        ],
        current_topic="neural_architecture",
        metadata={
            'user_tier': 'free',
            'message_count': 3
        }
    )
    
    # Process conversation and extract memories
    new_memories = await memory_manager.process_conversation_turn(
        conversation,
        should_extract=True
    )
    
    print(f"Extracted {len(new_memories)} new memories")
    for memory in new_memories:
        print(f"  - {memory.memory_type.value}: {memory.content}")
    
    # Later conversation - retrieve and inject context
    new_messages = [
        {"role": "user", "content": "Can you help me implement this architecture?"}
    ]
    
    messages_with_memory = await memory_manager.prepare_prompt_with_memory(
        messages=new_messages,
        user_id="user_456"
    )
    
    print("\nMessages with injected memory:")
    for msg in messages_with_memory:
        print(f"{msg['role']}: {msg['content'][:200]}...")
    
    # Get memory summary
    summary = await memory_manager.get_user_memory_summary("user_456")
    print(f"\nMemory Summary: {json.dumps(summary, indent=2, default=str)}")


if __name__ == '__main__':
    import asyncio
    asyncio.run(example_usage())

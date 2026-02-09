"""
Integrated Prompt Router + Memory System
Complete chat application orchestrator
"""

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import asyncio

# Assuming previous implementations are imported
# from router_system import NeuralPromptRouter, RouterConfig
# from memory_system import MemoryManager, ConversationContext


# ============================================================================
# Unified Chat Application Manager
# ============================================================================

@dataclass
class ChatRequest:
    """Incoming chat request"""
    user_id: str
    conversation_id: str
    message: str
    message_history: List[Dict]
    user_profile: Dict
    metadata: Dict


@dataclass
class ChatResponse:
    """Outgoing chat response"""
    response: str
    system_prompt: str
    memories_used: List[str]
    router_trace: Dict
    processing_time: float


class IntegratedChatSystem:
    """
    Complete chat system with neural routing + memory
    """
    def __init__(
        self,
        router_config: Dict,
        memory_config: Dict,
        enable_memory: bool = True,
        enable_neural_router: bool = True
    ):
        # Initialize prompt router
        from router_core_implementation import NeuralPromptRouter, RouterConfig
        self.router = NeuralPromptRouter(RouterConfig(**router_config))
        self.enable_neural_router = enable_neural_router
        
        # Initialize memory system
        from memory_injection_system import MemoryManager
        self.memory_manager = MemoryManager(config=memory_config)
        self.enable_memory = enable_memory
        
        # Fallback Jinja2 template (your existing one)
        self.jinja_template = self._load_jinja_template()
        
    async def process_chat(
        self,
        request: ChatRequest
    ) -> ChatResponse:
        """
        Main entry point for chat processing
        
        Flow:
        1. Retrieve relevant memories
        2. Build conversation context
        3. Route to appropriate system prompt
        4. Generate response (call LLM)
        5. Extract new memories from conversation
        6. Return response
        """
        import time
        start_time = time.time()
        
        # Step 1: Memory retrieval
        memories_used = []
        if self.enable_memory:
            enriched_messages = await self.memory_manager.prepare_prompt_with_memory(
                messages=request.message_history + [
                    {"role": "user", "content": request.message}
                ],
                user_id=request.user_id
            )
            
            # Track which memories were used
            if enriched_messages[0].get('role') == 'developer':
                memory_content = enriched_messages[0]['content']
                # Parse memory content to extract used memories
                # (simplified - in production, track this properly)
                memories_used = self._parse_memory_block(memory_content)
        else:
            enriched_messages = request.message_history + [
                {"role": "user", "content": request.message}
            ]
        
        # Step 2: Build conversation context for router
        conversation_context = ConversationContext(
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            messages=enriched_messages,
            current_topic=self._infer_topic(request.message),
            metadata=request.metadata
        )
        
        # Step 3: Generate system prompt via router
        router_trace = {}
        if self.enable_neural_router:
            system_prompt, router_trace = await self._route_with_neural(
                conversation_context,
                request
            )
        else:
            system_prompt = await self._route_with_jinja(
                conversation_context,
                request
            )
        
        # Step 4: Call LLM with generated system prompt
        # (This is where you'd call OpenAI/Anthropic API)
        response_text = await self._generate_response(
            system_prompt=system_prompt,
            messages=enriched_messages
        )
        
        # Step 5: Extract new memories (background task)
        if self.enable_memory:
            # Update conversation context with assistant response
            conversation_context.messages.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Extract memories asynchronously
            asyncio.create_task(
                self.memory_manager.process_conversation_turn(
                    conversation_context,
                    should_extract=True
                )
            )
        
        # Step 6: Build response
        processing_time = time.time() - start_time
        
        return ChatResponse(
            response=response_text,
            system_prompt=system_prompt,
            memories_used=memories_used,
            router_trace=router_trace,
            processing_time=processing_time
        )
    
    async def _route_with_neural(
        self,
        conversation_context: ConversationContext,
        request: ChatRequest
    ) -> Tuple[str, Dict]:
        """Use neural router to generate system prompt"""
        
        # Encode conversation for router
        inputs = self._prepare_router_inputs(
            conversation_context,
            request.user_profile
        )
        
        try:
            # Call router
            system_prompt, trace = self.router(
                **inputs,
                return_trace=True
            )
            return system_prompt, trace
            
        except Exception as e:
            # Fallback to Jinja2
            print(f"Neural router failed: {e}, falling back to Jinja2")
            system_prompt = await self._route_with_jinja(
                conversation_context,
                request
            )
            return system_prompt, {'method': 'fallback', 'error': str(e)}
    
    async def _route_with_jinja(
        self,
        conversation_context: ConversationContext,
        request: ChatRequest
    ) -> str:
        """Fallback to Jinja2 template"""
        
        # Extract features for Jinja2
        context = {
            'model_identity': 'You are ChatGPT, a large language model trained by OpenAI.',
            'reasoning_effort': self._infer_reasoning_effort(request),
            'builtin_tools': self._infer_tools(request),
            'tools': None,  # Custom tools
        }
        
        return self.jinja_template.render(**context)
    
    async def _generate_response(
        self,
        system_prompt: str,
        messages: List[Dict]
    ) -> str:
        """
        Call LLM API to generate response
        Placeholder - implement with actual API
        """
        # In production:
        # - Call OpenAI/Anthropic API
        # - Pass system_prompt and messages
        # - Handle streaming if needed
        
        # Mock response
        return "This is a mock response. In production, this would call the LLM API."
    
    def _prepare_router_inputs(
        self,
        conversation_context: ConversationContext,
        user_profile: Dict
    ) -> Dict:
        """
        Convert conversation context to router inputs
        """
        # Encode messages (simplified - use actual embeddings in production)
        message_embeddings = torch.randn(1, len(conversation_context.messages), 768)
        
        # Encode user profile
        profile_embedding = self._encode_user_profile(user_profile)
        
        # Encode metadata
        metadata_embedding = self._encode_metadata(conversation_context.metadata)
        
        return {
            'message_embs': message_embeddings,
            'user_profile': profile_embedding,
            'metadata': metadata_embedding,
            'context_metadata': conversation_context.metadata
        }
    
    def _encode_user_profile(self, profile: Dict) -> torch.Tensor:
        """Encode user profile to tensor"""
        # Simplified - in production, use learned embeddings
        features = []
        features.append(1.0 if profile.get('tier') == 'pro' else 0.0)
        features.extend([0.0] * 127)  # Pad to 128 dims
        return torch.tensor(features).unsqueeze(0)
    
    def _encode_metadata(self, metadata: Dict) -> torch.Tensor:
        """Encode conversation metadata to tensor"""
        features = []
        features.append(float(metadata.get('message_count', 0)) / 100.0)
        features.append(1.0 if metadata.get('has_code', False) else 0.0)
        features.extend([0.0] * 62)  # Pad to 64 dims
        return torch.tensor(features).unsqueeze(0)
    
    def _infer_topic(self, message: str) -> Optional[str]:
        """Infer conversation topic from message"""
        # Simplified topic detection
        keywords = {
            'code': ['code', 'programming', 'function', 'class'],
            'research': ['research', 'paper', 'study', 'analysis'],
            'creative': ['story', 'poem', 'creative', 'write'],
        }
        
        message_lower = message.lower()
        for topic, words in keywords.items():
            if any(word in message_lower for word in words):
                return topic
        return 'general'
    
    def _infer_reasoning_effort(self, request: ChatRequest) -> str:
        """Infer appropriate reasoning effort"""
        # Simplified inference
        if request.metadata.get('message_count', 0) < 3:
            return 'low'
        elif 'explain' in request.message.lower() or 'analyze' in request.message.lower():
            return 'high'
        else:
            return 'medium'
    
    def _infer_tools(self, request: ChatRequest) -> List[str]:
        """Infer which tools should be enabled"""
        tools = []
        
        message_lower = request.message.lower()
        
        if any(word in message_lower for word in ['search', 'find', 'look up', 'latest']):
            tools.append('browser')
        
        if any(word in message_lower for word in ['code', 'execute', 'run', 'calculate']):
            tools.append('python')
        
        return tools
    
    def _parse_memory_block(self, memory_content: str) -> List[str]:
        """Parse memory block to extract used memories"""
        # Simplified parser
        memories = []
        if '<memory>' in memory_content:
            # Extract bullet points
            lines = memory_content.split('\n')
            for line in lines:
                if line.strip().startswith('-'):
                    memories.append(line.strip()[2:])
        return memories
    
    def _load_jinja_template(self):
        """Load Jinja2 template (your existing one)"""
        # In production, load from file
        # For now, return mock
        class MockTemplate:
            def render(self, **kwargs):
                return f"<|start|>system<|message|>Mock template<|end|>"
        return MockTemplate()


# ============================================================================
# Enhanced Memory Features
# ============================================================================

class MemoryInsights:
    """
    Generate insights from memory bank
    Useful for user-facing memory management UI
    """
    def __init__(self, memory_manager: 'MemoryManager'):
        self.memory_manager = memory_manager
    
    async def get_memory_timeline(
        self,
        user_id: str,
        days: int = 30
    ) -> List[Dict]:
        """
        Get chronological timeline of memories
        """
        from datetime import datetime, timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        
        # Get all user memories
        all_memories = []
        for memory_type in ['fact', 'preference', 'entity', 'topic']:
            memories = await self.memory_manager.memory_store.retrieve_by_type(
                user_id, memory_type, limit=1000
            )
            all_memories.extend(memories)
        
        # Filter by date and sort
        recent_memories = [
            m for m in all_memories
            if m.created_at >= cutoff
        ]
        recent_memories.sort(key=lambda m: m.created_at, reverse=True)
        
        # Format for timeline
        timeline = []
        for memory in recent_memories:
            timeline.append({
                'date': memory.created_at.isoformat(),
                'type': memory.memory_type.value,
                'content': memory.content,
                'importance': memory.importance_score,
                'source_conversation': memory.source_conversation_id
            })
        
        return timeline
    
    async def get_memory_clusters(
        self,
        user_id: str
    ) -> Dict[str, List[str]]:
        """
        Cluster related memories by topic/theme
        """
        from collections import defaultdict
        
        # Get all memories
        all_memories = []
        for memory_type in ['fact', 'preference', 'entity', 'topic']:
            memories = await self.memory_manager.memory_store.retrieve_by_type(
                user_id, memory_type, limit=1000
            )
            all_memories.extend(memories)
        
        if not all_memories:
            return {}
        
        # Simple clustering by tags (in production, use proper clustering)
        clusters = defaultdict(list)
        for memory in all_memories:
            if memory.tags:
                for tag in memory.tags:
                    clusters[tag].append(memory.content)
            else:
                clusters['uncategorized'].append(memory.content)
        
        return dict(clusters)
    
    async def search_memories(
        self,
        user_id: str,
        query: str,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        User-facing memory search
        """
        memory_types = None
        if filters and 'types' in filters:
            memory_types = filters['types']
        
        min_importance = filters.get('min_importance', 0.0) if filters else 0.0
        
        retrieval_result = await self.memory_manager.memory_store.retrieve_memories(
            user_id=user_id,
            query=query,
            top_k=20,
            memory_types=memory_types,
            min_importance=min_importance
        )
        
        results = []
        for memory, score in zip(
            retrieval_result.memories,
            retrieval_result.relevance_scores
        ):
            results.append({
                'id': memory.id,
                'content': memory.content,
                'type': memory.memory_type.value,
                'relevance': float(score),
                'created': memory.created_at.isoformat(),
                'importance': memory.importance_score
            })
        
        return results


# ============================================================================
# API Endpoints (FastAPI Example)
# ============================================================================

"""
Example FastAPI endpoints for chat + memory management

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Initialize system
chat_system = IntegratedChatSystem(
    router_config={'context_dim': 768, 'num_templates': 16},
    memory_config={'auto_extract': True},
    enable_memory=True,
    enable_neural_router=True
)

memory_insights = MemoryInsights(chat_system.memory_manager)


class ChatRequestModel(BaseModel):
    user_id: str
    conversation_id: str
    message: str
    message_history: List[Dict]
    user_profile: Dict
    metadata: Dict


@app.post("/chat")
async def chat_endpoint(request: ChatRequestModel):
    chat_request = ChatRequest(**request.dict())
    response = await chat_system.process_chat(chat_request)
    
    return {
        'response': response.response,
        'memories_used': response.memories_used,
        'processing_time': response.processing_time
    }


@app.get("/memory/summary/{user_id}")
async def memory_summary(user_id: str):
    summary = await chat_system.memory_manager.get_user_memory_summary(user_id)
    return summary


@app.get("/memory/timeline/{user_id}")
async def memory_timeline(user_id: str, days: int = 30):
    timeline = await memory_insights.get_memory_timeline(user_id, days)
    return {'timeline': timeline}


@app.post("/memory/search")
async def search_memories(user_id: str, query: str, filters: Optional[Dict] = None):
    results = await memory_insights.search_memories(user_id, query, filters)
    return {'results': results}


@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str):
    success = await chat_system.memory_manager.memory_store.delete_memory(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {'success': True}
"""


# ============================================================================
# Memory Optimization Strategies
# ============================================================================

class MemoryOptimizer:
    """
    Optimize memory storage and retrieval
    """
    def __init__(self, memory_store: 'MemoryStore'):
        self.memory_store = memory_store
    
    async def consolidate_memories(
        self,
        user_id: str,
        similarity_threshold: float = 0.9
    ) -> int:
        """
        Merge highly similar memories to reduce redundancy
        """
        from memory_injection_system import MemoryType
        
        consolidated_count = 0
        
        for memory_type in MemoryType:
            memories = await self.memory_store.retrieve_by_type(
                user_id, memory_type, limit=1000
            )
            
            if len(memories) < 2:
                continue
            
            # Find similar pairs
            embeddings = np.stack([m.embedding for m in memories])
            
            for i in range(len(memories)):
                for j in range(i + 1, len(memories)):
                    similarity = self._cosine_similarity(
                        embeddings[i], embeddings[j]
                    )
                    
                    if similarity > similarity_threshold:
                        # Merge j into i
                        await self._merge_memories(memories[i], memories[j])
                        consolidated_count += 1
        
        return consolidated_count
    
    async def prune_low_importance(
        self,
        user_id: str,
        threshold: float = 0.2,
        min_age_days: int = 30
    ) -> int:
        """
        Remove low-importance old memories
        """
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=min_age_days)
        
        pruned_count = 0
        
        user_memory_ids = self.memory_store.user_memories.get(user_id, set())
        
        for memory_id in list(user_memory_ids):
            memory = self.memory_store.memories.get(memory_id)
            if not memory:
                continue
            
            if (memory.importance_score < threshold and
                memory.created_at < cutoff_date and
                memory.access_count < 2):
                
                await self.memory_store.delete_memory(memory_id)
                pruned_count += 1
        
        return pruned_count
    
    async def boost_frequently_accessed(
        self,
        user_id: str,
        access_threshold: int = 5
    ):
        """
        Increase importance of frequently accessed memories
        """
        user_memory_ids = self.memory_store.user_memories.get(user_id, set())
        
        for memory_id in user_memory_ids:
            memory = self.memory_store.memories.get(memory_id)
            if not memory:
                continue
            
            if memory.access_count >= access_threshold:
                # Boost importance
                new_importance = min(1.0, memory.importance_score + 0.1)
                await self.memory_store.update_memory(
                    memory_id,
                    {'importance_score': new_importance}
                )
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    
    async def _merge_memories(self, target: 'MemoryItem', source: 'MemoryItem'):
        """Merge source memory into target"""
        # Combine confidence and importance
        target.confidence = max(target.confidence, source.confidence)
        target.importance_score = max(target.importance_score, source.importance_score)
        target.access_count += source.access_count
        target.related_memories.add(source.id)
        
        # Delete source
        await self.memory_store.delete_memory(source.id)


# ============================================================================
# Example: Complete Usage Flow
# ============================================================================

async def complete_example():
    """
    End-to-end example of integrated system
    """
    
    # 1. Initialize system
    chat_system = IntegratedChatSystem(
        router_config={
            'context_dim': 768,
            'num_templates': 16,
            'num_tools': 32
        },
        memory_config={
            'auto_extract': True,
            'extraction_interval': 5,
            'max_context_tokens': 2000
        },
        enable_memory=True,
        enable_neural_router=True
    )
    
    # 2. Process initial chat
    request1 = ChatRequest(
        user_id="user_123",
        conversation_id="conv_abc",
        message="I'm working on a frequency-based neural architecture",
        message_history=[],
        user_profile={'tier': 'free', 'name': 'Daeron'},
        metadata={'message_count': 1, 'has_code': False}
    )
    
    response1 = await chat_system.process_chat(request1)
    print(f"Response 1: {response1.response}")
    print(f"Memories used: {response1.memories_used}")
    
    # 3. Continue conversation (memory will be injected)
    request2 = ChatRequest(
        user_id="user_123",
        conversation_id="conv_abc",
        message="Can you help me implement the frequency encoder?",
        message_history=[
            {"role": "user", "content": request1.message},
            {"role": "assistant", "content": response1.response}
        ],
        user_profile={'tier': 'free', 'name': 'Daeron'},
        metadata={'message_count': 3, 'has_code': True}
    )
    
    response2 = await chat_system.process_chat(request2)
    print(f"\nResponse 2: {response2.response}")
    print(f"Memories used: {response2.memories_used}")
    print(f"Router trace: {response2.router_trace}")
    
    # 4. Check memory bank
    memory_insights = MemoryInsights(chat_system.memory_manager)
    
    timeline = await memory_insights.get_memory_timeline("user_123", days=1)
    print(f"\nMemory Timeline:")
    for item in timeline:
        print(f"  {item['date']}: [{item['type']}] {item['content']}")
    
    # 5. Search memories
    search_results = await memory_insights.search_memories(
        "user_123",
        "neural architecture"
    )
    print(f"\nSearch Results:")
    for result in search_results:
        print(f"  Relevance {result['relevance']:.2f}: {result['content']}")
    
    # 6. Optimize memory
    optimizer = MemoryOptimizer(chat_system.memory_manager.memory_store)
    consolidated = await optimizer.consolidate_memories("user_123")
    print(f"\nConsolidated {consolidated} similar memories")


if __name__ == '__main__':
    import asyncio
    asyncio.run(complete_example())
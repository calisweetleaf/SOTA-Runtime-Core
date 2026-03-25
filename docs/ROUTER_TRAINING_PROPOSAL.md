# Neural Router Training Proposal

> **SOTA-Grade Training Pipeline for Prompt Routing**

This document specifies the complete training strategy for `neural_router.py` using the RLHF infrastructure in `rlhf.py`. The goal is to train a production-ready neural router that achieves indistinguishable performance from industry-grade prompt engineering systems.

---

## Executive Summary

| Aspect | Specification |
|--------|---------------|
| **Training Paradigm** | Multi-stage: SFT → RM → Policy Optimization |
| **Primary Method** | DPO (Direct Preference Optimization) on template pairs |
| **Fallback Method** | GRPO for self-generated preference data |
| **Inference Enhancement** | Best-of-N with reward model reranking |
| **Target Performance** | Template selection accuracy ≥95%, User satisfaction ≥90% |

---

## 1. Architecture Overview

### 1.1 Trainable Components

The neural router consists of **6 trainable nn.Modules** that form an end-to-end differentiable pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NEURAL ROUTER ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐                 │
│  │ HashText     │    │ Profile      │    │ Metadata      │                 │
│  │ Encoder      │    │ Encoder      │    │ Encoder       │                 │
│  │ (768-dim)    │    │ (128-dim)    │    │ (64-dim)      │                 │
│  └──────┬───────┘    └──────┬───────┘    └───────┬───────┘                 │
│         │                   │                    │                          │
│         └───────────────────┼────────────────────┘                          │
│                             ▼                                               │
│                   ┌───────────────────┐                                     │
│                   │  Context Encoder  │                                     │
│                   │  (Transformer)    │                                     │
│                   └─────────┬─────────┘                                     │
│                             ▼                                               │
│                   ┌───────────────────┐                                     │
│                   │  Slot Predictor   │                                     │
│                   │  Network          │                                     │
│                   └─────────┬─────────┘                                     │
│                             ▼                                               │
│                   ┌───────────────────┐                                     │
│                   │ Template Selector │                                     │
│                   │ Network           │                                     │
│                   └─────────┬─────────┘                                     │
│                             ▼                                               │
│                   [Template Weights]                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Parameter Count

| Module | Parameters | Trainable |
|--------|-----------|-----------|
| HashTextEncoder | ~38.5M | ✓ |
| ProfileEncoder | ~0.1M | ✓ |
| MetadataEncoder | ~0.05M | ✓ |
| ContextEncoder | ~3.2M | ✓ |
| SlotPredictorNetwork | ~0.8M | ✓ |
| TemplateSelectorNetwork | ~0.2M | ✓ |
| **Total** | **~42.85M** | ✓ |

---

## 2. Training Data Specification

### 2.1 Data Formats

#### SFT Data (Stage 1)

```json
{
  "context": {
    "messages": [
      {"role": "user", "content": "Write me python code to sort a list"},
      {"role": "assistant", "content": "Here's a sorting function..."}
    ],
    "user_tier": "free",
    "has_tool_calls": false
  },
  "target_template": "code_generation",
  "target_slots": {
    "reasoning_effort": "medium",
    "tool_enables": {"python": true, "browser": false},
    "tool_weights": [0.8, 0.1, 0.05, 0.05]
  }
}
```

#### Preference Data (Stage 2-3)

```json
{
  "context": {
    "messages": [...],
    "user_tier": "paid"
  },
  "prompt_chosen": "<generated prompt from template A>",
  "prompt_rejected": "<generated prompt from template B>",
  "chosen_template": "analytical_reasoning",
  "rejected_template": "casual_chat",
  "quality_delta": 0.35
}
```

#### GRPO Rollout Data (Alternative Stage 3)

```json
{
  "context": {...},
  "rollouts": [
    {"template": "A", "prompt": "...", "llm_response": "...", "reward": 0.82},
    {"template": "B", "prompt": "...", "llm_response": "...", "reward": 0.65},
    {"template": "C", "prompt": "...", "llm_response": "...", "reward": 0.91}
  ],
  "best_template": "C"
}
```

### 2.2 Data Collection Strategy

| Source | Method | Volume Target |
|--------|--------|---------------|
| Synthetic | GPT-4 generated context → template pairs | 50K samples |
| User Feedback | A/B testing with explicit preference | 10K samples |
| Implicit Signals | Response regeneration rate, edit distance | 100K samples |
| Self-Play | Router vs Router with reward model judge | 500K samples |

---

## 3. Training Pipeline

### 3.1 Stage 0: Pre-training (Optional)

**Objective**: Initialize encoders with meaningful representations before task-specific training.

```python
# Pre-train HashTextEncoder on contrastive objective
pretrain_config = {
    "method": "SimCSE",
    "batch_size": 512,
    "epochs": 10,
    "learning_rate": 1e-4,
    "data": "conversation_corpus.jsonl"  # 1M+ conversations
}
```

### 3.2 Stage 1: Supervised Fine-Tuning (SFT)

**Objective**: Learn basic context → template mapping from labeled data.

```python
from rlhf import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    learning_rate=2e-5,
    batch_size=32,
    epochs=5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    gradient_accumulation_steps=4
)

# Router as the "model" (it's an nn.Module)
trainer = SFTTrainer(
    model=neural_router,
    config=sft_config,
    train_dataset=RouterSFTDataset(sft_data),
    eval_dataset=RouterSFTDataset(eval_data)
)

trainer.train()
```

**Loss Function**:

```python
def router_sft_loss(predictions, targets):
    """
    Multi-task loss for router SFT.
    
    Components:
    - Template selection: CrossEntropy
    - Slot prediction: BCE for binary, CrossEntropy for categorical
    - Tool weights: MSE
    """
    template_loss = F.cross_entropy(
        predictions.template_logits, 
        targets.template_idx
    )
    
    reasoning_loss = F.cross_entropy(
        predictions.reasoning_logits,
        targets.reasoning_level  # 0=low, 1=medium, 2=high
    )
    
    tool_enable_loss = F.binary_cross_entropy_with_logits(
        predictions.tool_enables,
        targets.tool_enables
    )
    
    tool_weight_loss = F.mse_loss(
        predictions.tool_weights,
        targets.tool_weights
    )
    
    return (
        template_loss + 
        0.5 * reasoning_loss + 
        0.3 * tool_enable_loss + 
        0.2 * tool_weight_loss
    )
```

### 3.3 Stage 2: Reward Model Training

**Objective**: Train a reward model to score prompt quality.

```python
from rlhf import RewardModelTrainer, RewardModelConfig

rm_config = RewardModelConfig(
    learning_rate=1e-5,
    batch_size=16,
    epochs=3,
    ensemble_size=3,  # Ensemble for uncertainty estimation
    margin=0.1
)

# RewardModel scores generated prompts
reward_model = RewardModel(
    base_model="microsoft/deberta-v3-base",  # Efficient encoder
    hidden_size=768,
    num_labels=1
)

rm_trainer = RewardModelTrainer(
    model=reward_model,
    config=rm_config,
    train_dataset=PromptPreferenceDataset(preference_data)
)

rm_trainer.train()
```

**Reward Model Architecture**:

```
┌─────────────────────────────────────────────────┐
│           PROMPT REWARD MODEL                   │
├─────────────────────────────────────────────────┤
│  [Generated Prompt] + [Original Context]        │
│            ▼                                    │
│  ┌─────────────────┐                           │
│  │  DeBERTa-v3     │                           │
│  │  Encoder        │                           │
│  └────────┬────────┘                           │
│           ▼                                    │
│  ┌─────────────────┐                           │
│  │  Reward Head    │                           │
│  │  (2-layer MLP)  │                           │
│  └────────┬────────┘                           │
│           ▼                                    │
│     Scalar Reward ∈ [-1, 1]                    │
└─────────────────────────────────────────────────┘
```

### 3.4 Stage 3: Policy Optimization

**Primary: DPO (Direct Preference Optimization)**

```python
from rlhf import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    learning_rate=5e-6,
    batch_size=8,
    epochs=2,
    beta=0.1,  # KL penalty coefficient
    loss_type="sigmoid",  # sigmoid | hinge | ipo
    reference_free=False
)

# Create reference model (frozen copy)
reference_router = copy.deepcopy(neural_router)
for param in reference_router.parameters():
    param.requires_grad = False

dpo_trainer = DPOTrainer(
    model=neural_router,
    ref_model=reference_router,
    config=dpo_config,
    train_dataset=RouterPreferenceDataset(preference_data)
)

dpo_trainer.train()
```

**DPO Loss for Router**:

```python
def router_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """
    DPO loss adapted for router template selection.
    
    Instead of token log-probs, we use template selection log-probs.
    """
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss
```

**Alternative: GRPO (Group Relative Policy Optimization)**

For scenarios with reward model but no explicit preferences:

```python
from rlhf import GRPOTrainer, GRPOConfig

grpo_config = GRPOConfig(
    learning_rate=1e-5,
    batch_size=4,
    group_size=8,  # Generate 8 candidates per context
    epochs=3,
    kl_coef=0.05,
    clip_range=0.2
)

grpo_trainer = GRPOTrainer(
    model=neural_router,
    reward_model=reward_model,
    config=grpo_config,
    train_dataset=RouterPromptDataset(contexts)
)

grpo_trainer.train()
```

**Recommended: SimPO (Reference-Free)**

SimPO requires no reference model, is 20% faster than DPO, and uses 10% less memory:

```python
from rlhf import SimPOTrainer, SimPOConfig

simpo_config = SimPOConfig(
    learning_rate=5e-6,
    batch_size=8,
    epochs=2,
    beta=2.0,   # Scaling factor for implicit reward
    gamma=0.5,  # Target reward margin
    label_smoothing=0.0
)

# SimPO only needs policy model - no reference!
simpo_trainer = SimPOTrainer(
    model=neural_router,
    config=simpo_config,
    train_dataset=RouterPreferenceDataset(preference_data)
)

simpo_trainer.train()
```

**SimPO Loss for Router**:

```python
def router_simpo_loss(
    chosen_logps: torch.Tensor,    # log(template_weights[chosen_idx])
    rejected_logps: torch.Tensor,  # log(template_weights[rejected_idx])
    beta: float = 2.0,
    gamma: float = 0.5
) -> torch.Tensor:
    """
    SimPO uses length-normalized log-probs as implicit reward.
    For router: "length" = 1 (single template selection).
    """
    # Implicit rewards (already length-normalized since length=1)
    r_chosen = chosen_logps
    r_rejected = rejected_logps
    
    # SimPO objective with margin
    logits = beta * (r_chosen - r_rejected - gamma)
    loss = -F.logsigmoid(logits).mean()
    return loss
```

---

## 4. RLHF Integration Architecture

### 4.1 Component Mapping

| RLHF Component | Router Equivalent | Integration Point |
|----------------|------------------|-------------------|
| `PolicyModel` | `NeuralPromptRouter` | Wrapped as policy |
| `RewardModel` | `PromptRewardModel` | Scores generated prompts |
| `ValueModel` | `RouterValueModel` | Optional, for PPO only |
| `SFTDataset` | `RouterSFTDataset` | Context → template pairs |
| `PreferenceDataset` | `RouterPreferenceDataset` | Template A vs B |

### 4.2 Integration Code

```python
# train_router.py

import torch
from neural_router import NeuralPromptRouter, RouterConfig
from rlhf import (
    RLHFOrchestrator,
    SFTConfig, RewardModelConfig, DPOConfig,
    DeviceManager, TrainingLogger, CheckpointManager
)

class RouterRLHFOrchestrator(RLHFOrchestrator):
    """
    Specialized orchestrator for neural router training.
    """
    
    def __init__(self, router_config: RouterConfig, **kwargs):
        self.router = NeuralPromptRouter(router_config)
        super().__init__(
            policy_model=RouterPolicyWrapper(self.router),
            **kwargs
        )
    
    def run_full_pipeline(
        self,
        sft_data: List[Dict],
        preference_data: List[Dict],
        eval_prompts: List[str]
    ) -> Dict[str, Any]:
        """
        Execute complete training pipeline.
        """
        history = {}
        
        # Stage 1: SFT
        self.logger.info("Stage 1: Supervised Fine-Tuning")
        sft_history = self.run_sft(sft_data)
        history['sft'] = sft_history
        
        # Stage 2: Reward Model
        self.logger.info("Stage 2: Reward Model Training")
        rm_history = self.run_reward_model_training(preference_data)
        history['rm'] = rm_history
        
        # Stage 3: DPO
        self.logger.info("Stage 3: DPO Policy Optimization")
        dpo_history = self.run_policy_optimization(
            preference_data,
            method='dpo'
        )
        history['dpo'] = dpo_history
        
        # Stage 4: Evaluation
        self.logger.info("Stage 4: Evaluation")
        eval_results = self.evaluate(eval_prompts)
        history['eval'] = eval_results
        
        return history


class RouterPolicyWrapper(torch.nn.Module):
    """
    Wraps NeuralPromptRouter to match RLHF PolicyModel interface.
    """
    
    def __init__(self, router: NeuralPromptRouter):
        super().__init__()
        self.router = router
    
    def forward(self, contexts: List[Dict]) -> Tuple[List[str], torch.Tensor]:
        """
        Generate prompts and return log-probabilities.
        """
        prompts = []
        log_probs = []
        
        for ctx in contexts:
            prompt, trace = self.router(
                message_embs=ctx['message_embs'],
                user_profile=ctx['user_profile'],
                metadata=ctx['metadata'],
                context_metadata=ctx['context_metadata'],
                return_trace=True
            )
            prompts.append(prompt)
            
            # Log-prob of selected template
            template_logp = torch.log(trace['template_weights'].max())
            log_probs.append(template_logp)
        
        return prompts, torch.stack(log_probs)
    
    def get_log_probs(self, contexts: List[Dict], templates: List[int]) -> torch.Tensor:
        """
        Get log-probabilities for specific template selections.
        """
        log_probs = []
        
        for ctx, template_idx in zip(contexts, templates):
            # Forward pass
            context_emb = self.router.context_encoder(
                ctx['message_embs'],
                ctx['user_profile'],
                ctx['metadata']
            )
            slot_preds = self.router.slot_predictor(
                context_emb,
                self.router.tool_embeddings
            )
            template_weights = self.router.template_selector(slot_preds)
            
            # Log-prob of specified template
            log_probs.append(torch.log(template_weights[0, template_idx] + 1e-8))
        
        return torch.stack(log_probs)
```

---

## 5. Inference-Time Enhancements

### 5.1 Best-of-N Sampling

Generate multiple template selections and pick the best:

```python
from inference_optimizations import BestOfNSampler

class RouterBestOfN:
    """
    Generate N prompts, rerank with reward model.
    """
    
    def __init__(
        self,
        router: NeuralPromptRouter,
        reward_model: RewardModel,
        n: int = 4,
        temperature: float = 1.0
    ):
        self.router = router
        self.reward_model = reward_model
        self.n = n
        self.temperature = temperature
    
    def generate(self, context: Dict) -> Tuple[str, Dict]:
        """
        Generate N candidates, return best.
        """
        candidates = []
        
        for _ in range(self.n):
            # Sample with temperature
            prompt, trace = self.router.sample(
                context,
                temperature=self.temperature
            )
            
            # Score with reward model
            reward = self.reward_model.score(prompt, context)
            
            candidates.append({
                'prompt': prompt,
                'reward': reward,
                'trace': trace
            })
        
        # Select best
        best = max(candidates, key=lambda x: x['reward'])
        
        return best['prompt'], {
            'method': 'best_of_n',
            'n_candidates': self.n,
            'best_reward': best['reward'],
            'all_rewards': [c['reward'] for c in candidates]
        }
```

### 5.2 MCTS for Complex Routing

For multi-step routing decisions:

```python
from inference_optimizations import MCTSGenerator

class RouterMCTS:
    """
    Tree search for optimal template + slot combinations.
    """
    
    def __init__(
        self,
        router: NeuralPromptRouter,
        value_model: ValueModel,
        n_simulations: int = 50
    ):
        self.router = router
        self.value_model = value_model
        self.n_simulations = n_simulations
    
    def search(self, context: Dict) -> Tuple[str, Dict]:
        """
        MCTS search over routing decisions.
        
        Tree structure:
        - Root: Initial context
        - Level 1: Template selection (16 branches)
        - Level 2: Reasoning effort (3 branches)
        - Level 3: Tool configuration (varies)
        """
        root = MCTSNode(context=context)
        
        for _ in range(self.n_simulations):
            # Selection
            node = self.select(root)
            
            # Expansion
            if not node.is_terminal:
                node = self.expand(node)
            
            # Simulation
            reward = self.simulate(node)
            
            # Backpropagation
            self.backpropagate(node, reward)
        
        # Return best path
        best_path = self.get_best_path(root)
        prompt = self.router.generate_from_path(context, best_path)
        
        return prompt, {'method': 'mcts', 'path': best_path}
```

---

## 6. Evaluation Metrics

### 6.1 Offline Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Template Accuracy | Exact match on template selection | ≥95% |
| Slot MSE | Mean squared error on slot predictions | ≤0.05 |
| Ranking Accuracy | Correct preference ordering | ≥90% |
| KL Divergence | Drift from reference policy | ≤0.5 |

### 6.2 Online Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Response Quality | User rating of LLM response | ≥4.5/5 |
| Regeneration Rate | How often user asks for redo | ≤5% |
| Task Completion | User achieves stated goal | ≥85% |
| Latency | Router inference time | ≤10ms |

### 6.3 Evaluation Code

```python
from rlhf import RLHFEvaluator

class RouterEvaluator(RLHFEvaluator):
    """
    Comprehensive router evaluation suite.
    """
    
    def evaluate(
        self,
        router: NeuralPromptRouter,
        test_data: List[Dict]
    ) -> Dict[str, float]:
        """
        Run full evaluation suite.
        """
        results = {}
        
        # Template accuracy
        template_correct = 0
        for sample in test_data:
            pred = router.predict_template(sample['context'])
            if pred == sample['target_template']:
                template_correct += 1
        results['template_accuracy'] = template_correct / len(test_data)
        
        # Slot MSE
        slot_errors = []
        for sample in test_data:
            pred_slots = router.predict_slots(sample['context'])
            target_slots = sample['target_slots']
            error = self.compute_slot_mse(pred_slots, target_slots)
            slot_errors.append(error)
        results['slot_mse'] = sum(slot_errors) / len(slot_errors)
        
        # Ranking accuracy (needs preference data)
        if 'preferences' in test_data[0]:
            ranking_correct = 0
            for sample in test_data:
                chosen_score = router.score(sample['context'], sample['chosen'])
                rejected_score = router.score(sample['context'], sample['rejected'])
                if chosen_score > rejected_score:
                    ranking_correct += 1
            results['ranking_accuracy'] = ranking_correct / len(test_data)
        
        # Diversity (unique templates per context type)
        template_usage = defaultdict(set)
        for sample in test_data:
            ctx_type = sample.get('context_type', 'general')
            pred = router.predict_template(sample['context'])
            template_usage[ctx_type].add(pred)
        results['template_diversity'] = {
            k: len(v) for k, v in template_usage.items()
        }
        
        return results
```

---

## 7. Training Infrastructure

### 7.1 Hardware Requirements

| Scale | GPU | VRAM | Batch Size | Training Time |
|-------|-----|------|------------|---------------|
| Dev | 1x RTX 3090 | 24GB | 8 | ~4 hours |
| Standard | 1x A100 | 40GB | 32 | ~2 hours |
| Production | 4x A100 | 160GB | 128 | ~30 min |

### 7.2 Checkpointing

```python
from rlhf import CheckpointManager

checkpoint_manager = CheckpointManager(
    save_dir="checkpoints/router",
    save_steps=500,
    keep_last_n=5,
    save_best=True,
    metric_for_best="eval/template_accuracy"
)
```

### 7.3 Logging

```python
from rlhf import TrainingLogger

logger = TrainingLogger(
    project_name="neural_router",
    run_name="dpo_v1",
    use_wandb=True,
    log_every=10
)
```

---

## 8. Deployment

### 8.1 Model Export

```python
# Save trained router
router.save_pretrained("models/router_v1")

# Load for inference
router = NeuralPromptRouter.from_pretrained("models/router_v1")
```

### 8.2 ONNX Export (Optional)

```python
import torch.onnx

# Export for fast inference
dummy_input = {
    'message_embs': torch.randn(1, 10, 768),
    'user_profile': torch.randn(1, 128),
    'metadata': torch.randn(1, 64)
}

torch.onnx.export(
    router,
    dummy_input,
    "models/router_v1.onnx",
    opset_version=14,
    input_names=['message_embs', 'user_profile', 'metadata'],
    output_names=['template_weights', 'slot_predictions']
)
```

---

## 9. CLI Specification

```bash
# train_router.py CLI

# Full pipeline
python train_router.py \
    --sft-data data/sft_samples.jsonl \
    --preference-data data/preferences.jsonl \
    --output-dir models/router_v1 \
    --stages sft,rm,dpo \
    --device cuda \
    --wandb-project neural_router

# SFT only
python train_router.py \
    --sft-data data/sft_samples.jsonl \
    --stages sft \
    --epochs 5 \
    --batch-size 32

# DPO fine-tuning from checkpoint
python train_router.py \
    --preference-data data/preferences.jsonl \
    --checkpoint models/router_sft \
    --stages dpo \
    --dpo-beta 0.1

# Evaluation
python train_router.py \
    --eval-only \
    --checkpoint models/router_v1 \
    --eval-data data/test_samples.jsonl
```

---

## 10. Timeline & Milestones

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| 1 | Data Pipeline | RouterSFTDataset, RouterPreferenceDataset |
| 2 | SFT Training | Trained base router, eval metrics |
| 3 | Reward Model | PromptRewardModel, ranking accuracy |
| 4 | DPO Training | Policy-optimized router |
| 5 | Inference Opt | Best-of-N, MCTS integration |
| 6 | Evaluation | Comprehensive metrics, ablations |
| 7 | Documentation | Training guide, API docs |
| 8 | Release | v1.0 with pre-trained weights |

---

## Appendix A: Data Collection Script

```py
# collect_training_data.py

import json
from typing import List, Dict

def generate_synthetic_sft_data(
    num_samples: int = 10000,
    templates: List[str] = None
) -> List[Dict]:
    """
    Generate synthetic SFT data using an LLM.
    """
    samples = []
    
    for _ in range(num_samples):
        # Generate diverse contexts
        context = generate_random_context()
        
        # Determine best template (can use KIMI-K.25-thinking as oracle)
        best_template = oracle_template_selection(context)
        
        # Generate slot targets
        slots = oracle_slot_prediction(context, best_template)
        
        samples.append({
            'context': context,
            'target_template': best_template,
            'target_slots': slots
        })
    
    return samples


def collect_preference_data(
    router: NeuralPromptRouter,
    contexts: List[Dict],
    human_annotators: bool = False
) -> List[Dict]:
    """
    Collect preference data for DPO training.
    """
    preferences = []
    
    for context in contexts:
        # Generate two candidate prompts
        prompt_a, trace_a = router.sample(context, temperature=1.0)
        prompt_b, trace_b = router.sample(context, temperature=1.0)
        
        if human_annotators:
            # Get human preference
            preference = get_human_preference(context, prompt_a, prompt_b)
        else:
            # Use reward model as proxy
            reward_a = reward_model.score(prompt_a, context)
            reward_b = reward_model.score(prompt_b, context)
            preference = 'a' if reward_a > reward_b else 'b'
        
        if preference == 'a':
            chosen, rejected = prompt_a, prompt_b
            chosen_template = trace_a['selected_template']
            rejected_template = trace_b['selected_template']
        else:
            chosen, rejected = prompt_b, prompt_a
            chosen_template = trace_b['selected_template']
            rejected_template = trace_a['selected_template']
        
        preferences.append({
            'context': context,
            'prompt_chosen': chosen,
            'prompt_rejected': rejected,
            'chosen_template': chosen_template,
            'rejected_template': rejected_template
        })
    
    return preferences
```

---

## Appendix B: Hyperparameter Sweep

```python
# sweep_config.yaml

program: train_router.py
method: bayes
metric:
  name: eval/template_accuracy
  goal: maximize

parameters:
  learning_rate:
    min: 1e-6
    max: 1e-4
    distribution: log_uniform
  
  batch_size:
    values: [8, 16, 32, 64]
  
  dpo_beta:
    min: 0.01
    max: 0.5
    distribution: uniform
  
  warmup_ratio:
    values: [0.0, 0.05, 0.1, 0.2]
  
  dropout:
    values: [0.0, 0.1, 0.2]
```

---

*Document Version: 1.0*
*Last Updated: 2026-02-09*
*Author: Christian Trey Rowell*

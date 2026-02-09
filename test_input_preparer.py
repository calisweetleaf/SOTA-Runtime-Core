"""Quick verification test for neural_router.py InputPreparer implementation."""
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("NEURAL ROUTER INPUT PREPARER VERIFICATION")
print("=" * 60)

# Test 1: Import
print("\n[1] Testing imports...")
from neural_router import (
    RouterConfig, InputPreparer, HashTextEncoder, 
    ProfileEncoder, MetadataEncoder, SafeRouterWrapper,
    NeuralPromptRouter
)
print("    ✓ All imports successful")

# Test 2: Config
print("\n[2] Testing RouterConfig...")
config = RouterConfig(context_dim=768, num_templates=16, num_tools=32)
print(f"    ✓ Config created: dim={config.context_dim}, dropout={config.dropout}")

# Test 3: InputPreparer initialization
print("\n[3] Testing InputPreparer initialization...")
preparer = InputPreparer(config)
print(f"    ✓ InputPreparer created")
print(f"    ✓ HashTextEncoder: {preparer.text_encoder.vocab_buckets} buckets, {preparer.text_encoder.num_hashes} hashes")
print(f"    ✓ ProfileEncoder: output_dim={preparer.profile_encoder.output_dim}")
print(f"    ✓ MetadataEncoder: output_dim={preparer.metadata_encoder.output_dim}")

# Test 4: Prepare with real context
print("\n[4] Testing context encoding...")
test_context = {
    'messages': [
        {'role': 'user', 'content': 'Hello, can you help me write Python code?'},
        {'role': 'assistant', 'content': 'Of course! What would you like to build?'},
        {'role': 'user', 'content': 'I want to create a neural network for text classification.'}
    ],
    'user_tier': 'free',
    'has_tool_calls': False,
    'message_count': 3
}

result = preparer.prepare(test_context)
print(f"    ✓ message_embs shape: {result['message_embs'].shape}")
print(f"    ✓ user_profile shape: {result['user_profile'].shape}")
print(f"    ✓ metadata shape: {result['metadata'].shape}")

# Test 5: Verify shapes match expected
print("\n[5] Verifying tensor shapes...")
assert result['message_embs'].shape == (1, 10, 768), f"Bad msg shape: {result['message_embs'].shape}"
assert result['user_profile'].shape == (1, 128), f"Bad profile shape: {result['user_profile'].shape}"
assert result['metadata'].shape == (1, 64), f"Bad metadata shape: {result['metadata'].shape}"
print("    ✓ All shapes correct!")

# Test 6: HashTextEncoder directly
print("\n[6] Testing HashTextEncoder directly...")
text_encoder = preparer.text_encoder
test_texts = ["Hello world", "def foo(): pass", "What is machine learning?"]
encoded = text_encoder(test_texts)
print(f"    ✓ Batch encoding shape: {encoded.shape}")
assert encoded.shape == (3, 512, 768), f"Bad encoding shape: {encoded.shape}"
print("    ✓ Batch encoding correct!")

# Test 7: Empty context handling
print("\n[7] Testing empty context handling...")
empty_context = {}
empty_result = preparer.prepare(empty_context)
print(f"    ✓ Empty context handled gracefully")
print(f"    ✓ Fallback shapes: {empty_result['message_embs'].shape}")

# Test 8: Full router test
print("\n[8] Testing full NeuralPromptRouter...")
router = NeuralPromptRouter(config)
print(f"    ✓ Router created with {len(router.template_library.templates)} templates")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)

(.venv) PS C:\Users\treyr\Desktop\Dev-Drive\SOTA_Runtime_Tools> python test_input_preparer.py        
============================================================
NEURAL ROUTER INPUT PREPARER VERIFICATION
============================================================

[1] Testing imports...
    ✓ All imports successful

[2] Testing RouterConfig...
    ✓ Config created: dim=768, dropout=0.1

[3] Testing InputPreparer initialization...
    ✓ InputPreparer created
    ✓ HashTextEncoder: 50000 buckets, 4 hashes
    ✓ ProfileEncoder: output_dim=128
    ✓ MetadataEncoder: output_dim=64

[4] Testing context encoding...
    ✓ message_embs shape: torch.Size([1, 10, 768])
    ✓ user_profile shape: torch.Size([1, 128])
    ✓ metadata shape: torch.Size([1, 64])

[5] Verifying tensor shapes...
    ✓ All shapes correct!

[6] Testing HashTextEncoder directly...
    ✓ Batch encoding shape: torch.Size([3, 512, 768])
    ✓ Batch encoding correct!

[7] Testing empty context handling...
    ✓ Empty context handled gracefully
    ✓ Fallback shapes: torch.Size([1, 10, 768])

[8] Testing full NeuralPromptRouter...
[templates] loaded og_jinja2_template.jinja2
[templates] loaded system_prompt.jinja2
[templates] loaded tool_manifest.jinja2
[templates] loaded channel_format.jinja2
[templates] loaded reference_appendix.jinja2
[templates] loaded tokenizer_profile.jinja2
[templates] loaded message_metadata.md
[templates] loaded jinja2_template.md
[templates] loaded offline_reasoning_agent.md
    ✓ Router created with 9 templates

============================================================
ALL TESTS PASSED ✓
============================================================
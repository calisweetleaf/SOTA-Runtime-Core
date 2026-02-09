"""
Verification Script for Prompt Template Integration
Validates that offline_reasoning_agent.md is correctly injected into the system prompt.
Outputs the full rendered prompt for review and saves to file.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_template_library():
    """Initialize and return the TemplateLibrary with error handling."""
    try:
        from neural_router import RouterConfig, TemplateLibrary
        config = RouterConfig()
        lib = TemplateLibrary(config)
        return lib
    except ImportError as e:
        logger.error(f"Failed to import neural_router: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize TemplateLibrary: {e}")
        raise


def render_system_prompt(lib) -> Tuple[Optional[str], dict]:
    """
    Render the system_prompt template and return the result with metadata.
    
    Returns:
        Tuple of (rendered_prompt, metadata_dict)
    """
    metadata = {
        'template_key': 'system_prompt',
        'render_time': datetime.now().isoformat(),
        'success': False,
        'char_count': 0,
        'line_count': 0,
        'error': None
    }
    
    try:
        rendered = lib._render_jinja_template('system_prompt', {})
        if rendered is None:
            metadata['error'] = "Template render returned None"
            return None, metadata
        
        metadata['success'] = True
        metadata['char_count'] = len(rendered)
        metadata['line_count'] = rendered.count('\n') + 1
        return rendered, metadata
        
    except Exception as e:
        metadata['error'] = str(e)
        logger.error(f"Failed to render template: {e}")
        return None, metadata


def validate_personality_injection(rendered: str) -> Tuple[bool, list]:
    """
    Validate that the offline personality content is present in the rendered prompt.
    
    Returns:
        Tuple of (all_passed, results_list)
    """
    expected_phrases = [
        "OFFLINE REASONING AGENT",
        "You are an advanced AI assistant, an offline reasoning agent.",
        "Extended Thinking",
        "Constitutional AI",
    ]
    
    results = []
    all_passed = True
    
    for phrase in expected_phrases:
        found = phrase in rendered
        results.append({
            'phrase': phrase,
            'found': found
        })
        if not found:
            all_passed = False
            
    return all_passed, results


def save_prompt_to_file(rendered: str, output_dir: Path) -> Path:
    """
    Save the rendered prompt to a timestamped file for review.
    
    Returns:
        Path to the saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"generated_system_prompt_{timestamp}.txt"
    
    try:
        output_file.write_text(rendered, encoding='utf-8')
        logger.info(f"Prompt saved to: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Failed to save prompt to file: {e}")
        raise


def print_prompt_with_delimiters(rendered: str, max_preview_lines: int = 50):
    """Print the rendered prompt with clear visual delimiters."""
    lines = rendered.split('\n')
    total_lines = len(lines)
    
    print("\n" + "=" * 80)
    print("  GENERATED SYSTEM PROMPT - FULL OUTPUT")
    print("=" * 80 + "\n")
    
    print(rendered)
    
    print("\n" + "=" * 80)
    print(f"  END OF PROMPT ({len(rendered):,} chars, {total_lines} lines)")
    print("=" * 80 + "\n")


def verify_integration(output_prompt: bool = True, save_to_file: bool = True) -> bool:
    """
    Main verification function.
    
    Args:
        output_prompt: If True, print the full rendered prompt
        save_to_file: If True, save prompt to output directory
        
    Returns:
        True if all validations passed
    """
    print("\n" + "-" * 60)
    print("  Prompt Template Integration Verification")
    print("-" * 60 + "\n")
    
    # Step 1: Initialize
    print("[1/4] Initializing TemplateLibrary...")
    lib = initialize_template_library()
    print("      ✓ TemplateLibrary initialized\n")
    
    # Step 2: Render
    print("[2/4] Rendering system_prompt template...")
    rendered, metadata = render_system_prompt(lib)
    
    if rendered is None:
        print(f"      ✗ FAILED: {metadata['error']}")
        return False
        
    print(f"      ✓ Rendered successfully")
    print(f"        - Characters: {metadata['char_count']:,}")
    print(f"        - Lines: {metadata['line_count']}")
    print(f"        - Timestamp: {metadata['render_time']}\n")
    
    # Step 3: Validate
    print("[3/4] Validating personality injection...")
    all_passed, results = validate_personality_injection(rendered)
    
    for result in results:
        status = "✓" if result['found'] else "✗"
        print(f"      [{status}] '{result['phrase'][:50]}...'")
    
    if all_passed:
        print("\n      ✓ All validation checks passed\n")
    else:
        print("\n      ✗ Some validation checks failed\n")
    
    # Step 4: Output
    print("[4/4] Output options...")
    
    if save_to_file:
        project_root = Path(__file__).parent
        output_dir = project_root / "output"
        saved_path = save_prompt_to_file(rendered, output_dir)
        print(f"      ✓ Saved to: {saved_path}\n")
    
    if output_prompt:
        print_prompt_with_delimiters(rendered)
    
    # Summary
    print("-" * 60)
    if all_passed:
        print("  RESULT: SUCCESS - Offline personality integrated correctly")
    else:
        print("  RESULT: FAILURE - Personality injection incomplete")
    print("-" * 60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify prompt template integration and output generated prompt"
    )
    parser.add_argument(
        '--no-output', 
        action='store_true',
        help="Don't print the full prompt to console"
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help="Don't save the prompt to a file"
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Minimal output (validation only)"
    )
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    try:
        success = verify_integration(
            output_prompt=not args.no_output,
            save_to_file=not args.no_save
        )
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Verification failed with error: {e}")
        sys.exit(1)

# Python Production Doctor - Quick Start Guide

## Setup Requirements

**Zero Dependencies Required!**

This tool uses only Python standard library modules. No pip install needed!

## Configuration Files

### 1. Main Config: `DOCTOR_CONFIG.json`

Create this file in your project root:

```json
{
  "min_function_lines": 5,
  "min_docstring_length": 15,
  "test_coverage_threshold": 0.7,
  "ignore_patterns": [
    "__pycache__/*",
    "*.pyc",
    ".git/*",
    ".venv/*",
    "venv/*",
    "env/*",
    "node_modules/*",
    "build/*",
    "dist/*"
  ],
  "ignore_functions": [
    "__init__",
    "__str__",
    "__repr__"
  ],
  "severity_levels": {
    "syntax_errors": "critical",
    "unimplemented_abstracts": "critical",
    "stubs": "serious",
    "simple_returns": "serious",
    "incomplete_methods": "serious",
    "test_gaps": "serious",
    "todos": "minor",
    "missing_docstrings": "minor",
    "suspicious_short_functions": "minor",
    "type_hint_gaps": "minor"
  }
}
```

### 2. Optional API Keys: `openrouter_keys.txt`

Only needed if you plan to extend with AI features in the future:

```
# One API key per line
sk-or-xxxxxxxxx
sk-or-yyyyyyyyy
```

## Basic Usage

### Scan your project

```bash
python python_production_doctor.py /path/to/your/project
```

### With custom config

```bash
python python_production_doctor.py /path/to/project -c DOCTOR_CONFIG.json
```

### Generate JSON for CI/CD

```bash
python python_production_doctor.py /path/to/project -f json -o report.json
```

### Control parallel workers (default: 4)

```bash
python python_production_doctor.py /path/to/project -j 8
```

## Command Reference

```bash
python python_production_doctor.py [PROJECT_PATH] [OPTIONS]

Options:
  -c, --config CONFIG    Config file path (JSON)
  -o, --output OUTPUT    Output file (default: production_report.md)
  -f, --format FORMAT    Output format: markdown or json (default: markdown)
  -j, --jobs JOBS        Number of parallel workers (default: 4)
  -v, --verbose          Enable verbose logging
  -h, --help            Show help
```

## Understanding the Report

The tool generates a markdown report with:

1. **Summary Section**
   - Total files scanned
   - Issues by severity (Critical, Serious, Minor)
   - Production readiness status

2. **Code Quality Metrics**
   - Documentation coverage %
   - Type hint coverage %

3. **File-by-File Analysis**
   - Each file with issues
   - Issues categorized by type
   - Line numbers and descriptions

4. **Action Plan**
   - Priority recommendations
   - What to fix before deployment

## What Gets Analyzed

| Check | Description | Severity |
|-------|-------------|----------|
| Syntax Errors | Python syntax violations | Critical |
| TODOs/FIXMEs | Technical debt markers | Minor |
| Stub Implementations | `pass`, `...`, `NotImplementedError` | Serious |
| Placeholder Returns | `return None`, `return 0`, etc. | Serious |
| Incomplete Methods | Methods with only docstrings + stub | Serious |
| Missing Docstrings | Functions/classes without docs | Minor |
| Short Functions | Functions below minimum line count | Minor |
| Abstract Methods | Unimplemented abstract methods | Critical |
| Type Hints | Missing/incomplete type annotations | Minor |
| Test Gaps | Missing test files | Serious |

## Exit Codes

- `0`: Success (no critical issues)
- `1`: Critical issues found

## Tips

1. **Run before each commit:**

   ```bash
   python python_production_doctor.py . && git commit -m "your message"
   ```

2. **Add to CI/CD pipeline:**

   ```yaml
   - name: Code Health Check
     run: python python_production_doctor.py . -f json
   ```

3. **Customize thresholds:**
   Edit `DOCTOR_CONFIG.json` to adjust:
   - `min_function_lines` - adjust short function detection
   - `min_docstring_length` - adjust docstring requirements
   - `test_coverage_threshold` - adjust test expectations

4. **Ignore specific functions:**
   Add function names to `ignore_functions` in config

## Troubleshooting

**Unicode errors on Windows?**

- Run in PowerShell or VS Code terminal
- Or use Windows Terminal with UTF-8 support

**Scanning too many files?**

- Add more patterns to `ignore_patterns` in config

**False positives?**

- Increase `min_function_lines` threshold
- Add specific functions to `ignore_functions`

**Slow on large projects?**

- Increase parallel workers: `-j 8` or higher
- Add more patterns to `ignore_patterns`

## Files Generated

- `production_report.md` (default) - Human-readable report
- `production_doctor.log` - Debug log (if verbose)

## Example Output Structure

```
# Python Production Doctor Report

**Project Root:** `/path/to/project`
**Scan Date:** 2026-01-08 22:45:00
**Python Version:** 3.12.0

---

## Summary

🟠 **SERIOUS ISSUES** - Requires significant work

**Files Scanned:** 15 | **Total Issues:** 23
- 🔴 Critical: 2
- 🟠 Serious: 12
- 🟡 Minor: 9

## Code Quality Metrics

- **Documentation Coverage:** 67.5% (27/40)
- **Type Hint Coverage:** 45.0% (18/40)

## File Analysis

### 📄 `main.py`

**Total Issues:** 5

#### 🚧 Stub Implementations
1. `process_data()` at line 42 - **pass statement**

#### ⚠️ Placeholder Returns
1. `get_config()` at line 88 - returns **None**

... (more details)
```

## Next Steps

1. **Review Critical Issues First**
   - Fix syntax errors
   - Implement abstract methods

2. **Address Serious Issues**
   - Replace stubs with real code
   - Add missing tests
   - Remove placeholder returns

3. **Plan for Minor Issues**
   - Add docstrings
   - Complete type hints
   - Address technical debt

## Integration Ideas

- **Pre-commit hook:** Prevent commits with critical issues
- **CI/CD gate:** Block deployment on critical issues
- **Weekly reports:** Track code health over time
- **Team dashboard:** Display metrics for all projects

# Running CI/CD Workflows Locally

This guide shows you how to test GitHub Actions workflows locally before pushing to GitHub.

## Prerequisites

Install `act` (already done if you're reading this):
```bash
brew install act
```

## Quick Start

### List all available workflows
```bash
act -l
```

### Run all jobs
```bash
act
```

### Run specific job
```bash
act -j lint          # Run linting only
act -j test          # Run tests only
act -j validate-manifest  # Validate manifest only
act -j docs          # Check documentation
```

### Run specific event
```bash
act push            # Simulate a push event
act pull_request    # Simulate a PR
act workflow_dispatch  # Manual trigger
```

## Common Use Cases

### 1. Quick Lint Check Before Committing
```bash
act -j lint
```

### 2. Run Tests on Specific Python Version
```bash
act -j test --matrix python-version:3.9
```

### 3. Validate Everything (like CI does)
```bash
act push
```

### 4. Dry Run (see what would execute)
```bash
act -n
```

### 5. Run with Specific Platform
```bash
act -j test --matrix os:ubuntu-latest
act -j test --matrix os:macos-latest
```

## Useful Flags

- `-n, --dryrun` - Don't actually run, just show what would run
- `-l, --list` - List all workflows/jobs
- `-j, --job` - Run a specific job
- `-v, --verbose` - Verbose output
- `--container-architecture` - Set architecture (e.g., linux/arm64)
- `-s, --secret` - Pass secrets (e.g., `-s GITHUB_TOKEN=...`)
- `--env` - Set environment variables
- `-P, --platform` - Use specific platform

## Examples for This Repository

### Before Committing Code
```bash
# Run linting and basic checks
act -j lint -j validate-manifest

# If you have Docker, run full test suite
act -j test
```

### Test Manifest Changes
```bash
act -j validate-manifest
```

### Test Documentation
```bash
act -j docs
```

### Full CI Pipeline
```bash
act push --verbose
```

## Limitations

1. **Docker Required**: Most jobs need Docker to run (act uses Docker containers)
2. **Platform Differences**: macOS jobs won't run exactly like on GitHub (uses Linux containers)
3. **Secrets**: You need to pass secrets manually with `-s` flag
4. **Performance**: May be slower locally than on GitHub's infrastructure

## Faster Alternative: Run Jobs Directly

If you don't want to use Docker, you can run the commands directly:

### Lint
```bash
pip install black flake8 mypy
black --check datasets/scripts/ manifest/ tests/
flake8 datasets/scripts/ manifest/ tests/ --max-line-length=100
mypy datasets/scripts/ manifest/ --ignore-missing-imports
```

### Test
```bash
pip install pytest pytest-cov Pillow numpy
pytest tests/ -v
```

### Validate Manifest
```bash
pip install jsonschema
cd manifest
python -c "
import json
import jsonschema

with open('styles_schema.json') as f:
    schema = json.load(f)
with open('styles.json') as f:
    manifest = json.load(f)
    
jsonschema.validate(manifest, schema)
print('✓ Manifest is valid')
"
```

## Troubleshooting

### "Cannot connect to Docker daemon"
- Make sure Docker Desktop is running
- Or use the direct command approach above

### "Job failed" but no error shown
- Add `-v` flag for verbose output: `act -j <job-name> -v`

### Platform errors
- Use `--container-architecture linux/amd64` for compatibility

### Too slow
- Use `-j` to run specific jobs instead of everything
- Or run commands directly without Docker

## Resources

- [act GitHub Repository](https://github.com/nektos/act)
- [act Documentation](https://nektosact.com)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

**Pro Tip**: Create an alias in your `~/.zshrc`:
```bash
alias ci-lint="act -j lint"
alias ci-test="act -j test"
alias ci-check="act -j lint -j validate-manifest -j docs"
```

Then just run `ci-check` before committing! 🚀

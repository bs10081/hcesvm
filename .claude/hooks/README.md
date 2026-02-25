# HCESVM Hooks

This directory contains event-driven automation hooks for the HCESVM project.

## Configured Hooks

### PreToolUse Hooks

#### 1. Documentation File Control
- **Trigger**: Before Write tool execution
- **Purpose**: Enforce documentation consolidation
- **Behavior**: 
  - ✅ Allow: README.md, CLAUDE.md, AGENTS.md, CONTRIBUTING.md
  - ✅ Allow: `.claude/*/*.md`, `docs/*/*.md`
  - ❌ Block: Other .md/.txt files
- **Rationale**: Keep documentation centralized (everything-claude-code principle)

### PostToolUse Hooks

#### 2. Auto-Format with Black
- **Trigger**: After Edit/Write on .py files
- **Purpose**: Automatic code formatting
- **Behavior**:
  - Runs `black --quiet <file>` automatically
  - Skips if Black not installed
  - Formats to 88 character line length (PEP 8)
- **Install Black**: `pip install black`

#### 3. Flake8 Linting
- **Trigger**: After Edit/Write on .py source files (excludes tests/)
- **Purpose**: Code quality checks
- **Behavior**:
  - Runs `flake8 --max-line-length=88 <file>`
  - Shows warnings but doesn't block
  - Ignores E203, W503 (Black compatibility)
- **Install Flake8**: `pip install flake8`

### Stop Hooks

#### 4. Console.log Detection
- **Trigger**: Before response ends
- **Purpose**: Warn about debug statements
- **Behavior**:
  - Checks staged files for console.log/print statements
  - Warning only (doesn't block)
  - Reminder to clean up before commit

## Hook Management

### Enable/Disable Hooks

To disable a specific hook, edit `hooks.json` and remove or comment out the hook entry.

### Debugging Hooks

If hooks are not working:
```bash
# Check Claude Code version (requires v2.1.0+)
claude --version

# Verify hooks.json syntax
cat .claude/hooks/hooks.json | jq .

# Check hook execution in Claude Code logs
tail -f ~/.claude/logs/*.log
```

### Performance Considerations

- **Black**: Fast (~10ms per file)
- **Flake8**: Fast (~50ms per file)
- **Console.log check**: Fast (~10ms)

Total overhead: < 100ms per file edit

## Best Practices

1. **Don't Skip Hooks** - They enforce code quality
2. **Install Tools** - Black and Flake8 for full benefits
3. **Review Warnings** - Address linter warnings promptly
4. **Keep Hooks Simple** - Complex logic should be in scripts

## Troubleshooting

### Black Not Found
```bash
# Install Black in virtual environment
source .venv/bin/activate
pip install black
```

### Flake8 Not Found
```bash
# Install Flake8
source .venv/bin/activate
pip install flake8
```

### Hooks Not Triggering
- Ensure `.claude/hooks/hooks.json` exists
- Check Claude Code version ≥ v2.1.0
- Verify JSON syntax (no trailing commas)
- Restart Claude Code session

## References

- [Claude Code Hooks Documentation](https://code.claude.com/docs/hooks)
- [Black Formatter](https://black.readthedocs.io/)
- [Flake8 Linter](https://flake8.pycqa.org/)
- [everything-claude-code Hooks](https://github.com/affaan-m/everything-claude-code)

---

**Version**: 1.0  
**Last Updated**: 2026-02-25

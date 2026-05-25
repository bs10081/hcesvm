# Git Workflow Rules for HCESVM

## 📝 Commit Message Format

```
<type>: <description>

<optional body>

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

### Types:
- **feat**: New feature
- **fix**: Bug fix
- **refactor**: Code restructuring
- **docs**: Documentation
- **test**: Adding tests
- **chore**: Maintenance

### Example:
```
feat: add test3 strategy with balanced class weighting

Implements sample-weighted objective function.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

## 🚫 Git Safety Protocol

### NEVER:
- ❌ `git push --force` to main/master
- ❌ `git reset --hard` without backup
- ❌ Skip hooks (`--no-verify`)

### ALWAYS:
- ✅ Review changes (`git status`, `git diff`)
- ✅ Stage specific files
- ✅ Write descriptive messages
- ✅ Create new commits (don't amend after push)

## 🔄 Daily Workflow

```bash
# 1. Check status
git status

# 2. Review changes
git diff

# 3. Stage specific files
git add README.md docs/strategies/README.md

# 4. Commit
git commit -S -m "docs: refresh strategy guidance"

# 5. Push
git push
```

## 📦 What to Commit

### ✅ Always:
- Source code (`src/`)
- Tests (`tests/`)
- Documentation
- Configuration

### ❌ Never:
- `.venv/`, `__pycache__/`
- `.env`, `*.lic`
- Large datasets
- Temporary logs

## 🔍 Pre-Commit Checks

```bash
# Run tests
source .venv/bin/activate && pytest tests/ -v

# Check for secrets
grep -r "password" src/

# Verify .gitignore
git status
```

---

**Version**: 1.0  
**Last Updated**: 2026-02-25

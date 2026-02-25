# Security Rules for HCESVM

## 🔒 Mandatory Security Checks

### Before ANY Commit
- [ ] No hardcoded secrets (API keys, Gurobi license paths, credentials)
- [ ] All user inputs validated (especially in data_loader.py)
- [ ] No sensitive paths in version control
- [ ] Error messages don't leak sensitive information

## 🛡️ Secure Coding Practices

### 1. No Hardcoded Credentials
```python
# ❌ BAD
GUROBI_LICENSE = "/home/user/gurobi.lic"

# ✅ GOOD
import os
GUROBI_LICENSE = os.getenv("GUROBI_LICENSE_PATH")
```

### 2. Input Validation
```python
# ✅ Always validate file paths
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Invalid path: {file_path}")
```

### 3. Safe Data Handling
```python
# ✅ Validate array shapes
def fit(self, X, y):
    if X.shape[0] != len(y):
        raise ValueError("X and y dimensions mismatch")
```

## 🚫 Prohibited Patterns

Never hardcode file paths, expose internal errors, or store credentials in code.

## ✅ Security Checklist

- [ ] Validate file paths exist
- [ ] Check file permissions
- [ ] Validate data format
- [ ] Handle missing values explicitly

---

**Version**: 1.0  
**Last Updated**: 2026-02-25

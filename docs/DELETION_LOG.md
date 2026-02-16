# Code Deletion Log

## Analysis Date: 2026-02-14

This document tracks dead code, unused imports, and consolidation opportunities identified in the Polymarket codebase.

---

## Dead Code Identified

### 1. Unused Alternate Implementation Files

#### `scanner/fees_v2.py` (203 lines)
- **Status**: UNUSED - No imports found
- **Location**: `/home/fiod/Polymarket/scanner/fees_v2.py`
- **Replacement**: `scanner/fees.py` is actively used by all scanners
- **Reason**: Appears to be an alternate fee model implementation that was never integrated
- **Key Differences**:
  - Updated DCM parabolic fee calculation
  - Slightly different fee logic in `adjust_profit()`
  - Has a `__main__` block with example calculations
- **Risk Level**: SAFE - No references anywhere in codebase
- **Recommendation**: DELETE - Keep fees.py as the single source of truth

#### `executor/cross_platform_v2.py` (577 lines)
- **Status**: UNUSED - No imports found
- **Location**: `/home/fiod/Polymarket/executor/cross_platform_v2.py`
- **Replacement**: `executor/cross_platform.py` (435 lines) is actively used
- **Reason**: Appears to be a newer implementation with state machine improvements
- **Risk Level**: CAREFUL - Larger file with more features, may have valuable improvements
- **Recommendation**: REVIEW - Check if v2 has improvements worth backporting to v1 before deletion
- **Used by**:
  - `executor/engine.py` imports from `cross_platform.py` (not v2)
  - `run.py` imports `CrossPlatformUnwindFailed` from `cross_platform.py`

### 2. Partially Implemented/Disabled Scanners

#### `executor/maker_lifecycle.py` (entire file)
- **Status**: IMPLEMENTED but UNUSED
- **Location**: `/home/fiod/Polymarket/executor/maker_lifecycle.py`
- **Function**: GTC maker order lifecycle manager
- **Reason**: Maker orders require different execution model than current FAK-only approach
- **Used by**: NONE (grep found only class definition, no imports)
- **Risk Level**: SAFE - Complete module with no dependencies
- **Recommendation**: DELETE or move to `/archive` if maker strategy may be revived

### 3. File Management Issues

#### `__pycache__` directories
- **Status**: IGNORED by .gitignore but present in filesystem
- **Location**: `/home/fiod/Polymarket/scanner/__pycache__/*`
- **Files Found**: 19+ .pyc files
- **Risk Level**: SAFE - Should not be in git
- **Recommendation**: Verify not tracked, run `git clean -fdX` to remove

---

## Configuration Fields Analysis

### Unused Config Fields (Potentially)

From `/home/fiod/Polymarket/config.py`:

#### Scanners with Enable Flags

1. **`value_scanner_enabled`** (line 117)
   - Default: `False`
   - Status: Scanner exists (`scanner/value.py`), imported in `run.py`
   - Used: Conditionally in run.py
   - Risk: Safe to keep (documented as disabled by design)

2. **`resolution_sniping_enabled`** (line 132)
   - Default: `True`
   - Status: Scanner exists (`scanner/resolution.py`), imported in `run.py`
   - Used: Conditionally in run.py
   - Risk: Safe to keep (active feature)

3. **`stale_quote_enabled`** (line 122)
   - Default: `True`
   - Status: Scanner exists (`scanner/stale_quote.py`), imported in `run.py`
   - Used: Conditionally in run.py via `StaleQuoteDetector`
   - Risk: Safe to keep (active feature)

4. **`fanatics_enabled`** (line 141)
   - Default: `False`
   - Status: Client stub exists (`client/fanatics.py`) with NotImplementedError
   - Used: In run.py for platform initialization
   - Risk: Safe to keep (future integration)
   - Note: All Fanatics API methods raise NotImplementedError with "_NOT_AVAILABLE" message

#### Fanatics-Specific Config (All Unused)

- `fanatics_api_key` (line 137) - Default: `""`
- `fanatics_api_secret` (line 138) - Default: `""`
- `fanatics_host` (line 139) - Default: `""`
- `fanatics_position_limit` (line 140) - Default: `25000.0`

**Recommendation**: Keep for future use, but document that Fanatics integration is pending API availability.

---

## Code Quality Issues

### 1. Type Ignore Annotations

Found 4 instances of `# type: ignore`:

1. **`tests/test_kalshi_cache.py:87`**
   ```python
   snap.version = 99  # type: ignore[misc]
   ```
   - Reason: Mutating frozen dataclass in test
   - Fix: Use `replace(snap, version=99)` instead

2. **`tests/test_fanatics.py:54`**
   ```python
   m.ticker = "T2"  # type: ignore[misc]
   ```
   - Reason: Mutating frozen dataclass in test
   - Fix: Create new instance instead of mutation

3. **`executor/safety.py:445`**
   ```python
   ext_books["kalshi"] = platform_books  # type: ignore[assignment]
   ```
   - Reason: Type mismatch in dict assignment
   - Fix: Add proper type annotation

4. **`executor/safety.py:447`**
   ```python
   ext_books = platform_books  # type: ignore[assignment]
   ```
   - Reason: Type mismatch
   - Fix: Add proper type annotation

**Recommendation**: Fix these type issues for better type safety.

### 2. Comments and Documentation

#### Single Note Comment
- **`client/fanatics_auth.py:8`**
  ```
  NOTE: Fanatics event contract API endpoints are TBD.
  ```
  - Status: Informational, not actionable
  - Keep: Yes (documents incomplete integration)

#### No TODO/FIXME/HACK/XXX found
- Excellent code hygiene

---

## Duplication Analysis

### No Significant Duplicates Found

Validation functions are properly separated:
- **`scanner/validation.py`**: Input validation (price, size, gas bounds)
  - `validate_price()` - NaN/Inf/range checks
  - `validate_size()` - NaN/Inf/negative checks
  - `validate_gas_gwei()` - Gas price validation

- **`executor/safety.py`**: Business logic verification (staleness, depth, edge)
  - `verify_prices_fresh()` - Staleness checks
  - `verify_depth()` - Orderbook depth validation
  - `verify_edge_intact()` - Profit edge revalidation
  - 9+ verification functions

These serve different purposes and are not duplicates.

---

## File Size Analysis

### Large Files (>500 lines)

1. **`executor/cross_platform_v2.py`**: 577 lines (UNUSED)
2. **`executor/safety.py`**: Likely >500 lines (9+ functions)
3. **`run.py`**: Likely >500 lines (main pipeline)

**Recommendation**: After deleting unused files, consider splitting remaining large files by feature.

---

## Dependencies (pyproject.toml)

### Current Dependencies
```toml
dependencies = [
    "py-clob-client>=0.18.0",
    "httpx>=0.27.0",
    "websockets>=13.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "python-dotenv>=1.0",
    "eth-account>=0.13.0",
    "rapidfuzz>=3.0",
    "cryptography>=42.0",
]
```

### Analysis
- All dependencies appear used:
  - `py-clob-client`: Polymarket CLOB client (client/clob.py)
  - `httpx`: HTTP client (gamma.py, kalshi.py, gas.py)
  - `websockets`: WebSocket feeds (client/ws.py, ws_bridge.py)
  - `pydantic`: Config + models (config.py, scanner/models.py)
  - `pydantic-settings`: Config loading (config.py)
  - `python-dotenv`: .env loading (via pydantic-settings)
  - `eth-account`: Wallet auth (client/auth.py)
  - `rapidfuzz`: Fuzzy matching (scanner/matching.py)
  - `cryptography`: RSA signing for Kalshi (client/kalshi_auth.py)

**Recommendation**: No unused dependencies detected. All are actively used.

### Dev Dependencies
```toml
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0",
    "respx>=0.22.0",
]
```

All dev dependencies are standard testing tools. No cleanup needed.

---

## Summary of Findings

### High-Priority Deletions (Safe)
1. `scanner/fees_v2.py` - 203 lines - Unused alternate implementation
2. `executor/maker_lifecycle.py` - Unused GTC lifecycle manager

### Medium-Priority Review
1. `executor/cross_platform_v2.py` - 577 lines - May contain valuable improvements
   - Action: Review differences, backport improvements, then delete

### Low-Priority Cleanup
1. Fix 4 `# type: ignore` annotations for better type safety
2. Verify `__pycache__` not tracked in git

### Keep (Intentionally Disabled/Future Use)
1. `scanner/value.py` - Disabled by design (documented in CLAUDE.md)
2. `scanner/resolution.py` - Active feature
3. `scanner/stale_quote.py` - Active feature
4. `client/fanatics.py` + config - Future integration (API pending)
5. `client/kalshi_cache.py` - Newly added (in git status as untracked)

### No Action Needed
1. No unused dependencies detected
2. No TODO/FIXME/HACK comments found
3. No significant code duplication
4. Validation vs safety functions are properly separated

---

## Proposed Deletion Plan

### Phase 1: Safe Deletions (Low Risk)
- [ ] Delete `scanner/fees_v2.py`
- [ ] Delete `executor/maker_lifecycle.py`
- [ ] Run full test suite
- [ ] Commit: "refactor: remove unused alternate implementations"

### Phase 2: Review and Backport
- [ ] Compare `executor/cross_platform_v2.py` vs `cross_platform.py`
- [ ] Identify valuable improvements in v2
- [ ] Backport improvements to v1 if needed
- [ ] Delete `cross_platform_v2.py`
- [ ] Run full test suite
- [ ] Commit: "refactor: consolidate cross-platform execution"

### Phase 3: Type Safety Improvements
- [ ] Fix frozen dataclass mutations in tests
- [ ] Fix type ignore annotations in safety.py
- [ ] Run mypy/pyright if available
- [ ] Commit: "fix: improve type safety annotations"

### Phase 4: Documentation
- [ ] Update CLAUDE.md if needed
- [ ] Document Fanatics integration status
- [ ] Mark this deletion log as complete

---

## Impact Estimation

### Immediate Cleanup (Phase 1)
- **Files deleted**: 2
- **Lines of code removed**: ~400 lines
- **Build impact**: None (files not imported)
- **Test impact**: None (no tests reference these files)
- **Bundle size reduction**: Minimal (~10KB source)

### Post-Review Cleanup (Phase 2)
- **Files deleted**: 1
- **Lines of code removed**: ~140-580 lines (depending on backport)
- **Build impact**: None if properly reviewed
- **Test impact**: None if cross_platform.py maintains compatibility
- **Bundle size reduction**: ~15KB source

### Total Potential Reduction
- **Files**: 3
- **Lines**: 540-980 lines
- **Complexity**: Reduced maintenance surface

---

## Testing Checklist

Before ANY deletion:
- [ ] Grep for all imports of target file
- [ ] Check if file is in `__init__.py` exports
- [ ] Search for dynamic imports (string patterns)
- [ ] Review git history for context
- [ ] Create backup branch

After each deletion:
- [ ] Build succeeds: `uv sync --all-extras`
- [ ] All tests pass: `PYTHONPATH=. uv run pytest tests/ -v`
- [ ] No import errors in dry-run: `uv run python run.py --dry-run --limit 10`
- [ ] Create atomic git commit
- [ ] Update this log

---

## Notes

1. **kalshi_cache.py** appears in git status as untracked - this is NEW code, not dead code
2. **Fanatics integration** is intentionally stubbed - keep for future use
3. **Value scanner** is intentionally disabled - documented as producing false positives
4. **No commented-out code blocks** found - excellent code hygiene
5. **All config fields** are either used or documented as future/disabled features

---

## Risk Assessment

### Overall Risk: LOW

- Most identified dead code has zero references
- No dependencies on deleted code found
- All active scanners properly integrated in run.py
- Test coverage should catch any missed dependencies
- Backup branches provide easy rollback

### Safety Measures
1. Work on feature branch: `git checkout -b refactor/dead-code-cleanup`
2. Atomic commits per deletion category
3. Run full test suite after each phase
4. Manual dry-run smoke test
5. Peer review before merge

---

*Analysis performed with Claude Code refactor-cleaner agent*
*Repository: /home/fiod/Polymarket*
*Analysis Date: 2026-02-14*

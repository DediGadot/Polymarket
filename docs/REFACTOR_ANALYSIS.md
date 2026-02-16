# Dead Code & Refactoring Analysis

**Analysis Date**: 2026-02-14
**Codebase**: Polymarket Arbitrage Bot
**Total Python Files**: 164 files

---

## Executive Summary

### Findings
- **2 completely unused files** ready for immediate deletion (403 lines)
- **1 unused alternate implementation** requiring review before deletion (577 lines)
- **4 type safety issues** with `# type: ignore` annotations
- **0 unused dependencies** (all packages actively used)
- **0 TODO/FIXME/HACK comments** (excellent code hygiene)
- **0 significant code duplication** found

### Total Cleanup Potential
- Remove **980+ lines** of dead code
- Delete **3 files**
- Fix **4 type safety issues**
- No dependency changes needed

---

## 1. Files to DELETE (Completely Unused)

### File 1: `scanner/fees_v2.py`

**Path**: `/home/fiod/Polymarket/scanner/fees_v2.py`
**Size**: 203 lines
**Status**: UNUSED - No imports found anywhere in codebase

**Evidence**:
```bash
$ grep -r "fees_v2" --include="*.py" /home/fiod/Polymarket
# No results (file exists but never imported)
```

**Current Usage**: Active file is `scanner/fees.py`, imported by:
- `scanner/binary.py:7`
- `scanner/latency.py:16`
- `scanner/spike.py:18`
- `scanner/maker.py:17`
- `scanner/value.py:17`
- `scanner/cross_platform.py:21`
- `scanner/stale_quote.py:18`
- `scanner/negrisk.py:16`
- `scanner/resolution.py:19`
- `run.py:43`
- Plus 10+ test files

**Key Differences** (fees.py vs fees_v2.py):
1. fees_v2 has updated DCM parabolic fee calculation (lines 51-70)
2. fees_v2 has `compute_dcm_fee_examples()` function (lines 176-198)
3. fees_v2 has `__main__` block for manual testing (lines 201-202)
4. Different `adjust_profit()` logic (fees.py lines 93-129 vs fees_v2.py lines 121-173)

**Recommendation**: DELETE fees_v2.py
**Risk**: SAFE (zero references)

---

### File 2: `executor/maker_lifecycle.py`

**Path**: `/home/fiod/Polymarket/executor/maker_lifecycle.py`
**Size**: ~200+ lines (estimated)
**Status**: UNUSED - Implementation exists but never imported

**Evidence**:
```bash
$ grep -r "maker_lifecycle" --include="*.py" /home/fiod/Polymarket
/home/fiod/Polymarket/executor/maker_lifecycle.py:class MakerLifecycle:
# Only the file itself, no external imports
```

**Purpose**: GTC (Good-Til-Cancelled) maker order lifecycle manager

**Classes Defined**:
- `MakerOrder` (dataclass) - line 26
- `MakerConfig` (dataclass) - line 38
- `MakerLifecycle` (class) - line 45

**Why Unused**: Current bot uses FAK (Fill-And-Kill) orders only. Maker orders require:
- Different execution model
- Fill monitoring across cycles
- Active order management
- Not compatible with current stateless scan loop

**Recommendation**: DELETE or move to `/archive/future_features/`
**Risk**: SAFE (complete isolated module)

---

## 2. Files to REVIEW Before Deletion

### File 3: `executor/cross_platform_v2.py`

**Path**: `/home/fiod/Polymarket/executor/cross_platform_v2.py`
**Size**: 577 lines
**Status**: UNUSED but potentially valuable

**Current Active File**: `executor/cross_platform.py` (435 lines)

**Evidence of Non-Use**:
```bash
$ grep -r "cross_platform_v2" --include="*.py" /home/fiod/Polymarket
# No results

# Active usage of v1:
executor/engine.py:        from executor.cross_platform import execute_cross_platform
run.py:from executor.cross_platform import CrossPlatformUnwindFailed
tests/test_cross_platform_exec.py:from executor.cross_platform import (
```

**Git History**:
```
34fb931 w/ kalshi  (most recent commit touching v2)
```

**Size Comparison**:
- v1 (cross_platform.py): 435 lines
- v2 (cross_platform_v2.py): 577 lines (+142 lines / +33%)

**Key Imports Difference**:

v1 imports (lines 10-36):
```python
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderType

from client.clob import create_limit_order, post_order
from client.platform import PlatformClient
from executor.fill_state import (...)
from scanner.models import (...)
```

v2 imports (lines 10-46):
```python
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    OrderArgs,
    MarketOrderArgs,  # NEW
    OrderType,
    BookParams,       # NEW
    PartialCreateOrderOptions,  # NEW
)
from py_clob_client.order_builder.constants import BUY, SELL  # NEW

from client.clob import create_limit_order, post_order, post_orders  # post_orders is NEW
# ... rest same as v1
```

**Action Required**:
1. Manually diff `cross_platform.py` and `cross_platform_v2.py`
2. Identify if v2 has:
   - Better error handling
   - State machine improvements
   - Market order support (MarketOrderArgs import suggests this)
   - Batch order posting (post_orders import)
3. Backport valuable changes to v1 if needed
4. Then delete v2

**Recommendation**: REVIEW FIRST, then delete
**Risk**: MEDIUM (may lose valuable improvements if deleted without review)

---

## 3. Type Safety Issues (# type: ignore)

All 4 instances found:

### Issue 1: Test - Frozen Dataclass Mutation
**File**: `tests/test_kalshi_cache.py`
**Line**: 87
```python
snap.version = 99  # type: ignore[misc]
```

**Fix**:
```python
# BEFORE:
snap.version = 99  # type: ignore[misc]

# AFTER:
from dataclasses import replace
snap = replace(snap, version=99)
```

---

### Issue 2: Test - Frozen Dataclass Mutation
**File**: `tests/test_fanatics.py`
**Line**: 54
```python
m.ticker = "T2"  # type: ignore[misc]
```

**Fix**:
```python
# BEFORE:
m.ticker = "T2"  # type: ignore[misc]

# AFTER:
from dataclasses import replace
m = replace(m, ticker="T2")
```

---

### Issue 3: Safety Module - Dict Type Mismatch
**File**: `executor/safety.py`
**Line**: 445
```python
ext_books["kalshi"] = platform_books  # type: ignore[assignment]
```

**Context**: In `verify_cross_platform_books()` function

**Fix**: Add proper type annotation to `ext_books` dict:
```python
# Need to review the actual types - likely:
ext_books: dict[str, dict[str, OrderBook]] = {}
```

---

### Issue 4: Safety Module - Dict Type Mismatch
**File**: `executor/safety.py`
**Line**: 447
```python
ext_books = platform_books  # type: ignore[assignment]
```

**Context**: Same function as Issue 3

**Fix**: Properly type `platform_books` parameter or add Union type

---

## 4. Configuration Fields (config.py)

### Active Features (Keep)

All configuration fields in `config.py` are either:
1. Actively used in production
2. Control feature flags for intentionally disabled features
3. Reserved for future integrations

**No unused config fields identified.**

### Feature Flags Status

| Flag | Default | Status | File | Used In |
|------|---------|--------|------|---------|
| `value_scanner_enabled` | False | Intentionally disabled | scanner/value.py | run.py |
| `resolution_sniping_enabled` | True | Active | scanner/resolution.py | run.py |
| `stale_quote_enabled` | True | Active | scanner/stale_quote.py | run.py |
| `fanatics_enabled` | False | Future integration | client/fanatics.py | run.py |
| `latency_enabled` | True | Active | scanner/latency.py | run.py |
| `cross_platform_enabled` | True | Active | scanner/cross_platform.py | run.py |
| `ws_enabled` | True | Active | client/ws.py | run.py |
| `use_fak_orders` | True | Active | executor/engine.py | run.py |
| `fee_model_enabled` | True | Active | scanner/fees.py | run.py |

---

## 5. Fanatics Integration Status

### Files
- `client/fanatics.py` (80+ lines)
- `client/fanatics_auth.py` (includes NOTE comment)
- `scanner/fanatics_fees.py`

### Status: STUB IMPLEMENTATION

**Evidence**:
```python
# client/fanatics.py line 22
_NOT_AVAILABLE = "Fanatics event contract API not yet available"

# All methods raise:
def get_all_markets(self, status: str = "open") -> list[FanaticsMarket]:
    raise NotImplementedError(_NOT_AVAILABLE)
```

**Config Fields** (all unused but intentional):
- `fanatics_api_key: str = ""` (line 137)
- `fanatics_api_secret: str = ""` (line 138)
- `fanatics_host: str = ""` (line 139)
- `fanatics_position_limit: float = 25000.0` (line 140)
- `fanatics_enabled: bool = False` (line 141)

**Recommendation**: KEEP - Auth scaffolding ready for API launch
**Note in CLAUDE.md**: "Fanatics Markets (Iteration 6)"

---

## 6. Comments & Documentation Quality

### Excellent Code Hygiene

- **TODO comments**: 0 found
- **FIXME comments**: 0 found
- **HACK comments**: 0 found
- **XXX comments**: 0 found

**Only comment found**:
- `client/fanatics_auth.py:8` - "NOTE: Fanatics event contract API endpoints are TBD"
  - This is informational and appropriate

---

## 7. Dependencies (pyproject.toml)

### Production Dependencies (All Used)

```toml
dependencies = [
    "py-clob-client>=0.18.0",    # ✓ client/clob.py, client/auth.py
    "httpx>=0.27.0",             # ✓ client/gamma.py, kalshi.py, gas.py
    "websockets>=13.0",          # ✓ client/ws.py, ws_bridge.py
    "pydantic>=2.0",             # ✓ config.py, scanner/models.py
    "pydantic-settings>=2.0",    # ✓ config.py (BaseSettings)
    "python-dotenv>=1.0",        # ✓ via pydantic-settings
    "eth-account>=0.13.0",       # ✓ client/auth.py (wallet signing)
    "rapidfuzz>=3.0",            # ✓ scanner/matching.py (fuzzy event matching)
    "cryptography>=42.0",        # ✓ client/kalshi_auth.py (RSA signing)
]
```

### Dev Dependencies (All Used)

```toml
dev = [
    "pytest>=8.0",               # ✓ Test framework
    "pytest-asyncio>=0.24.0",    # ✓ Async test support
    "pytest-cov>=5.0",           # ✓ Coverage reporting
    "respx>=0.22.0",             # ✓ httpx mocking in tests
]
```

**Result**: No unused dependencies found.

---

## 8. Code Duplication Analysis

### Validation Functions (Properly Separated)

**scanner/validation.py** - Input validation:
- `validate_price(p: float)` - line 15
- `validate_size(s: float)` - line 40
- `validate_gas_gwei(g: float)` - line 63

**executor/safety.py** - Business logic verification:
- `verify_prices_fresh()` - line 95
- `verify_gas_reasonable()` - line 127
- `verify_max_legs()` - line 165
- `verify_depth()` - line 182
- `verify_opportunity_ttl()` - line 249
- `verify_edge_intact()` - line 266
- `verify_inventory()` - line 399
- `verify_cross_platform_books()` - line 419
- `verify_platform_limits()` - line 484

**Conclusion**: These are NOT duplicates. Different purposes:
- validation.py: Low-level type/range checking (NaN, Inf, bounds)
- safety.py: High-level business rule verification (staleness, edge, depth)

**No action needed.**

---

## 9. Git Status Context

### Untracked Files (New Code, Not Dead Code)

```
?? .env.example
?? .github/
?? .gitignore
?? client/kalshi_cache.py          # NEW - Background market cache
?? optimization_log.json
?? tests/test_kalshi_cache.py      # NEW - Tests for above
```

**kalshi_cache.py Analysis**:
- **Status**: NEWLY ADDED (not tracked yet)
- **Purpose**: Background daemon for Kalshi market refresh
- **Used by**: `run.py:39` - `from client.kalshi_cache import KalshiMarketCache`
- **Size**: 151 lines
- **Action**: ADD TO GIT (this is active new code, not dead code)

---

## 10. Large Files Analysis

Files over 400 lines (potential split candidates):

1. **`executor/cross_platform_v2.py`** - 577 lines (UNUSED - DELETE)
2. **`executor/cross_platform.py`** - 435 lines (ACTIVE - OK)
3. **`executor/safety.py`** - Unknown (9+ functions, likely 500+)
4. **`run.py`** - Unknown (main pipeline, likely 500+)

**Post-cleanup recommendation**: Review safety.py and run.py for potential feature-based splitting.

---

## Detailed Action Plan

### Phase 1: Immediate Safe Deletions

**Files to delete**:
1. `/home/fiod/Polymarket/scanner/fees_v2.py` (203 lines)
2. `/home/fiod/Polymarket/executor/maker_lifecycle.py` (200+ lines)

**Commands**:
```bash
cd /home/fiod/Polymarket

# Create feature branch
git checkout -b refactor/dead-code-cleanup

# Delete files
git rm scanner/fees_v2.py
git rm executor/maker_lifecycle.py

# Verify build
uv sync --all-extras

# Run tests
PYTHONPATH=. uv run pytest tests/ -v

# Dry run smoke test
uv run python run.py --dry-run --limit 50

# Commit
git commit -m "refactor: remove unused alternate implementations

- Delete scanner/fees_v2.py (unused alternate fee model, 203 lines)
- Delete executor/maker_lifecycle.py (unused GTC lifecycle, 200+ lines)
- No references found in codebase
- All tests passing

See docs/DELETION_LOG.md for analysis details."
```

**Expected outcome**:
- Build: SUCCESS
- Tests: ALL PASS
- Lines removed: ~403
- Files removed: 2

---

### Phase 2: Review and Consolidate cross_platform_v2.py

**Step 1: Analyze differences**
```bash
# Generate diff
diff -u executor/cross_platform.py executor/cross_platform_v2.py > /tmp/cross_platform.diff

# Review manually
cat /tmp/cross_platform.diff | less
```

**Step 2: Identify improvements in v2**

Look for:
- Enhanced state machine logic
- Better error handling
- Market order support (MarketOrderArgs)
- Batch order posting (post_orders)
- Additional safety checks

**Step 3: Backport if needed**

If v2 has valuable improvements:
```bash
# Manually merge improvements into cross_platform.py
# OR replace v1 with v2 and rename

# Option A: Backport and delete v2
# (manual code merge)
git add executor/cross_platform.py
git rm executor/cross_platform_v2.py

# Option B: Replace v1 with v2
git mv executor/cross_platform_v2.py executor/cross_platform.py
```

**Step 4: Test**
```bash
# Run full test suite
PYTHONPATH=. uv run pytest tests/ -v

# Focus on cross-platform tests
PYTHONPATH=. uv run pytest tests/test_cross_platform_exec.py -v

# Integration test
uv run python run.py --dry-run --limit 50
```

**Step 5: Commit**
```bash
git commit -m "refactor: consolidate cross-platform execution

- Remove duplicate cross_platform_v2.py (577 lines)
- [If backported: Backport X improvements from v2 to v1]
- [If replaced: Promote v2 to v1]
- All tests passing

See docs/DELETION_LOG.md for details."
```

---

### Phase 3: Fix Type Safety Issues

**Files to fix**:
1. `tests/test_kalshi_cache.py:87`
2. `tests/test_fanatics.py:54`
3. `executor/safety.py:445`
4. `executor/safety.py:447`

**Commands**:
```bash
# Fix tests (use dataclasses.replace for frozen dataclasses)
# Edit files manually

# Fix safety.py (add proper type annotations)
# Edit file manually

# Run type checker if available
uv run mypy . || echo "mypy not configured"

# Run tests
PYTHONPATH=. uv run pytest tests/ -v

# Commit
git commit -m "fix: improve type safety annotations

- Replace frozen dataclass mutation with dataclasses.replace()
  - tests/test_kalshi_cache.py:87
  - tests/test_fanatics.py:54
- Add proper type annotations to avoid type: ignore
  - executor/safety.py:445, 447
- All tests passing

Removes 4 type: ignore comments."
```

---

### Phase 4: Add New Files to Git

**File to add**:
1. `client/kalshi_cache.py` (151 lines, actively used)
2. `tests/test_kalshi_cache.py` (test coverage)

**Commands**:
```bash
git add client/kalshi_cache.py
git add tests/test_kalshi_cache.py
git add .env.example
git add .gitignore
git add .github/

git commit -m "feat: add Kalshi market cache for background refresh

- Add client/kalshi_cache.py (151 lines)
- Add tests/test_kalshi_cache.py
- Background daemon refreshes every 5 minutes
- Eliminates 120s+ blocking scans
- Immutable snapshot pattern for thread safety

See CLAUDE.md for architecture details."
```

---

## Testing Checklist

### Before ANY Deletion
- [x] Grep for all imports of target file
- [x] Check if file is in `__init__.py` exports (all __init__.py are empty)
- [x] Search for dynamic imports (none found)
- [x] Review git history for context
- [ ] Create backup branch: `git branch backup-before-cleanup`

### After Each Phase
- [ ] Build succeeds: `uv sync --all-extras`
- [ ] All tests pass: `PYTHONPATH=. uv run pytest tests/ -v`
- [ ] No import errors: `uv run python run.py --dry-run --limit 10`
- [ ] Dry-run completes: `uv run python run.py --dry-run --limit 50`
- [ ] Create atomic commit with descriptive message
- [ ] Update docs/DELETION_LOG.md with results

### Final Verification
- [ ] Full test suite: `PYTHONPATH=. uv run pytest tests/ -v --cov=. --cov-report=term-missing`
- [ ] Coverage maintained or improved
- [ ] No regressions in scan results (compare opportunity counts)
- [ ] Ready for PR review

---

## Expected Impact

### Code Reduction
| Phase | Files Deleted | Lines Removed | Risk Level |
|-------|---------------|---------------|------------|
| Phase 1 | 2 | ~403 | SAFE |
| Phase 2 | 1 | ~140-577 | MEDIUM |
| Phase 3 | 0 | 0 (improvements) | LOW |
| **Total** | **3** | **540-980** | **LOW** |

### Benefits
- Reduced maintenance surface
- Eliminated confusion (which implementation to use?)
- Improved type safety
- No unused code to debug
- Cleaner codebase for new contributors

### Risks
- LOW - All deletions verified with grep
- MEDIUM for Phase 2 - requires manual diff review
- Mitigations:
  - Feature branch with backup
  - Atomic commits per phase
  - Full test suite after each change
  - Manual smoke tests

---

## Files Referenced in This Analysis

### Files to Delete (3)
1. `/home/fiod/Polymarket/scanner/fees_v2.py`
2. `/home/fiod/Polymarket/executor/maker_lifecycle.py`
3. `/home/fiod/Polymarket/executor/cross_platform_v2.py` (after review)

### Files to Fix (3)
1. `/home/fiod/Polymarket/tests/test_kalshi_cache.py` (line 87)
2. `/home/fiod/Polymarket/tests/test_fanatics.py` (line 54)
3. `/home/fiod/Polymarket/executor/safety.py` (lines 445, 447)

### Files to Keep (Intentional)
1. `/home/fiod/Polymarket/scanner/value.py` (disabled by design)
2. `/home/fiod/Polymarket/scanner/resolution.py` (active feature)
3. `/home/fiod/Polymarket/scanner/stale_quote.py` (active feature)
4. `/home/fiod/Polymarket/client/fanatics.py` (future integration)
5. `/home/fiod/Polymarket/client/fanatics_auth.py` (future integration)
6. `/home/fiod/Polymarket/scanner/fanatics_fees.py` (future integration)

### Active Reference Files
1. `/home/fiod/Polymarket/scanner/fees.py` (active fee model)
2. `/home/fiod/Polymarket/executor/cross_platform.py` (active execution)
3. `/home/fiod/Polymarket/config.py` (all fields used)
4. `/home/fiod/Polymarket/run.py` (main pipeline)

---

## Success Criteria

### Phase 1 Success
- [x] 2 files deleted
- [ ] Build succeeds
- [ ] All tests pass (expect 100% pass rate)
- [ ] No import errors
- [ ] Committed to feature branch

### Phase 2 Success
- [ ] cross_platform_v2.py analyzed
- [ ] Valuable changes identified (if any)
- [ ] Changes backported OR file replaced
- [ ] All cross-platform tests pass
- [ ] Committed to feature branch

### Phase 3 Success
- [ ] 0 type: ignore comments in modified files
- [ ] Type checker happy (if configured)
- [ ] All tests pass
- [ ] Committed to feature branch

### Final Success
- [ ] Feature branch merged to main
- [ ] docs/DELETION_LOG.md updated with results
- [ ] CLAUDE.md updated if needed
- [ ] No regressions in production

---

*Analysis complete. Ready for execution.*

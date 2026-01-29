# PyAEDT Version Compatibility Investigation

## Current Environment
- **pyAEDT:** 0.23.0
- **pyEPR-quantum:** 0.9.0
- **qiskit-metal:** ≥0.5.0

## Problem Statement
Workflows that previously worked with pyAEDT 0.6.x are now breaking with pyAEDT 0.23.0, manifesting as gRPC errors.

## Root Cause Analysis

### 1. gRPC Transport Mechanism Change
**pyAEDT 0.23.0** (released Nov 27, 2025) introduced:
> "Add compatibility with new grpc transport mechanism"

This is a **breaking change** in how pyAEDT communicates with Ansys AEDT.

**Key Changes:**
- **AEDT 2022 R2+**: gRPC is now the default and preferred interface
- **Windows**: gRPC replaced COM interface starting from AEDT 2022 R2
- **Linux**: gRPC is the exclusive interface for all AEDT versions
- **Older versions**: Used COM interface (Windows) which is now deprecated

### 2. Version Numbering Confusion
PyAEDT uses **calendar versioning**, not semantic versioning:
- `0.6.x` likely refers to a 2021-2022 release
- `0.23.0` is from November 2025
- This is NOT a minor version bump - it's **3+ years** of changes!

### 3. gRPC Library Version Constraints
PyAEDT has specific `grpcio` requirements:
- **v0.22.1**: `grpcio <1.77, >=1.50.0`
- **v0.20.1**: `grpcio <1.76, >=1.50.0`

Mismatches between your environment's `grpcio` and AEDT backend can cause errors.

## Potential Issues

### Issue 1: pyEPR Compatibility
**pyEPR underwent major refactoring around v0.8-dev (2020):**
- Class renames and API changes
- If using old pyEPR code with new pyAEDT, expect breakage

**Current pyEPR version (0.9.0)** should be compatible, but:
- May have assumptions about pyAEDT API that changed
- gRPC communication patterns may differ from COM

### Issue 2: Qiskit Metal Integration
**Qiskit Metal uses pyAEDT via `renderer_ansys_pyaedt`:**
- `QPyaedt` class initiates `pyAEDT.Desktop` sessions
- Default options include pyEPR analysis settings
- If Qiskit Metal was developed/tested against older pyAEDT, it may use deprecated APIs

### Issue 3: AEDT Version Mismatch
**Critical Question:** What version of Ansys AEDT are you running?
- If **AEDT < 2022 R2**: May not support new gRPC mechanism
- If **AEDT ≥ 2022 R2**: Should work, but may need configuration

## Diagnostic Steps

### Step 1: Check gRPC Configuration
```python
import pyaedt
print(f"PyAEDT version: {pyaedt.__version__}")

# Check if gRPC is being used
from pyaedt import Desktop
desktop = Desktop()
print(f"Connection type: {desktop.odesktop}")
```

### Step 2: Check grpcio Version
```bash
uv pip list | grep grpcio
```

Expected: `grpcio>=1.50.0,<1.77`

### Step 3: Enable gRPC Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show detailed gRPC communication errors.

### Step 4: Test Minimal pyAEDT Example
```python
from pyaedt import Hfss

# This should fail with gRPC error if there's a fundamental issue
hfss = Hfss(specified_version="2023.1", non_graphical=True)
hfss.close_desktop()
```

## Potential Solutions

### Solution 1: Pin to Compatible pyAEDT Version
**If you need immediate stability:**
```toml
[project]
dependencies = [
    "pyaedt==0.9.9",  # Last stable version before major gRPC changes
    # OR
    "pyaedt>=0.20.0,<0.23.0",  # Avoid 0.23.x
]
```

**Pros:** Immediate fix, known working state
**Cons:** Miss out on bug fixes and new features

### Solution 2: Update AEDT Version
**If using older AEDT:**
- Upgrade to **AEDT 2023 R1+** for full gRPC support
- Ensure license supports newer version

### Solution 3: Update Qiskit Metal
**Check if newer Qiskit Metal version exists:**
```bash
uv pip install --upgrade qiskit-metal
```

Newer versions may have pyAEDT 0.23.0 compatibility fixes.

### Solution 4: Use pyAEDT in Compatibility Mode
**Try forcing COM interface (Windows only):**
```python
from pyaedt import Desktop
desktop = Desktop(specified_version="2023.1", use_grpc=False)
```

**Note:** This may not work with AEDT 2022 R2+ on Windows.

### Solution 5: Isolate and Test
**Create minimal reproduction:**
1. Test pyAEDT alone (without pyEPR/Qiskit Metal)
2. Test pyEPR with pyAEDT
3. Test full stack

This will identify which layer is failing.

## Recommended Action Plan

### Immediate (Today):
1. ✅ Document current AEDT version
2. ✅ Check `grpcio` version compatibility
3. ✅ Enable gRPC debug logging
4. ✅ Run minimal pyAEDT test to isolate issue

### Short-term (This Week):
1. If AEDT < 2022 R2: Consider upgrading AEDT
2. If gRPC errors persist: Pin pyAEDT to 0.9.9 or 0.22.x
3. Test with different pyAEDT versions to find working range
4. Report issue to Qiskit Metal if it's their integration layer

### Long-term (Next Sprint):
1. Work with Qiskit Metal team to ensure 0.23.0 compatibility
2. Update SQuADDS to handle both old and new pyAEDT APIs
3. Add version compatibility matrix to documentation

## Additional Resources

- **PyAEDT Release Notes:** https://pyaedt.docs.pyansys.com/version/stable/Getting_started/ReleaseNotes.html
- **gRPC Troubleshooting:** https://pyaedt.docs.pyansys.com/version/stable/Getting_started/FAQ.html
- **Qiskit Metal Ansys Integration:** https://qiskit.org/documentation/metal/

## Next Steps

Please provide:
1. **AEDT version** you're running
2. **Exact gRPC error message** you're seeing
3. **Code snippet** that triggers the error
4. **Output of** `uv pip list | grep -E "(grpc|pyaedt|pyepr)"`

This will help narrow down the exact issue and solution.

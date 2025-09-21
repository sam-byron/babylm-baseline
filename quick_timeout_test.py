#!/usr/bin/env python3
"""
Test some of the TIMEOUT scripts to see if any can be fixed quickly
"""

import subprocess
import sys
import time

def test_script(script_name, timeout=15):
    """Test a single script with shorter timeout"""
    try:
        start_time = time.time()
        result = subprocess.run([
            sys.executable, f'auto_tests/{script_name}'
        ], capture_output=True, text=True, timeout=timeout)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            return "SUCCESS", duration, ""
        else:
            # Get first few lines of stderr for context
            stderr_preview = result.stderr[:300] + "..." if len(result.stderr) > 300 else result.stderr
            return "FAIL", duration, stderr_preview
            
    except subprocess.TimeoutExpired:
        return "TIMEOUT", timeout, "Timeout after {}s".format(timeout)
    except Exception as e:
        return "ERROR", 0, str(e)

# Test some TIMEOUT scripts that might be quick fixes
timeout_scripts = [
    "run_all_tests.py",
    "run_data_loader_tests.py", 
    "specialized_tests.py",
    "test_current_pipeline.py"
]

print("🔍 Testing some TIMEOUT scripts with shorter timeout:")
print("=" * 60)

for i, script in enumerate(timeout_scripts, 1):
    print(f"[{i}/{len(timeout_scripts)}] Testing {script}...")
    
    status, duration, error = test_script(script, timeout=15)
    
    if status == "SUCCESS":
        print(f"  ✅ PASSED ({duration:.2f}s)")
    elif status == "TIMEOUT":
        print(f"  ⏰ TIMEOUT ({duration:.2f}s) - likely needs computation optimization")
    elif status == "FAIL":
        print(f"  ❌ FAILED ({duration:.2f}s)")
        # Show first line of error for quick diagnosis
        if error:
            first_error_line = error.split('\n')[0] if '\n' in error else error[:100]
            print(f"    Quick diagnosis: {first_error_line}")
    else:
        print(f"  💥 ERROR: {error}")
    print()

print("Done testing TIMEOUT scripts!")
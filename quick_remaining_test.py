#!/usr/bin/env python3
"""
Quick test of remaining failed scripts to continue our fixing process
"""

import subprocess
import sys
import time

def test_script(script_name, timeout=20):
    """Test a single script with timeout"""
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
            stderr_preview = result.stderr[:200] + "..." if len(result.stderr) > 200 else result.stderr
            return "FAIL", duration, stderr_preview
            
    except subprocess.TimeoutExpired:
        return "TIMEOUT", timeout, "Timeout after {}s".format(timeout)
    except Exception as e:
        return "ERROR", 0, str(e)

# List of remaining OTHER failures from our previous analysis
remaining_other_scripts = [
    "run_tests.py",
    "test_blimp.py", 
    "test_blimp_simple.py"
]

print("🔍 Testing remaining OTHER failure scripts:")
print("=" * 50)

for i, script in enumerate(remaining_other_scripts, 1):
    print(f"[{i}/{len(remaining_other_scripts)}] Testing {script}...")
    
    status, duration, error = test_script(script)
    
    if status == "SUCCESS":
        print(f"  ✅ PASSED ({duration:.2f}s)")
    elif status == "TIMEOUT":
        print(f"  ⏰ TIMEOUT ({duration:.2f}s)")
    elif status == "FAIL":
        print(f"  ❌ FAILED ({duration:.2f}s)")
        if error:
            print(f"    Error: {error}")
    else:
        print(f"  💥 ERROR: {error}")
    print()

print("Done testing remaining OTHER scripts!")
#!/usr/bin/env python3
"""
Final comprehensive test to count our overall success rate
"""

import subprocess
import sys
import os
from pathlib import Path

def test_script(script_name, timeout=15):
    """Test a single script"""
    try:
        result = subprocess.run([
            sys.executable, f'auto_tests/{script_name}'
        ], capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception:
        return False

# Get all Python scripts in auto_tests
auto_tests_dir = Path('auto_tests')
all_scripts = [f.name for f in auto_tests_dir.glob('*.py') if not f.name.startswith('__')]
all_scripts.sort()

print(f"🧪 FINAL COMPREHENSIVE TEST - Found {len(all_scripts)} scripts")
print("=" * 70)

successful = 0
timeouts = 0
failures = 0

# Test each script
for i, script in enumerate(all_scripts, 1):
    print(f"[{i:2d}/{len(all_scripts)}] {script:35s} ", end="", flush=True)
    
    result = test_script(script)
    
    if result is True:
        print("✅")
        successful += 1
    elif result == "TIMEOUT":
        print("⏰")
        timeouts += 1
    else:
        print("❌") 
        failures += 1

# Print summary
print("\n" + "=" * 70)
print("📊 FINAL RESULTS SUMMARY")
print("=" * 70)
print(f"✅ Successful: {successful:2d}/{len(all_scripts)} ({successful/len(all_scripts)*100:.1f}%)")
print(f"⏰ Timeouts:   {timeouts:2d}/{len(all_scripts)} ({timeouts/len(all_scripts)*100:.1f}%)")
print(f"❌ Failures:   {failures:2d}/{len(all_scripts)} ({failures/len(all_scripts)*100:.1f}%)")
print()

if successful >= len(all_scripts) * 0.6:  # 60% or higher
    print("🎉 EXCELLENT PROGRESS! Over 60% success rate achieved!")
elif successful >= len(all_scripts) * 0.5:  # 50% or higher  
    print("🎯 GOOD PROGRESS! Over 50% success rate achieved!")
else:
    print("📈 Progress made, but more work needed.")

print(f"\n💡 Summary: {successful} working scripts out of {len(all_scripts)} total")
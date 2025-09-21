#!/usr/bin/env python3
"""
Quick check to identify which scripts are still failing due to timeouts
"""
import os
import subprocess
import sys
import time

def test_script(script_path, timeout=20):
    """Test a single script with timeout"""
    try:
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=timeout)
        
        if result.returncode == 0:
            return "SUCCESS"
        else:
            return f"ERROR: {result.stderr.strip()[:100]}..."
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"EXCEPTION: {str(e)[:100]}..."

def main():
    # Focus on previously identified timeout scripts
    timeout_scripts = [
        "test_data_loader.py",
        "test_current_pipeline.py", 
        "specialized_tests.py",
        "test_dynamic_collator.py",
        "test_full_collator.py",
        "test_large_dataset.py",
        "test_tokenizer_performance.py",
        "test_validation_logic.py",
        "test_comprehensive_fix.py"
    ]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("🔍 QUICK TIMEOUT CHECK")
    print("=" * 50)
    
    success_count = 0
    timeout_count = 0
    error_count = 0
    
    for script in timeout_scripts:
        script_path = os.path.join(base_dir, script)
        if not os.path.exists(script_path):
            print(f"⚠️  {script}: NOT FOUND")
            continue
            
        print(f"Testing {script}...", end=" ", flush=True)
        status = test_script(script_path)
        
        if status == "SUCCESS":
            print("✅")
            success_count += 1
        elif status == "TIMEOUT":
            print("⏱️  TIMEOUT")
            timeout_count += 1
        else:
            print(f"❌ {status}")
            error_count += 1
    
    total = success_count + timeout_count + error_count
    print("\n" + "=" * 50)
    print(f"📊 RESULTS: {success_count}/{total} SUCCESS, {timeout_count} TIMEOUT, {error_count} ERROR")
    
if __name__ == "__main__":
    main()
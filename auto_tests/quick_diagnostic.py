#!/usr/bin/env python3
"""
Quick diagnostic script to identify current failing scripts
"""
import os
import subprocess
import sys
import time

def test_script(script_path, timeout=60):
    """Test a single script with timeout"""
    try:
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=timeout, 
        cwd="/home/sam-byron/engineering/ML/playground/babylm/bnc/bl-bnc-custom-ltg-bert")
        
        if result.returncode == 0:
            return "SUCCESS"
        else:
            # Get first line of error for classification
            if result.stderr:
                first_error = result.stderr.split('\n')[0]
                return f"ERROR: {first_error[:80]}..."
            else:
                return f"ERROR: Exit code {result.returncode}"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"EXCEPTION: {str(e)[:80]}..."

def main():
    base_dir = "/home/sam-byron/engineering/ML/playground/babylm/bnc/bl-bnc-custom-ltg-bert/auto_tests"
    
    # Get all Python scripts
    all_scripts = [f for f in os.listdir(base_dir) if f.endswith('.py')]
    
    # Skip utility scripts
    skip_scripts = {
        'run_all_tests.py',
        'fix_imports.py', 
        'fix_config_keys.py',
        'quick_timeout_check.py',
        'quick_diagnostic.py'  # This script
    }
    
    test_scripts = [s for s in all_scripts if s not in skip_scripts]
    test_scripts.sort()
    
    print(f"🔍 DIAGNOSTIC: Testing {len(test_scripts)} scripts")
    print("=" * 70)
    
    success_count = 0
    timeout_count = 0 
    error_count = 0
    blimp_related = 0
    
    failed_scripts = []
    timeout_scripts = []
    blimp_scripts = []
    
    for script in test_scripts:
        script_path = os.path.join(base_dir, script)
        print(f"Testing {script}...", end=" ", flush=True)
        
        status = test_script(script_path)
        
        if status == "SUCCESS":
            print("✅")
            success_count += 1
        elif status == "TIMEOUT":
            print("⏱️ ")
            timeout_count += 1
            timeout_scripts.append(script)
        elif "blimp" in script.lower() or "BLIMP" in status:
            print("🔵 BLIMP")
            blimp_related += 1
            blimp_scripts.append(script)
        else:
            print(f"❌")
            error_count += 1
            failed_scripts.append((script, status))
    
    total = len(test_scripts)
    non_blimp_failures = error_count + timeout_count
    
    print("\n" + "=" * 70)
    print("📊 SUMMARY RESULTS")
    print("=" * 70)
    print(f"✅ Successful: {success_count}/{total} ({success_count/total*100:.1f}%)")
    print(f"❌ Failed (non-BLIMP): {non_blimp_failures}/{total}")
    print(f"   - Error failures: {error_count}")
    print(f"   - Timeout failures: {timeout_count}")
    print(f"🔵 BLIMP-related: {blimp_related}")
    
    if timeout_scripts:
        print(f"\n⏱️  TIMEOUT SCRIPTS ({len(timeout_scripts)}):")
        for script in timeout_scripts:
            print(f"   - {script}")
    
    if failed_scripts:
        print(f"\n❌ ERROR SCRIPTS ({len(failed_scripts)}):")
        for script, error in failed_scripts:
            print(f"   - {script}")
            print(f"     {error}")
    
    if blimp_scripts:
        print(f"\n🔵 BLIMP SCRIPTS ({len(blimp_scripts)}):")
        for script in blimp_scripts:
            print(f"   - {script}")
    
    print(f"\n🎯 NON-FUNCTIONING SCRIPTS (excluding BLIMP): {non_blimp_failures}")
    return non_blimp_failures

if __name__ == "__main__":
    main()
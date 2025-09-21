#!/usr/bin/env python3
"""
Script runner to test all auto_tests scripts systematically.
"""


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import os
import sys
import subprocess
import time

def run_script(script_name, timeout=180):
    """Run a single script and capture its result."""
    print(f"\n{'='*50}")
    print(f"TESTING: {script_name}")
    print('='*50)
    
    try:
        # Set working directory to project root
        cwd = "/home/sam-byron/engineering/ML/playground/babylm/bnc/bl-bnc-custom-ltg-bert"
        
        # Run the script
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, f"auto_tests/{script_name}"], 
            cwd=cwd,
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        end_time = time.time()
        
        success = result.returncode == 0
        duration = end_time - start_time
        
        print(f"✅ SUCCESS ({duration:.2f}s)" if success else f"❌ FAILED ({duration:.2f}s)")
        
        if result.stdout:
            print("\n--- STDOUT ---")
            print(result.stdout[:1000] + ("..." if len(result.stdout) > 1000 else ""))
            
        if result.stderr:
            print("\n--- STDERR ---")
            print(result.stderr[:1000] + ("..." if len(result.stderr) > 1000 else ""))
        
        return {
            "script": script_name,
            "success": success,
            "duration": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT after {timeout}s")
        return {
            "script": script_name,
            "success": False,
            "duration": timeout,
            "stdout": "",
            "stderr": f"Script timed out after {timeout} seconds",
            "returncode": -1
        }
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")
        return {
            "script": script_name,
            "success": False,
            "duration": 0,
            "stdout": "",
            "stderr": str(e),
            "returncode": -2
        }

def main():
    # Get all Python scripts in auto_tests
    auto_tests_dir = "/home/sam-byron/engineering/ML/playground/babylm/bnc/bl-bnc-custom-ltg-bert/auto_tests"
    all_scripts = [f for f in os.listdir(auto_tests_dir) if f.endswith('.py')]
    
    # Skip scripts that are known to be slow or problematic
    skip_scripts = {
        'run_all_tests.py',  # Don't run ourselves
        'fix_imports.py',  # Utility script
        'fix_config_keys.py',  # Utility script
        'quick_timeout_check.py',  # Our diagnostic script
    }
    
    # Filter out only utility scripts - allow all test scripts to run for realistic scenarios
    test_scripts = [s for s in all_scripts if s not in skip_scripts]
    test_scripts.sort()
    
    print(f"Found {len(all_scripts)} total scripts, testing {len(test_scripts)} (comprehensive realistic testing)")
    
    results = []
    
    # Test scripts one by one with generous timeout for large datasets
    for script in test_scripts:
        result = run_script(script, timeout=180)  # 3 minutes per script for realistic scenarios
        results.append(result)
        
        # Brief pause between tests to allow system recovery
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print('='*70)
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"✅ Successful: {len(successful)}/{len(test_scripts)}")
    print(f"❌ Failed: {len(failed)}/{len(test_scripts)}")
    print(f"⏭️  Skipped: {len(skip_scripts)} utility scripts (running all test scripts)")
    
    if failed:
        print("\n❌ FAILED SCRIPTS:")
        for result in failed:
            print(f"  - {result['script']} (exit code: {result['returncode']})")
            # Print first line of error for quick diagnosis
            if result['stderr']:
                error_line = result['stderr'].split('\n')[0]
                print(f"    Error: {error_line[:100]}...")
    
    if successful:
        print(f"\n✅ SUCCESSFUL SCRIPTS ({len(successful)}):")
        for result in successful:
            print(f"  - {result['script']} ({result['duration']:.1f}s)")
    
    return len(failed) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
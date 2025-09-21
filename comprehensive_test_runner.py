#!/usr/bin/env python3
"""
Final comprehensive test runner for all auto_tests scripts
"""

import os
import sys
import subprocess
import time
import json

def run_single_test(script_name, timeout=30):
    """Run a single test script with timeout and error handling."""
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, f"auto_tests/{script_name}"],
            cwd="/home/sam-byron/engineering/ML/playground/babylm/bnc/bl-bnc-custom-ltg-bert",
            capture_output=True,
            text=True,
            timeout=timeout
        )
        end_time = time.time()
        
        return {
            "script": script_name,
            "success": result.returncode == 0,
            "duration": round(end_time - start_time, 2),
            "returncode": result.returncode,
            "error_type": classify_error(result.stderr) if result.returncode != 0 else None,
            "stderr_preview": result.stderr[:200] + "..." if len(result.stderr) > 200 else result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "script": script_name,
            "success": False,
            "duration": timeout,
            "returncode": -1,
            "error_type": "TIMEOUT",
            "stderr_preview": f"Timeout after {timeout}s"
        }
    except Exception as e:
        return {
            "script": script_name,
            "success": False,
            "duration": 0,
            "returncode": -2,
            "error_type": "EXCEPTION",
            "stderr_preview": str(e)
        }

def classify_error(stderr):
    """Classify the type of error from stderr."""
    if "No module named 'torch'" in stderr:
        return "MISSING_TORCH"
    elif "cannot import name" in stderr:
        return "IMPORT_ERROR"
    elif "CUDA error: out of memory" in stderr:
        return "CUDA_OOM"
    elif "KeyError:" in stderr:
        return "CONFIG_KEY_ERROR"
    elif "TypeError:" in stderr:
        return "TYPE_ERROR"
    elif "AttributeError:" in stderr:
        return "ATTRIBUTE_ERROR"
    elif "FileNotFoundError" in stderr or "No such file" in stderr:
        return "FILE_NOT_FOUND"
    else:
        return "OTHER"

def main():
    print("🧪 COMPREHENSIVE AUTO_TESTS REPORT")
    print("=" * 60)
    
    # Get all Python scripts
    auto_tests_dir = "/home/sam-byron/engineering/ML/playground/babylm/bnc/bl-bnc-custom-ltg-bert/auto_tests"
    all_scripts = sorted([f for f in os.listdir(auto_tests_dir) if f.endswith('.py')])
    
    print(f"Found {len(all_scripts)} Python scripts")
    print()
    
    results = []
    
    # Test each script
    for i, script in enumerate(all_scripts, 1):
        print(f"[{i:2d}/{len(all_scripts)}] Testing {script}...", end=" ")
        result = run_single_test(script, timeout=20)
        results.append(result)
        
        if result["success"]:
            print(f"✅ ({result['duration']}s)")
        else:
            print(f"❌ {result['error_type']} ({result['duration']}s)")
    
    # Analyze results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {len(successful)}/{len(all_scripts)} ({len(successful)/len(all_scripts)*100:.1f}%)")
    print(f"❌ Failed: {len(failed)}/{len(all_scripts)} ({len(failed)/len(all_scripts)*100:.1f}%)")
    
    # Group failures by error type
    error_groups = {}
    for result in failed:
        error_type = result["error_type"]
        if error_type not in error_groups:
            error_groups[error_type] = []
        error_groups[error_type].append(result["script"])
    
    if error_groups:
        print("\n🔍 FAILURE BREAKDOWN:")
        for error_type, scripts in sorted(error_groups.items()):
            print(f"  {error_type}: {len(scripts)} scripts")
            for script in scripts[:3]:  # Show first 3
                print(f"    - {script}")
            if len(scripts) > 3:
                print(f"    ... and {len(scripts) - 3} more")
    
    if successful:
        print(f"\n✅ WORKING SCRIPTS ({len(successful)}):")
        for result in successful:
            print(f"  - {result['script']} ({result['duration']}s)")
    
    print("\n📝 Full results saved to test_results.json")
    
    # Save detailed results
    with open("test_results.json", "w") as f:
        json.dump({
            "summary": {
                "total": len(all_scripts),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful)/len(all_scripts)*100
            },
            "error_groups": error_groups,
            "results": results
        }, f, indent=2)
    
    return len(failed) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
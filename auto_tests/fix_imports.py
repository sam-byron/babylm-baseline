#!/usr/bin/env python3
"""
Script to fix imports in all auto_tests scripts by adding parent directory to sys.path
"""

import os
import sys
import glob

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def fix_imports_in_file(filepath):
    """Add sys.path fix to the beginning of a Python file"""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Skip if already has path fix
    if 'sys.path.insert(0, parent_dir)' in content:
        return False
    
    # Find the first import statement or class/function definition
    lines = content.split('\n')
    insert_line = 0
    
    # Skip shebang and docstring
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#!') or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        if (stripped.startswith('import ') or stripped.startswith('from ') or 
            stripped.startswith('def ') or stripped.startswith('class ') or
            stripped.startswith('if __name__')):
            insert_line = i
            break
    
    # Insert the path fix
    path_fix = [
        "",
        "# Add parent directory to path for imports",
        "import os",
        "import sys",
        "parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))",
        "if parent_dir not in sys.path:",
        "    sys.path.insert(0, parent_dir)",
        ""
    ]
    
    # Insert at the right location
    new_lines = lines[:insert_line] + path_fix + lines[insert_line:]
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(new_lines))
    
    return True

def main():
    # Get all Python files in auto_tests directory
    auto_tests_dir = os.path.dirname(os.path.abspath(__file__))
    python_files = glob.glob(os.path.join(auto_tests_dir, '*.py'))
    
    fixed = 0
    skipped = 0
    
    for filepath in python_files:
        filename = os.path.basename(filepath)
        if filename == 'fix_imports.py':  # Skip this script
            continue
            
        try:
            if fix_imports_in_file(filepath):
                print(f"✓ Fixed imports in {filename}")
                fixed += 1
            else:
                print(f"- Skipped {filename} (already fixed)")
                skipped += 1
        except Exception as e:
            print(f"✗ Error fixing {filename}: {e}")
    
    print(f"\nSummary: {fixed} files fixed, {skipped} files skipped")

if __name__ == "__main__":
    main()
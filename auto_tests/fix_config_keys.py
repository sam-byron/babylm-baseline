#!/usr/bin/env python3
"""
Script to fix common config key issues in auto_tests scripts
"""


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import os
import re

def fix_config_keys():
    """Fix common config key mismatches."""
    
    auto_tests_dir = "/home/sam-byron/engineering/ML/playground/babylm/bnc/bl-bnc-custom-ltg-bert/auto_tests"
    
    # Common replacements needed
    replacements = [
        (r"config\[\'max_position_embeddings\'\]", "config.get('max_position_embeddings', config.get('block_size', 512))"),
        (r"config\[\"max_position_embeddings\"\]", "config.get('max_position_embeddings', config.get('block_size', 512))"),
        (r"config_dict\[\'max_position_embeddings\'\]", "config_dict.get('max_position_embeddings', config_dict.get('block_size', 512))"),
        (r"config_dict\[\"max_position_embeddings\"\]", "config_dict.get('max_position_embeddings', config_dict.get('block_size', 512))"),
        # Default tokenizer path fixes
        (r'"tokenizer\.json"', '"data/pretrain/wordpiece_vocab.json"'),
        (r"'tokenizer\.json'", "'data/pretrain/wordpiece_vocab.json'"),
        (r'default="tokenizer\.json"', 'default="data/pretrain/wordpiece_vocab.json"'),
    ]
    
    fixed_files = []
    
    for filename in os.listdir(auto_tests_dir):
        if not filename.endswith('.py'):
            continue
            
        filepath = os.path.join(auto_tests_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply replacements
            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)
            
            # Write back if changed
            if content != original_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_files.append(filename)
                print(f"✅ Fixed: {filename}")
        
        except Exception as e:
            print(f"❌ Error fixing {filename}: {e}")
    
    print(f"\n🔧 Fixed {len(fixed_files)} files")
    return fixed_files

if __name__ == "__main__":
    fixed = fix_config_keys()
    print("Done!")
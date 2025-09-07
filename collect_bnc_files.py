#!/usr/bin/env python3
import os
import glob
from pathlib import Path

def collect_bnc_files():
    """Traverse all subfolders in bnc, collect all .md files and concatenate them into a single train.md file"""
    
    bnc_path = "/home/sam-byron/engineering/ML/playground/babylm/bnc/bl-bnc-custom-ltg-bert/raw_corpus/bnc"
    output_file = "train.md"
    
    print(f"Collecting .md files from {bnc_path}")
    
    # Find all .md files recursively
    md_files = []
    for root, dirs, files in os.walk(bnc_path):
        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))
    
    print(f"Found {len(md_files)} .md files")
    
    # Sort files for consistent ordering
    md_files.sort()
    
    # Concatenate all files
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i, md_file in enumerate(md_files):
            try:
                with open(md_file, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    outfile.write('\n\n')  # Add separator between files
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(md_files)} files")
                    
            except Exception as e:
                print(f"Error reading {md_file}: {e}")
                continue
    
    print(f"Successfully created {output_file}")

if __name__ == "__main__":
    collect_bnc_files()
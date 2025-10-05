#!/usr/bin/env bash
#
# collect_xml.sh — Copy all .xml files from a source tree into a flat destination folder
#
# Purpose
#   Recursively finds every .xml file under <source_dir> and copies it into <dest_dir>.
#   Uses -print0 and a null-delimited read loop to safely handle filenames with spaces
#   and special characters.
#
# Usage
#   collect_xml.sh <source_dir> <dest_dir>
#
# Examples
#   ./collect_xml.sh ./temp/Texts ./data/pretrain/bnc_xml
#   ./collect_xml.sh /mnt/BNC_raw/Texts ./temp/flat_xml
#
# Arguments
#   <source_dir>  Root directory to search for .xml files (searched recursively)
#   <dest_dir>    Destination directory (created if absent) where files are copied
#
# Notes & efficiency
#   - Uses `find … -print0` piped into a `read -d ''` loop for robustness with filenames.
#   - `cp -v` prints each copy operation (useful for auditing). Remove -v for quieter runs.
#   - Destination is not re-flattened by subfolders; consider adding a hashing/rename step
#     if source trees contain duplicates with identical basenames.

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <source_dir> <dest_dir>"
  exit 1
fi

src="$1"
dst="$2"

# make sure source exists
if [ ! -d "$src" ]; then
  echo "Error: source directory '$src' does not exist."
  exit 1
fi

# create destination if needed
mkdir -p "$dst"

# find and copy
find "$src" -type f -name '*.xml' -print0 | while IFS= read -r -d '' file; do
  cp -v "$file" "$dst"
done

echo "All .xml files copied from '$src' to '$dst'."
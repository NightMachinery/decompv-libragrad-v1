#!/usr/bin/env python3
##
import argparse
import json
import sys
from collections import defaultdict

from decompv.early_boot import (
    DONE_EXPERIMENTS_TO_MERGE,
)

def make_hashable(hash_dict):
    """
    Convert a hash dictionary to a hashable type.

    Method 1: Using frozenset of items
    Method 2: Using a sorted tuple of key-value pairs
    """
    # Method 1: Using frozenset
    return frozenset(hash_dict.items())

    # Method 2: Using sorted tuple
    # return tuple(sorted(hash_dict.items()))

def main():
    parser = argparse.ArgumentParser(description="Find duplicate hashes in a JSONL file.")
    parser.add_argument(
        "--path",
        default=DONE_EXPERIMENTS_TO_MERGE,
        help="Path to the JSONL file (default: DONE_EXPERIMENTS_TO_MERGE)",
    )
    args = parser.parse_args()

    hash_counts = defaultdict(list)

    try:
        with open(args.path, "r") as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                try:
                    data = json.loads(line)
                    hash_dict = data.get("hash")
                    if hash_dict is None:
                        print(f"Line {line_num}: 'hash' field is missing.", file=sys.stderr)
                        continue
                    # Convert hash dict to a hashable type
                    hash_key = make_hashable(hash_dict)
                    hash_counts[hash_key].append(line)

                except json.JSONDecodeError as e:
                    print(f"Line {line_num}: JSON decode error: {e}", file=sys.stderr)

    except Exception as e:
        print(f"Error reading file '{args.path}': {e}", file=sys.stderr)
        sys.exit(1)

    duplicates_found = False
    for hash_key, entries in hash_counts.items():
        if len(entries) > 1:
            duplicates_found = True
            for entry in entries:
                print(entry)

    if not duplicates_found:
        print("No duplicates found.", file=sys.stderr)

if __name__ == "__main__":
    main()

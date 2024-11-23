#!/usr/bin/env python3

import jsonlines
import shutil
import os

from decompv.early_boot import (
    DONE_EXPERIMENTS,
    tlg_me,
)

# Define the path to your DONE_EXPERIMENTS file
DONE_EXPERIMENTS = DONE_EXPERIMENTS

# Define the path for the backup and updated files
BACKUP_PATH = f"{DONE_EXPERIMENTS}.bak"
UPDATED_PATH = f"{DONE_EXPERIMENTS}.updated"

# Define the mapping from 'kind' to 'dataset'
KIND_TO_DATASET_MAPPING = {
    "CLIP": "CLIP1",
    # Add more mappings as needed, e.g.,
    # "CLIP2_KIND_VALUE": "CLIP2",
    # "CLIP3_KIND_VALUE": "CLIP3",
}

def backup_file(original_path, backup_path):
    """Create a backup of the original DONE_EXPERIMENTS file."""
    if os.path.exists(original_path):
        shutil.copy(original_path, backup_path)
        print(f"Backup created at '{backup_path}'.")
    else:
        print(f"Original file '{original_path}' does not exist. Exiting.")
        exit(1)

def process_done_experiments(original_path, updated_path, mapping):
    """Process the DONE_EXPERIMENTS file and update entries."""
    with jsonlines.open(original_path, mode='r') as reader, \
         jsonlines.open(updated_path, mode='w') as writer:

        for obj in reader:
            # Ensure 'hash' and 'script_type' exist
            hash_dict = obj.get("hash", {})
            script_type = hash_dict.get("script_type", "")

            # Check if the script_type is 'qual' and 'kind' exists
            if script_type == "qual" and "kind" in hash_dict:
                kind_value = hash_dict.pop("kind")

                # Map the 'kind' value to 'dataset'
                dataset_value = mapping.get(kind_value)

                if dataset_value:
                    hash_dict["dataset"] = dataset_value
                    print(f"Updated 'kind': '{kind_value}' to 'dataset': '{dataset_value}'.")
                else:
                    # Handle cases where the kind value is not in the mapping
                    print(f"Warning: 'kind' value '{kind_value}' not found in mapping. Skipping replacement.")
                    # Optionally, you can decide to skip, assign a default value, or raise an error
                    # For example, assigning a default dataset:
                    # hash_dict["dataset"] = f"{kind_value}1"

            # Write the (potentially) updated object to the new file
            writer.write(obj)

def overwrite_original(original_path, updated_path):
    """Overwrite the original DONE_EXPERIMENTS file with the updated one."""
    shutil.move(updated_path, original_path)
    print(f"Original file '{original_path}' has been overwritten with the updated data.")

def main():
    # Step 1: Backup the original file
    backup_file(DONE_EXPERIMENTS, BACKUP_PATH)

    # Step 2: Process the DONE_EXPERIMENTS file
    process_done_experiments(DONE_EXPERIMENTS, UPDATED_PATH, KIND_TO_DATASET_MAPPING)

    # Optional Step 3: Overwrite the original file with the updated one
    # Uncomment the following line if you want to replace the original file
    overwrite_original(DONE_EXPERIMENTS, UPDATED_PATH)

    print("Processing complete.")
    print(f"Updated file is available at '{UPDATED_PATH}'.")
    print(f"Backup of the original file is at '{BACKUP_PATH}'.")

if __name__ == "__main__":
    main()

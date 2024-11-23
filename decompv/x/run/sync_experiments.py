#!/usr/bin/env python3
##
import argparse
import subprocess
import os
import tempfile
import shutil
import sys
from typing import List
import sh

from pynight.common_iterable import (
    to_iterable,
)

from decompv.early_boot import (
    DONE_EXPERIMENTS,
    DONE_EXPERIMENTS_TO_MERGE,
    DONE_EXPERIMENTS_REMOVED,
    HOME,
)


def local_path_to_remote(*, path: str, hostname: str) -> str:
    """
    Construct a remote path relative to the user's home directory on the remote server.
    """
    relative_path = os.path.relpath(path, start=HOME)
    remote_path = f"{hostname}:{relative_path}"
    return remote_path


def run_rsync(*, sources: List[str], dest: str) -> bool:
    sources = to_iterable(sources)

    base_args = [
        "--verbose",
        "--checksum",
        "--protect-args",
        "--human-readable",
        "--xattrs",
        "--times",
        "--info=progress2",
        "--partial-dir=.rsync-partial",
        "--archive",
        "--compress",
        "--ignore-missing-args",  #: ignore missing source args without error
    ]
    try:
        sh.rsync(*base_args, "--", *sources, dest)
        return True

    except sh.ErrorReturnCode as e:
        print(
            f"\n------\nError during rsync operation: {str(e)}\n------\n",
            file=sys.stderr,
        )
        return False


def merge_sort_files(
    *,
    file_paths: List[str],
    output_path: str,
) -> None:
    #: Exclude non-existent paths from file_paths:
    file_paths = [path for path in file_paths if os.path.exists(path)]

    if not file_paths:
        print("No valid file paths provided for sorting.", file=sys.stderr)
        return

    try:
        #: Copy the current environment variables and set LC_ALL to 'C' for consistent sorting
        env = os.environ.copy()
        env["LC_ALL"] = "C"

        # Determine the sort command (using 'gsort' if available, fallback to 'sort')
        sort_cmd = (
            "gsort"
            if subprocess.run(["which", "gsort"], capture_output=True).returncode == 0
            else "sort"
        )

        # Paths for temporary sorted files
        with tempfile.NamedTemporaryFile(delete=False) as temp_all_experiments:
            sorted_all_experiments_path = temp_all_experiments.name

        # First, sort and merge all input files into sorted_all_experiments_path
        sort_args = [sort_cmd, "--unique", *file_paths]
        with open(sorted_all_experiments_path, "w") as sorted_all_experiments_file:
            subprocess.run(sort_args, env=env, stdout=sorted_all_experiments_file)

        # Check if DONE_EXPERIMENTS_REMOVED exists
        if os.path.exists(DONE_EXPERIMENTS_REMOVED):
            # If it exists, sort and filter out removed experiments using 'comm -23'
            with tempfile.NamedTemporaryFile(delete=False) as temp_removed_experiments:
                sorted_removed_experiments_path = temp_removed_experiments.name

            # Sort the removed experiments
            sort_removed_args = [sort_cmd, "--unique", DONE_EXPERIMENTS_REMOVED]
            with open(
                sorted_removed_experiments_path, "w"
            ) as sorted_removed_experiments_file:
                subprocess.run(
                    sort_removed_args, env=env, stdout=sorted_removed_experiments_file
                )

            # Exclude lines in removed experiments using 'comm -23'
            comm_args = [
                "comm",
                "-23",
                sorted_all_experiments_path,
                sorted_removed_experiments_path,
            ]
            with open(output_path, "w") as output_file:
                subprocess.run(comm_args, env=env, stdout=output_file)

            # Clean up temporary removed experiments file
            os.remove(sorted_removed_experiments_path)
        else:
            # If DONE_EXPERIMENTS_REMOVED doesn't exist, just copy sorted experiments to output
            shutil.move(sorted_all_experiments_path, output_path)

        print(
            f"Merged and sorted files into {output_path}, excluding items in {DONE_EXPERIMENTS_REMOVED} if it exists."
        )
    except Exception as e:
        print(f"Error merging and sorting files: {str(e)}", file=sys.stderr)
        raise


def backup_file(
    *,
    original_path: str,
    backup_path: str = None,
    keep_n: int = 5,
):
    """
    Create a backup of the original file and maintain the latest 'keep_n' backups.

    Args:
    original_path (str): Path to the original file
    backup_path (str): Base path for the backup files
    keep_n (int): Number of recent backups to keep
    """
    if backup_path is None:
        backup_path = original_path

    if not os.path.exists(original_path):
        print(f"Original file '{original_path}' does not exist.")
        return

    # Shift existing backups
    for i in range(keep_n - 1, 0, -1):
        old_backup = f"{backup_path}.bak{i}"
        new_backup = f"{backup_path}.bak{i+1}"
        if os.path.exists(old_backup):
            if i == keep_n - 1:
                os.remove(old_backup)
            else:
                os.rename(old_backup, new_backup)

    # Create new backup
    new_backup = f"{backup_path}.bak1"
    shutil.copy(original_path, new_backup)
    print(f"Backup created at '{new_backup}'.")

    # Remove excess backups
    for i in range(keep_n + 1, 100):  # Arbitrary upper limit
        old_backup = f"{backup_path}.bak{i}"
        if os.path.exists(old_backup):
            os.remove(old_backup)

        else:
            break


def sync_with_to_merge():
    backup_file(original_path=DONE_EXPERIMENTS)

    #: Merge the new experiments into the main file
    merge_sort_files(
        file_paths=[DONE_EXPERIMENTS, DONE_EXPERIMENTS_TO_MERGE],
        output_path=DONE_EXPERIMENTS,
    )


def run_sync_and_merge(*, hostnames: List[str], upload: bool) -> None:
    temp_dir = tempfile.mkdtemp()
    temp_files = []
    exit_code = 0

    try:
        # Files to process
        files_to_sync = [DONE_EXPERIMENTS]

        # Download files from servers
        for hostname in hostnames:
            for file_path in files_to_sync:
                temp_file = os.path.join(
                    temp_dir, f"{hostname}_{os.path.basename(file_path)}"
                )
                remote_file_path = local_path_to_remote(
                    path=file_path, hostname=hostname
                )
                if not run_rsync(sources=remote_file_path, dest=temp_file):
                    print(f"Failed to sync from {hostname}", file=sys.stderr)
                    exit_code = 1
                elif os.path.exists(temp_file):
                    temp_files.append(temp_file)

        if not temp_files:
            print("No files were downloaded. Exiting.", file=sys.stderr)
            sys.exit(1)

        if temp_files:
            #: Merge and sort all downloaded files into DONE_EXPERIMENTS_TO_MERGE
            merge_sort_files(
                file_paths=temp_files, output_path=DONE_EXPERIMENTS_TO_MERGE
            )

        else:
            print("No files were downloaded. Skipping merge operation.")

        ##
        # Merge the new experiments into the main file
        sync_with_to_merge()
        ##

        if upload:
            # Define the list of files to upload
            files_to_upload = [DONE_EXPERIMENTS_TO_MERGE]

            # Include DONE_EXPERIMENTS_REMOVED if it exists locally
            if os.path.exists(DONE_EXPERIMENTS_REMOVED):
                files_to_upload.append(DONE_EXPERIMENTS_REMOVED)

            # Upload each file to all specified hostnames
            for file_path in files_to_upload:
                for hostname in hostnames:
                    remote_file_path = local_path_to_remote(
                        path=file_path, hostname=hostname
                    )
                    if run_rsync(sources=file_path, dest=remote_file_path):
                        print(f"Uploaded {file_path} to {hostname}")
                    else:
                        print(
                            f"Failed to upload {file_path} to {hostname}",
                            file=sys.stderr,
                        )
                        exit_code = 1

    finally:
        # Clean up temporary files and directory
        shutil.rmtree(temp_dir)

    sys.exit(exit_code)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync and merge DONE_EXPERIMENTS files across servers"
    )
    parser.add_argument(
        "--hostnames", nargs="+", required=True, help="Hostnames of servers to sync"
    )
    parser.add_argument(
        "--upload", action="store_true", help="Upload merged file back to servers"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    run_sync_and_merge(hostnames=args.hostnames, upload=args.upload)


if __name__ == "__main__":
    main()

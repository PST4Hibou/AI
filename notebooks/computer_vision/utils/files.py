import os
import re


def list_files(directory: str, extensions: list, include_root_directory=False, recursive=False):
    matched_files = []

    # Ensure extensions is a tuple for endswith
    ext_tuple = tuple(extensions) if isinstance(extensions, list) else extensions

    if recursive:
        for root, _, files in os.walk(directory):
            for f in files:
                if f.endswith(ext_tuple):
                    matched_files.append(os.path.join(root, f) if include_root_directory else f)
    else:
        for f in os.listdir(directory):
            if f.endswith(ext_tuple) and os.path.isfile(os.path.join(directory, f)):
                matched_files.append(os.path.join(directory, f) if include_root_directory else f)

    return matched_files


def numeric_key(name):
    """Extract the first number from a filename for sorting."""
    nums = re.findall(r'\d+', name)
    return int(nums[0]) if nums else float('inf')


def sort_files_by_number(files: list):
    """
    Sort a list of filenames by the first number found in each name.

    Args:
        files (list): List of filenames (strings)

    Returns:
        list: Sorted list of filenames
    """
    return sorted(files, key=numeric_key)

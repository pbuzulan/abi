import os


def list_files_in_directory(directory: str):
    """
    List all files in a directory with their absolute paths.

    :param directory: The directory to list files from.
    :return: A list of absolute file paths.
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if '.csv' in filename:
                files.append(os.path.abspath(os.path.join(dirpath, filename)))
    return files

import os


def ensure_exists(dir_path):
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass
    else:
        print(f"Created directory {dir_path}!")

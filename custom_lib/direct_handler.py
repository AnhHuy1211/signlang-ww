import os


def get_file_list(path):
    return os.listdir(path)


def make_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def join_path(paths: list):
    return os.path.join(*paths)

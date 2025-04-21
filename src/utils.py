# -*- coding: utf-8 -*-

from distutils.util import strtobool

import pandas as pd



def load_csv(path):
    if not path.is_file():
        raise RuntimeError(f'Failed to load {path} because it is not a file!')

    for encoding in ['utf-8', 'ISO-8859-1']:
        try:
            df = pd.read_csv(path, encoding = encoding)
            return df
        except Exception:
            pass

    raise RuntimeError(f'Failed to load {path} due to unknown encoding!')



def user_query_yes_no(question):
    print(f'{question} [y/n]')
    while True:
        try:
            return strtobool(input().lower())
        except ValueError:
            print("Please respond with 'y' or 'n'.")



def iter_files(dir_path):
    for file_path in dir_path.glob("**/*"):
        if file_path.is_file():
            yield file_path
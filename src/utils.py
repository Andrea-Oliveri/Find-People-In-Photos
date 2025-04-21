# -*- coding: utf-8 -*-

from distutils.util import strtobool

import pandas as pd



def load_tsv(tsv_path):
    if not tsv_path.is_file(tsv_path):
        raise RuntimeError(f'Failed to load {tsv_path} because it is not a file!')

    for encoding in ['utf-8', 'ISO-8859-1']:
        try:
            df = pd.read_table(tsv_path, encoding = encoding)
            return df
        except Exception:
            pass

    raise RuntimeError(f'Failed to load {tsv_path} due to unknown encoding!')



def user_query_yes_no(question):
    print(f'{question} [y/n]')
    while True:
        try:
            return strtobool(input().lower())
        except ValueError:
            print("Please respond with 'y' or 'n'.")
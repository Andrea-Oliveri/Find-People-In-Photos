# -*- coding: utf-8 -*-

import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd



def dir_children_iter(topdir):
    for root, dirs, files in os.walk(topdir):
        for name in files:
            yield os.path.join(root, name)



def count_extensions_dir_children(topdir):

    counter = Counter(_get_extension(p) for p in dir_children_iter(topdir))
    counter = dict(counter.most_common())

    return sum(counter.values()), counter



def plot_extensions_barchart(extension_counts, output_path = 'extensions.png'):    
    plt.bar(extension_counts.keys(), extension_counts.values())
    plt.ylabel('Count')
    plt.yscale('log')
    plt.xticks(rotation = 90)
    plt.title(f'Number of occurrences of file extensions.\n{sum(extension_counts.values())} files present in total.')
    plt.tight_layout()
    plt.savefig(output_path, dpi = 600)



def load_tsv(tsv_path):
    if not os.path.isfile(tsv_path):
        raise RuntimeError(f'Failed to load {tsv_path} because it is not a file!')

    for encoding in ['utf-8', 'ISO-8859-1']:
        try:
            df = pd.read_table(tsv_path, encoding=encoding)
            return df
        except Exception:
            pass

    raise RuntimeError(f'Failed to load {tsv_path} due to unknown encoding!')



def _get_extension(path): 
    return os.path.splitext(path)[-1].lower()



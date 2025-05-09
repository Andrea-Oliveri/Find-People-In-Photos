# -*- coding: utf-8 -*-

# Constants is imported first so that it sets up the environment variables.
import constants

import pathlib
import argparse
import shutil

import numpy as np
import hdbscan
from tqdm import tqdm




def main():
    args = parse_args()

    cropped_faces_dir    = args.work_dir / constants.CROPPED_FACES_DIRNAME
    embeddings_file_path = args.work_dir / constants.EMBEDDINGS_FILENAME
    clustered_faces_dir  = args.work_dir / constants.CLUSTERED_FACES_DIRNAME
    hdbscan_cache_dir    = args.work_dir / constants.HDBSCAN_CACHE_DIRNAME

    print('Loading embeddings...')
    embeddings = np.load(embeddings_file_path)
    
    print('Clustering...')
    hdbscan_kwargs = {'min_samples'              : args.min_samples,
                      'min_cluster_size'         : args.min_cluster_size,
                      'cluster_selection_epsilon': args.cluster_selection_epsilon,
                      'metric'                   : args.metric,
                      'cluster_selection_method' : args.cluster_selection_method,
                      'memory'                   : str(hdbscan_cache_dir)}
    if not args.store_cache:
        hdbscan_kwargs.pop('memory')
    
    clusterer = hdbscan.HDBSCAN(**hdbscan_kwargs)
    labels = clusterer.fit_predict(embeddings)

    print(f'{np.unique(labels).size - 1} clusters found.')

    print('Grouping clustered faces into separate directories...')
    group_faces_images(cropped_faces_dir, clustered_faces_dir, labels)

    print(f'Process completed.')



def parse_args():
    parser = argparse.ArgumentParser(description = "This script will use HDBSCAN to cluster face embeddings and group face images into subfolders according to the results.",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-w', '--work_dir',
                        required = True,
                        type = pathlib.Path,
                        help = 'Output directory from step extract_faces. It contains the extracted faces and the CSV.')

    parser.add_argument('-n', '--min_samples', 
                        type = int,
                        default = 1,
                        help = 'The number of samples in a neighbourhood for a point to be considered a core point by HDBSCAN.')

    parser.add_argument('-m', '--min_cluster_size', 
                        type = int,
                        default = 20,
                        help = 'The minimum size of clusters used by HDBSCAN.')
    
    parser.add_argument('-e', '--cluster_selection_epsilon', 
                        type = float,
                        default = 0.0,
                        help = 'Distance threshold to merge clusters used by HDBSCAN.')
    
    parser.add_argument('-d', '--metric',
                        default = 'euclidean',
                        help = 'The distance metric used by HDBSCAN.')

    parser.add_argument('-s', '--cluster_selection_method',
                        default = 'eom',
                        help = 'The method used by HDBSCAN to select clusters from the condensed tree. Can be "eom" or "leaf".')
    
    parser.add_argument('-c', '--store_cache',
                        action = argparse.BooleanOptionalAction,
                        default = True,
                        help = 'Whether or not to store the cache for the HDBSCAN algorithm. This may help speed up consecutive runs.')
    
    args = parser.parse_args()

    return args



def group_faces_images(cropped_faces_dir, clustered_faces_dir, labels):
    
    # Delete folder if already existing and create subdirectory structure.
    if clustered_faces_dir.is_dir():
        shutil.rmtree(clustered_faces_dir)
    clustered_faces_dir.mkdir()
    cluster_dir_paths = {label: clustered_faces_dir / str(label) for label in np.unique(labels)}
    
    for directory in cluster_dir_paths.values():
       directory.mkdir()
       
    # Copy cropped faces.
    for image_id, label in enumerate(tqdm(labels, ascii = True, desc = 'Files copied')):
        image_filename = f"{image_id}.png"
        shutil.copy(cropped_faces_dir / image_filename, cluster_dir_paths[label] / image_filename)



if __name__ == '__main__':
    main()
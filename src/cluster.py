# -*- coding: utf-8 -*-

import os
import argparse
import shutil

import numpy as np
import hdbscan

from tqdm import tqdm





def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok = True)

    print('Loading embeddings...')
    embeddings = np.load(args.input_embeddings)
    
    print('Clustering...')
    clusterer_params = {'min_samples'              : args.min_samples, 
                        'min_cluster_size'         : args.min_cluster_size,
                        'metric'                   : args.metric,
                        'cluster_selection_method' : args.cluster_selection_method,
                        'memory'                   : args.cluster_cache_dir}
    if clusterer_params['memory'] is None:
        clusterer_params.pop('memory')

    clusterer = hdbscan.HDBSCAN(**clusterer_params)
    labels = clusterer.fit_predict(embeddings)

    print(f'{np.unique(labels).size - 1} clusters found.')

    print('Grouping clustered faces into separate directories...')
    group_faces_images(args.input_dir, args.output_dir, labels)

    print(f'Process completed.')

    




def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_dir',
                        required = True,
                        type = os.path.abspath,
                        help = 'Input directory in which extracted faces are saved.')

    parser.add_argument('-e', '--input_embeddings',
                        required = False,
                        default = None,
                        type = os.path.abspath,
                        help = 'Input file path that stores array of embeddings for each extracted face.')

    parser.add_argument('-o', '--output_dir',
                        required = False,
                        default = None,
                        type = os.path.abspath,
                        help = 'Output directory in which the extracted faces will be copied and grouped in clusters.')

    parser.add_argument('-c', '--cluster_cache_dir',
                        required = False,
                        default = None,
                        type = os.path.abspath,
                        help = 'Output directory in which the cache of the clustering algorithm will be saved to accelerate consecutive clustering calls. If None (default), no caching is done.')

    parser.add_argument('-m', '--min_cluster_size', 
                        type = int,
                        default = 20,
                        help = 'The minimum size of clusters used by HDBSCAN.')

    parser.add_argument('-n', '--min_samples', 
                        type = int,
                        default = 2,
                        help = 'The number of samples in a neighbourhood for a point to be considered a core point by HDBSCAN.')

    parser.add_argument('-d', '--metric',
                        default = 'euclidean',
                        help = 'Whether or not to parse video frames to detect and extract faces.')

    parser.add_argument('-s', '--cluster_selection_method',
                        default = 'eom',
                        help = 'The method used by HDBSCAN to select clusters from the condensed tree. Can be "eom" or "leaf".')


    args = parser.parse_args()


    # Default for args.input_embeddings
    if args.input_embeddings is None:
        args.input_embeddings = os.path.join(os.path.dirname(args.input_dir), 'faces_embeddings.npy')

    # Default for args.output_dir
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.input_dir), 'Clustered Faces')

    return args



def group_faces_images(input_dir, output_dir, labels):
    
    # Delete folder if already existing and create subdirectory structure.
    shutil.rmtree(output_dir)
    cluster_dir_paths = {label: os.path.join(output_dir, str(label)) for label in np.unique(labels)}
    for directory in cluster_dir_paths.values():
        os.makedirs(directory, exist_ok = False)

    for image_id, label in enumerate(tqdm(labels, ascii = True, desc = 'Files copied')):
        image_filename = f"{image_id}.png"

        shutil.copy(os.path.join(input_dir, image_filename), os.path.join(cluster_dir_paths[label], image_filename))

        

if __name__ == '__main__':
    main()
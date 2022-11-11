# -*- coding: utf-8 -*-

import os
import argparse

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'deepface')))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from deepface import DeepFace

from tqdm import tqdm

import numpy as np
import pandas as pd






def main():
    args = parse_args()

    print('Starting creating face embeddings...')
    make_embeddings(args.input_dir, args.output_file, args.model, args.normalization)



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_dir',
                        required = True,
                        type = os.path.abspath,
                        help = 'Input directory containing images of extracted faces.')

    parser.add_argument('-o', '--output_file',
                        default = None,
                        type = os.path.abspath,
                        help = 'Output file path that will store array of embeddings for each extracted face. If this argument does not end with .npy, this extension is automatically appended.')

    parser.add_argument('-m', '--model', 
                        default = 'Facenet512',
                        help = 'The name of the model used to create the face embeddings. Must be supported by deepface.DeepFace.represent.')

    parser.add_argument('-n', '--normalization', 
                        default = 'Facenet',
                        help = 'The normalization technique used in the pre-processing step of the face images. Must be supported by deepface.commons.functions.normalize_input.')

    args = parser.parse_args()

    # Default for args.output_file
    if args.output_file is None:
        args.output_file = os.path.join(os.path.dirname(args.input_dir), 'faces_embeddings.npy')
        
    return args



def make_embeddings(input_dir, output_file, model, normalization):
    
    tsv_path, = [filename for filename in os.listdir(input_dir) if filename.endswith('.tsv')]
    tsv_path = os.path.join(input_dir, tsv_path)

    face_paths = pd.read_table(tsv_path, encoding = 'ISO-8859-1')['id']
    face_paths = [os.path.join(input_dir, f"{id_}.png") for id_ in face_paths]

    model = DeepFace.build_model(model)

    embeddings = []
    for path in tqdm(face_paths, ascii = True, desc = 'Files processed'):
        e = DeepFace.represent(path, model = model, detector_backend = 'skip', normalization = normalization)
        embeddings.append(e)

    print('Saving embeddings...')
    embeddings = np.stack(embeddings, axis = 0)
    np.save(output_file, embeddings, allow_pickle = False)

    print('Process completed.')
    


if __name__ == '__main__':
    main()
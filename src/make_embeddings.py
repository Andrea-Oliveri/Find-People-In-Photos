# -*- coding: utf-8 -*-

import os
import argparse

from utils import load_tsv

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'deepface')))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from deepface import DeepFace

import numpy as np
import pandas as pd

from tqdm import tqdm





def main():
    args = parse_args()

    print('Starting creating face embeddings...')
    embeddings = make_embeddings(args.input_dir, args.input_faces_tsv_path, args.model, args.normalization)

    print('Saving embeddings...')
    embeddings = np.stack(embeddings, axis = 0)
    np.save(args.output_file, embeddings, allow_pickle = False)

    print('Process completed.')



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_dir',
                        required = True,
                        type = os.path.abspath,
                        help = 'Input directory containing images of extracted faces.')

    parser.add_argument('-e', '--input_faces_tsv_path',
                        required = False,
                        default = None,
                        type = os.path.abspath,
                        help = 'Input file path that stores the (patch_id, original_image_path) pairs in tsv format.')

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

    # Default for args.input_faces_tsv_path
    if args.input_faces_tsv_path is None:
        args.input_faces_tsv_path = os.path.abspath(os.path.join(args.input_dir, '../faces.tsv'))
    
    # Default for args.output_file
    if args.output_file is None:
        args.output_file = os.path.join(os.path.dirname(args.input_dir), 'faces_embeddings.npy')
        
    return args



def make_embeddings(input_dir, faces_tsv_path, model, normalization):
    
    face_paths = load_tsv(faces_tsv_path)['id']
    face_paths = [os.path.join(input_dir, f"{id_}.png") for id_ in face_paths]

    model = DeepFace.build_model(model)

    embeddings = []
    for path in tqdm(face_paths, ascii = True, desc = 'Files processed'):
        e = DeepFace.represent(path, model = model, detector_backend = 'skip', normalization = normalization)
        embeddings.append(e)

    return embeddings
    
    


if __name__ == '__main__':
    main()
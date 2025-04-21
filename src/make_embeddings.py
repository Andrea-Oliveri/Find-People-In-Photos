# -*- coding: utf-8 -*-


# Constants is imported first so that it sets up the environment variables.
import constants

import pathlib
import argparse

import numpy as np
from deepface import DeepFace
from tqdm import tqdm

import utils




def main():
    args = parse_args()

    faces_csv_path       = args.work_dir / constants.FACES_CSV_FILENAME
    cropped_faces_dir    = args.work_dir / constants.CROPPED_FACES_DIRNAME
    embeddings_file_path = args.work_dir / constants.EMBEDDINGS_FILENAME
    
    if embeddings_file_path.is_file():
        embeddings_file_path.unlink()
    
    print('Starting creating face embeddings...')
    embeddings = make_embeddings(cropped_faces_dir, faces_csv_path, args.model, args.normalization)

    print('Saving embeddings...')
    embeddings = np.stack(embeddings, axis = 0)
    np.save(embeddings_file_path, embeddings, allow_pickle = False)

    print('Process completed.')



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--work_dir',
                        required = True,
                        type = pathlib.Path,
                        help = 'Output directory from step extract_faces. It contains the extracted faces and the CSV.')

    parser.add_argument('-m', '--model',
                        default = 'Facenet512',
                        help = 'The name of the model used to create the face embeddings. Must be supported by DeepFace.represent().')

    parser.add_argument('-n', '--normalization',
                        default = 'Facenet',
                        help = 'The normalization technique used in the pre-processing step of the face images. Must be supported by deepface.commons.functions.normalize_input.')

    args = parser.parse_args()
    
    return args



def make_embeddings(cropped_faces_dir, faces_csv_path, model, normalization):
    
    face_paths = utils.load_csv(faces_csv_path)['id']
    face_paths = [cropped_faces_dir / f"{id_}.png" for id_ in face_paths]

    embeddings = []
    for path in tqdm(face_paths, ascii = True, desc = 'Files processed'):
        result, = DeepFace.represent(path, model_name = model, detector_backend = 'skip', normalization = normalization, anti_spoofing = False)
        embeddings.append(result['embedding'])

    return embeddings



if __name__ == '__main__':
    main()
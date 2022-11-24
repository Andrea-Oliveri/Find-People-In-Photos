# -*- coding: utf-8 -*-

import os
import argparse
import shutil

from utils import load_tsv

from tqdm import tqdm





def main():
    args = parse_args()
    
    # Delete folder if already existing and (re-)create it.
    shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok = True)

    print('Loading tsv...')
    df = load_tsv(args.input_faces_tsv_path)
    
    print('Copying...')
    num_to_copy, not_found = copy_images(args.input_cluster_dir, args.output_dir, df)

    print(f'Process completed.')
    if not_found:
        print(f'{len(not_found)} out of {num_to_copy} images were not found:')
        print('\n'.join(not_found))

    




def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_cluster_dir',
                        required = True,
                        type = os.path.abspath,
                        help = 'Input directory containing the extracted faces for the cluster/person.')

    parser.add_argument('-e', '--input_faces_tsv_path',
                        required = False,
                        default = None,
                        type = os.path.abspath,
                        help = 'Input file path that stores the (patch_id, original_image_path) pairs in tsv format.')

    parser.add_argument('-o', '--output_dir',
                        required = True,
                        type = os.path.abspath,
                        help = 'Output directory in which the images from which patches in input_cluster_dir have been cropped will be copied.')
    
    args = parser.parse_args()
    
    # Default for args.input_faces_tsv_path
    if args.input_faces_tsv_path is None:
        args.input_faces_tsv_path = os.path.abspath(os.path.join(args.input_cluster_dir, '../../faces.tsv'))

    return args



def get_unused_destination_path(output_dir, basename):
    # If basename already exists in output_dir, preserve both files.

    filename, extension = os.path.splitext(basename)
    i = 0

    destination_path = os.path.join(output_dir, basename)
    while os.path.isfile(destination_path):
        i += 1

        destination_path = os.path.join(output_dir, f'{filename} ({i}){extension}')

    return destination_path



def copy_images(input_cluster_dir, output_dir, df):
    patch_ids = [int(os.path.splitext(p)[0]) for p in os.listdir(input_cluster_dir)]
    images_to_copy = df[df.id.isin(patch_ids)].image_path.unique()

    images_not_found = []
    for image_path in tqdm(images_to_copy, ascii = True, desc = 'Files copied'):

        if not os.path.isfile(image_path):
            images_not_found.append(image_path)
            continue

        destination_path = get_unused_destination_path(output_dir, os.path.basename(image_path))
        shutil.copy(image_path, destination_path)

    return len(images_to_copy), images_not_found

        

if __name__ == '__main__':
    main()
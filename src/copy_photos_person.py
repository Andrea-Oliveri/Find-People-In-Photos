# -*- coding: utf-8 -*-

# Constants is imported first so that it sets up the environment variables.
import constants

import pathlib
import argparse
import shutil

from tqdm import tqdm

import utils



def main():
    args = parse_args()
    
    faces_csv_path      = args.work_dir / constants.FACES_CSV_FILENAME
    clustered_faces_dir = args.work_dir / constants.CLUSTERED_FACES_DIRNAME
    
    # Delete folder if already existing and (re-)create it.
    if args.output_dir.is_dir():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir()
    
    print('Loading CSV...')
    df = utils.load_csv(faces_csv_path)
    
    print('Copying...')
    num_to_copy, not_found = copy_images(clustered_faces_dir / str(args.label), args.output_dir, df)

    print(f'Process completed.')
    if not_found:
        print(f'{len(not_found)} out of {num_to_copy} images were not found:')
        print('\n'.join(not_found))



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--work_dir',
                        required = True,
                        type = pathlib.Path,
                        help = 'Output directory from step extract_faces. It contains the extracted faces and the CSV.')

    parser.add_argument('-l', '--label',
                        required = True,
                        type = str,
                        help = 'Label of the cluster to extract original images of.)directory from step extract_faces. It contains the extracted faces and the CSV.')

    parser.add_argument('-o', '--output_dir',
                        required = True,
                        type = pathlib.Path,
                        help = 'Output directory in which to copy original images containing the selected cropped faces.')

    args = parser.parse_args()
    
    return args



def get_unused_destination_path(output_dir, basename):
    # If basename already exists in output_dir, preserve both files.

    destination_path = output_dir / basename
    stem = destination_path.stem
    i = 0

    while destination_path.is_file():
        i += 1
        destination_path = destination_path.with_stem(f'{stem} ({i})')

    return destination_path



def copy_images(selected_dir, output_dir, df):
    patch_ids = [int(p.stem) for p in selected_dir.iterdir()]
    images_to_copy = [pathlib.Path(p) for p in df[df.id.isin(patch_ids)].image_path.unique()]

    images_not_found = []
    for image_path in tqdm(images_to_copy, ascii = True, desc = 'Files copied'):

        if not image_path.is_file():
            images_not_found.append(image_path)
            continue

        destination_path = get_unused_destination_path(output_dir, image_path.name)
        shutil.copy(image_path, destination_path)

    return len(images_to_copy), images_not_found



if __name__ == '__main__':
    main()
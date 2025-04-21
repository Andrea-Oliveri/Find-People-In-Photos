# -*- coding: utf-8 -*-

# Constants is imported first so that it sets up the environment variables.
import constants

import pathlib
import argparse
import csv
import contextlib
from shutil import rmtree
from collections import Counter

import matplotlib.pyplot as plt
from deepface import DeepFace
from tqdm import tqdm

import image_io
import utils




def main():
    args = _parse_args()

    extension_hist_path = args.output_dir / constants.EXTENSION_HIST_FILENAME
    faces_csv_path      = args.output_dir / constants.FACES_CSV_FILENAME
    cropped_faces_dir   = args.output_dir / constants.CROPPED_FACES_DIRNAME
    
    if args.output_dir.is_dir():
        if not utils.user_query_yes_no(f'Folder at "{args.output_dir}" already exists. Do you want to delete its content?'):
            return
        rmtree(args.output_dir)
        
    cropped_faces_dir.mkdir(parents = True)
    
    print('Counting number of files in directory...')
    n_files, extension_counts = _count_extensions_dir_children(args.input_dir)

    print('Generating extensions histogram...')
    _plot_extensions_barchart(extension_counts, extension_hist_path)

    print('Starting face detection and extraction...')
    n_faces = detect_and_extract_faces(args.input_dir,
                                       faces_csv_path,
                                       cropped_faces_dir,
                                       n_files,
                                       args.read_videos,
                                       args.secs_between_frames,
                                       args.detector_name,
                                       args.min_confidence,
                                       args.align_output_faces)

    print(f'Process completed. Extracted {n_faces} faces from {n_files} files.')



def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_dir',
                        required = True,
                        type = pathlib.Path,
                        help = 'Input directory containing images (and optionally videos) to extract faces from.')

    parser.add_argument('-o', '--output_dir',
                        required = True,
                        type = pathlib.Path,
                        help = 'Output directory in which extracted faces and additional metadata will be saved.')

    parser.add_argument('-v', '--read_videos', 
                        action = argparse.BooleanOptionalAction,
                        default = False,
                        help = 'Whether or not to parse video frames to detect and extract faces.')

    parser.add_argument('-s', '--seconds', 
                        type = float,
                        default = 1,
                        dest = 'secs_between_frames',
                        help = 'How many seconds of video to skip between two consecutively parsed frames.')

    parser.add_argument('-d', '--detector', 
                        default = 'retinaface',
                        dest = 'detector_name',
                        help = 'The name of the detector used to detect and extract faces from images. Must be supported by Deepface.extract_face().')

    parser.add_argument('-c', '--min_confidence', 
                        type = float,
                        default = 0.9,
                        help = 'Minimum confidence score to accept a face detection as valid.')
    
    parser.add_argument('-a', '--align_output_faces', 
                        action = argparse.BooleanOptionalAction,
                        default = False,
                        help = 'Whether or not to align the output faces. This can improve performance of subsequent steps but has a large overhead.')


    args = parser.parse_args()

    return args



def _count_extensions_dir_children(topdir):
    counter = Counter(p.suffix.lower() for p in utils.iter_files(topdir))
    counter = dict(counter.most_common())
    return sum(counter.values()), counter



def _plot_extensions_barchart(extension_counts, output_path = None):
    if output_path is None:
        return
    
    plt.bar(extension_counts.keys(), extension_counts.values())
    plt.ylabel('Count')
    plt.yscale('log')
    plt.xticks(rotation = 90)
    plt.title(f'Number of occurrences of file extensions.\n{sum(extension_counts.values())} files present in total.')
    plt.tight_layout()
    plt.savefig(output_path, dpi = 600)


@contextlib.contextmanager
def _get_csv_writer(path):
    with open(path, 'w', newline = '') as file:
        csv_writer = csv.writer(file, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_ALL)
        yield csv_writer



def detect_and_extract_faces(input_dir, faces_csv_path, output_dir, n_files,
                             read_videos = False, secs_between_frames = 1,
                             detector_name = 'retinaface', min_confidence = 0.9,
                             align_output_faces = True):

    with _get_csv_writer(faces_csv_path) as csv_writer:
        csv_writer.writerow(['id', 'image_path'])
        
        patch_id = 0
    
        with tqdm(total = n_files, ascii = True, desc = 'Files processed') as pbar:
            for file_idx, file_path, image in image_io.photos_video_frames_iterator(input_dir, read_videos, secs_between_frames):
                results = DeepFace.extract_faces(image,
                                                 detector_name,
                                                 align = align_output_faces,
                                                 enforce_detection = False,
                                                 color_face = 'bgr',
                                                 normalize_face = False,
                                                 anti_spoofing = False)

                for result in results:
                    if result['confidence'] < min_confidence:
                        continue

                    image_io.save_image(result['face'], output_dir / f"{patch_id}.png")
                    csv_writer.writerow([patch_id, file_path])
                    patch_id += 1
            
                pbar.n = file_idx + 1
                pbar.refresh()

            pbar.n = n_files
            pbar.refresh()

    return patch_id



if __name__ == '__main__':
    main()
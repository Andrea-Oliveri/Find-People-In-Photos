# -*- coding: utf-8 -*-

import pathlib
import argparse
import contextlib
import csv
from shutil import rmtree
from collections import Counter
from os import devnull

import matplotlib.pyplot as plt
from deepface.detectors import FaceDetector
from tqdm import tqdm

import image_io
import utils
import constants




def main():
    args = _parse_args()

    extension_chart_path = args.output_dir / constants.EXTENSION_HIST_FILENAME
    faces_csv_path       = args.output_dir / constants.FACES_TSV_FILENAME
    cropped_faces_dir    = args.output_dir / constants.CROPPED_FACES_DIRNAME
    
    if args.output_dir.is_dir():
        if not utils.user_query_yes_no(f'Folder at "{args.output_dir}" already exists. Do you want to delete its content?'):
            return
    
    rmtree(args.output_dir)
    cropped_faces_dir.mkdir(parents = True)
    
    print('Counting number of files in directory...')
    n_files, extension_counts = _count_extensions_dir_children(args.input_dir)

    print('Generating extensions histogram...')
    _plot_extensions_barchart(extension_counts, extension_chart_path)

    print('Starting face detection and extraction...')
    n_faces = detect_and_extract_faces(args.input_dir,
                                       faces_csv_path,
                                       cropped_faces_dir,
                                       n_files,
                                       args.read_videos,
                                       args.secs_between_frames,
                                       args.detector_name,
                                       args.min_confidence)

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
                        help = 'The name of the detector used to detect and extract faces from images. Must be supported by deepface.detectors.FaceDetector.')

    parser.add_argument('-c', '--min_confidence', 
                        type = float,
                        default = 0.9,
                        help = 'Minimum confidence score to accept a face detection as valid.')

    args = parser.parse_args()

    return args



def _count_extensions_dir_children(topdir):
    counter = Counter(p.suffix.lower() for p in topdir.glob("**/*"))
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



class DeepfaceDetectorWrapper:

    def __init__(self, detector_name):
        self.detector_name = detector_name
        self.detector = FaceDetector.build_model(detector_name)

    def detect_faces(self, image):
        with open(devnull, 'w') as devnull_stream:
            with contextlib.redirect_stdout(devnull_stream), contextlib.redirect_stderr(devnull_stream):
                faces_regions_scores = FaceDetector.detect_faces(self.detector, self.detector_name, image, align = False)

        return faces_regions_scores



def _get_csv_writer(path):
    with open(path, 'w', newline = '') as file:
        csv_writer = csv.writer(file, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_ALL)
        yield csv_writer



def detect_and_extract_faces(input_dir, faces_csv_path, output_dir, n_files,
                             read_videos = False, secs_between_frames = 1,
                             detector_name = 'retinaface', min_confidence = 0.9):

    with _get_csv_writer(faces_csv_path) as csv_writer:
        csv_writer.writerow(['id', 'image_path'])
        
        detector = DeepfaceDetectorWrapper(detector_name)
        patch_id = 0
    
        with tqdm(total = n_files, ascii = True, desc = 'Files processed') as pbar:
            for file_idx, file_path, image in image_io.photos_video_frames_iterator(input_dir, read_videos, secs_between_frames):
                faces_regions_scores = detector.detect_faces(image)

                for face, _, confidence in faces_regions_scores:
                    if confidence < min_confidence:
                        continue

                    image_io.imsave(face, output_dir / f"{patch_id}.png")
                    csv_writer.writerow([patch_id, file_path])
                    patch_id += 1
            
                pbar.n = file_idx + 1
                pbar.refresh()

            pbar.n = n_files
            pbar.refresh()

    return patch_id 



if __name__ == '__main__':
    main()
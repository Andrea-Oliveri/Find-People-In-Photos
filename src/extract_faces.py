# -*- coding: utf-8 -*-

import os
import argparse
import contextlib

import image_io
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from deepface.detectors import FaceDetector

from tqdm import tqdm





def main():
    args = parse_args()

    extension_chart_path = os.path.join(args.output_dir, 'extensions.png')
    cropped_faces_dir    = os.path.join(args.output_dir, 'Extracted Faces')
    os.makedirs(args.output_dir  , exist_ok = True)
    os.makedirs(cropped_faces_dir, exist_ok = True)

    print('Counting number of files in directory...')
    n_files, extension_counts = utils.count_extensions_dir_children(args.input_dir)

    print('Generating extensions histogram...')
    utils.plot_extensions_barchart(extension_counts, extension_chart_path)

    print('Starting face detection and extraction...')
    detect_and_extract_faces(args.input_dir, 
                             cropped_faces_dir,
                             n_files, 
                             args.read_videos,
                             args.secs_between_frames,
                             args.detector_name)



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_dir',
                        required = True,
                        type = os.path.abspath,
                        help = 'Input directory containing images (and optionally videos) to extract faces from.')

    parser.add_argument('-o', '--output_dir',
                        required = True,
                        type = os.path.abspath,
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

    args = parser.parse_args()

    return args




class DeepfaceDetectorWrapper:

    def __init__(self, detector_name):
        self.detector_name = detector_name
        self.detector = FaceDetector.build_model(detector_name)

    def detect_faces(self, image):
        with open(os.devnull, 'w') as devnull_stream:
            with contextlib.redirect_stdout(devnull_stream), contextlib.redirect_stderr(devnull_stream):
                faces_regions = FaceDetector.detect_faces(self.detector, self.detector_name, image, align = False)

        return faces_regions



class TSVWriter:

    def __init__(self, path):
        self.path = path

        if os.path.isfile(self.path):
            os.remove(self.path)

    def write_line(self, *args):
        with open(self.path, 'a') as file:
            file.write('\t'.join(map(str, args)))
            file.write('\n')

        return self
    



            
def detect_and_extract_faces(input_dir, output_dir, n_files, 
                             read_videos = False, secs_between_frames = 1, 
                             detector_name = 'retinaface'):    

    tsv_path = os.path.join(output_dir, "faces.tsv")
    tsv_writer = TSVWriter(tsv_path).write_line('id', 'image_path')

    detector = DeepfaceDetectorWrapper(detector_name)

    patch_id = 0
    
    with tqdm(total = n_files, ascii = True, desc = 'Files processed') as pbar:
        for file_idx, file_path, image in image_io.photos_video_frames_iterator(input_dir, read_videos, secs_between_frames):
            faces_regions = detector.detect_faces(image)

            for face, _ in faces_regions:
                image_io.imsave(face, os.path.join(output_dir, f"{patch_id}.png"))
                tsv_writer.write_line(patch_id, file_path)
                patch_id += 1
            
            pbar.n = file_idx + 1
            pbar.refresh()

        pbar.n = n_files
        pbar.refresh()

    print(f'Process completed. Extracted {patch_id} faces from {n_files} files.')

        

if __name__ == '__main__':
    main()



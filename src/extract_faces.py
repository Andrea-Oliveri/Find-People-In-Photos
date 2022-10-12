# -*- coding: utf-8 -*-

import os
import argparse
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from mtcnn.mtcnn import MTCNN




def main():
    args = parse_args()

    extension_chart_path = os.path.join(args.output_dir, 'extensions.png')
    cropped_faces_dir    = os.path.join(args.output_dir, 'Extracted Faces')
    os.makedirs(args.output_dir  , exist_ok = True)
    os.makedirs(cropped_faces_dir, exist_ok = True)

    print('Counting number of files in directory...')
    n_files, extension_counts = count_extensions_dir_children(args.input_dir)

    print('Generating extensions histogram...')
    plot_extensions_barchart(extension_counts, extension_chart_path)

    print('Starting face detection and extraction...')
    detect_and_extract_faces(args.input_dir, 
                             cropped_faces_dir,
                             n_files, 
                             args.update_every_n_files,
                             args.detection_confidence_thr,
                             args.crop_padding)





def parse_args():
    return argparse.Namespace(input_dir = r"",
                              output_dir = '../New Temporary Data',
                              update_every_n_files = 10,
                              detection_confidence_thr = 0.92,
                              crop_padding = 0.1)







def get_extension(path):
    return os.path.splitext(path)[-1].lower()


def count_extensions_dir_children(topdir):

    counter = Counter(get_extension(p) for p in dir_children_iter(topdir))
    counter = dict(counter.most_common())

    return sum(counter.values()), counter



def dir_children_iter(topdir):
    for root, dirs, files in os.walk(topdir):
        for name in files:
            yield os.path.join(root, name)


def plot_extensions_barchart(extension_counts, output_path = 'extensions.png'):    
    plt.bar(extension_counts.keys(), extension_counts.values())
    plt.ylabel('Count')
    plt.yscale('log')
    plt.xticks(rotation = 90)
    plt.title(f'Number of occurrences of file extensions.\n{sum(extension_counts.values())} files present in total.')
    plt.tight_layout()
    plt.savefig(output_path, dpi = 600)




def photos_video_frames_iterator(input_dir, ignore_videos = True):    
    
    IMAGE_EXTENSIONS = ('.bmp', '.dib' , '.jpeg', '.jpg' , '.jpe'  , '.jp2' , 
                        '.png', '.webp', '.pbm,', '.pgm,', '.ppm'  , '.pxm,', 
                        '.pnm', '.pfm' , '.sr,' , '.ras' , '.tiff,', '.tif' , 
                        '.exr', '.hdr,', '.pic')
    
    for file_idx, file_path in enumerate(dir_children_iter(input_dir)):

        if get_extension(file_path) in IMAGE_EXTENSIONS:
            
            # As opencv does not support unicode characters in image path,
            # We must read image bytes with numpy and then decode image with opencv.                
            image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if image is not None:
                yield file_idx, file_path, image
                
        elif not ignore_videos:
            cap = cv2.VideoCapture(file_path)

            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                yield file_idx, file_path, frame
                
            cap.release()
            

def show_results(image, faces):
    color = (0, 0, 255)
    
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness = 2)
        cv2.putText(image, f"{face['confidence']:.5f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.7, color = color, thickness = 2)
        
        for _, center in face['keypoints'].items():
            cv2.circle(image, center, radius = 2, color = color, thickness = -1)
    
    cv2.namedWindow("Results", cv2.WINDOW_AUTOSIZE) 
    cv2.imshow('Results', image)
    cv2.waitKey(1)


def crop_face(image, face_box, padding):
    x, y, width, height = face_box
        
    padding_x = int(padding * width)
    padding_y = int(padding * height)
        
    start_x = x - padding_x
    end_x   = x + padding_x + width
    start_y = y - padding_y
    end_y   = y + padding_y + height
        
    image_height, image_width, _ = image.shape
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    end_x   = min(end_x  , image_width)
    end_y   = min(end_y  , image_height)
                
    return image[start_y:end_y, start_x:end_x]


def bgr_rgb_convert(image):
    return image[..., ::-1]

            
def detect_and_extract_faces(input_dir, output_dir, n_files, update_every_n_files, confidence_thr = 0.92, crop_padding = 0.1):    

    tsv_path = os.path.join(output_dir, "faces.tsv")
    with open(tsv_path, 'w') as file:
        file.write("id\timage_path\n")
        
    
    detector = MTCNN()
    patch_id = 0
    
    for file_idx, file_path, image in photos_video_frames_iterator(input_dir):
        faces = detector.detect_faces(bgr_rgb_convert(image))
        
        for face in faces:
            if face['confidence'] > confidence_thr:
                patch = crop_face(image, face['box'], crop_padding)

                cv2.imwrite(os.path.join(output_dir, f"{patch_id}.png"), patch)
                with open(tsv_path, 'a') as file:
                    file.write(f"{patch_id}\t{file_path}\n")
                
                patch_id += 1
            
        
        if not file_idx % update_every_n_files:
            print(f'Processed {file_idx} files out of {n_files}', end = '\r')
            #show_results(image, faces)

    print(f'Processed all {n_files} files. Extracted {patch_id} faces.')

        
        

if __name__ == '__main__':
    main()



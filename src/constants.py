# -*- coding: utf-8 -*-

import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"   # Silence OpenCV's cv2.VideoCapture errors from stdout.
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8" # Silence FFMPEG logging.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Set Tensorflow logging to ERROR.


EXTENSION_HIST_FILENAME = 'Extensions.png'
FACES_CSV_FILENAME      = 'Faces.csv'
CROPPED_FACES_DIRNAME   = 'Extracted Faces'
EMBEDDINGS_FILENAME     = 'Embeddings.npy'
CLUSTERED_FACES_DIRNAME = 'Clusters'
HDBSCAN_CACHE_DIRNAME   = 'HDBSCAN Cache'
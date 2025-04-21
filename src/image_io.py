# -*- coding: utf-8 -*-

# Constants is imported first so that it sets up the environment variables.
import constants as _

from enum import IntEnum

import cv2
import pillow_heif
from PIL import Image, ImageOps
from numpy import asarray

import utils


pillow_heif.register_heif_opener()
pillow_heif.register_avif_opener()
_FileType = IntEnum('_FileType', ['IMAGE', 'VIDEO', 'UNKNOWN'])



def _load_image(path):
    image = None

    # Try reading with PIL.
    try:
        image = Image.open(path)
        image = ImageOps.exif_transpose(image)
        image = asarray(image.convert('BGR'))
    except Exception:
        image = None
    
    # OpenCV returns None if cv2.imread could not read image.
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    return image



def _iter_video(path, secs_between_frames):
    # Regardless of format, try reading it as a video. If OpenCV fails, it will write to
    # stderr but no exception is launched in Python. Therefore program will keep running
    # without interruptions, just cap.isOpened() will return False. 
    # CAP_FFMPEG is needed to allow auto-rotating frames from videos taken by iOS and Android devices.
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)

    fps = cap.get(cv2.CAP_PROP_FPS)    
    fps = 24 if fps == 0 else fps
    frames_to_skip = int(secs_between_frames * fps)

    while cap.isOpened():
        for _ in range(frames_to_skip):
            cap.grab()

        ret, frame = cap.read()

        if not ret:
            break
        
        yield frame
    
    cap.release()



def _file_is_image_or_video(path):
    try:
        Image.open(path).verify()
        return _FileType.IMAGE

    except Exception:
        pass

    cap = cv2.VideoCapture(path)

    if cap.isOpened():
        # cv2.VideoCapture is capable of reading images too, so need to check if it loaded the file, 
        # and if the file contains multiple frames for it to be a video.
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 1:
            return _FileType.VIDEO

        return _FileType.IMAGE    

    return _FileType.UNKNOWN



def photos_video_frames_iterator(input_dir, read_videos, secs_between_frames):
    for idx, path in enumerate(utils.iter_files(input_dir)):
        file_type = _file_is_image_or_video(path)
        
        if file_type == _FileType.IMAGE:
            image = _load_image(path)
            if image is not None:
                yield idx, path, image
            
        elif file_type == _FileType.VIDEO and read_videos:
            for image in _iter_video(path, secs_between_frames):
                yield idx, path, image



def save_image(image, path):
    cv2.imwrite(path, image)
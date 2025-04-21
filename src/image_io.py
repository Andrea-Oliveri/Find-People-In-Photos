# -*- coding: utf-8 -*-

# Constants is imported first so that it sets up the environment variables.
import constants

from numpy import asarray
from cv2 import VideoCapture, CAP_PROP_FPS, CAP_FFMPEG
from PIL import Image, ImageOps
import pillow_heif
pillow_heif.register_heif_opener()
pillow_heif.register_avif_opener()



def load_image(path):
    image_load_success = True
    image = None

    try:
        image = Image.open(path)
        image = ImageOps.exif_transpose(image)
        image = asarray(image.convert('BGR'))
    except Exception:
        image_load_success = False

    return image_load_success, image



def iter_video(path, secs_between_frames):
    # Regardless of format, try reading it as a video. If OpenCV fails, it will write to
    # stderr but no exception is launched in Python. Therefore program will keep running
    # without interruptions, just cap.isOpened() will return False. 
    # CAP_FFMPEG is needed to allow auto-rotating frames from videos taken by iOS and Android devices.
    cap = VideoCapture(path, CAP_FFMPEG)

    fps = cap.get(CAP_PROP_FPS)    
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
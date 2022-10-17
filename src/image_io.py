# -*- coding: utf-8 -*-


from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()
pillow_heif.register_avif_opener()

from numpy import asarray


# This is the only way to silence OpenCV's cv2.VideoCapture errors...
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

from cv2 import VideoCapture, CAP_PROP_FPS, CAP_FFMPEG, imwrite


from utils import dir_children_iter





def photos_video_frames_iterator(input_dir, read_videos = False, secs_between_frames = 1):    
    """
    
    image::[np.array]
        Image or video frame in BGR format.
    """


    for file_idx, file_path in enumerate(dir_children_iter(input_dir)):

        image_load_success, image = _load_image(file_path)
        
        if image_load_success:
            yield file_idx, file_path, image
        
        elif read_videos:
            for frame in _iter_video(file_path, secs_between_frames):
                yield file_idx, file_path, frame



def imsave(image, path):
    imwrite(path, image)




def _load_image(path):
    image_load_success = True
    image = None

    try:
        image = asarray(Image.open(path).convert("RGB"))
        image = image[..., ::-1]
    except Exception:
        image_load_success = False

    return image_load_success, image




def _iter_video(path, secs_between_frames):
    # Regardless of format, try reading it as a video. If OpenCV fails, it will write to
    # stderr but no exception is launched in Python. Therefore program will keep running
    # without interruptions, and cap.isOpened() will return False. 
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
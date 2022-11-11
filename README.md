### Install:

conda create -n faces python=3.10 matplotlib pandas
conda activate faces
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
conda install -c nvidia cuda-nvcc=11.8.89
pip install tensorflow==2.10 opencv-python==4.5 Pillow pillow-heif
pip install mtcnn hdbscan


conda create -n deepface python=3.10 matplotlib pandas
conda activate deepface
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install tensorflow==2.10 opencv-contrib-python==4.5.5.64 Pillow pillow-heif 
pip install cmake
pip install deepface hdbscan 


# Optionally
pip install mediapipe dlib cmake

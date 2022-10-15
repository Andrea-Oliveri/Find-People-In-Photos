### Install:

conda create -n faces python=3.10 matplotlib pandas
conda activate faces
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install tensorflow==2.10 opencv-python
pip install mtcnn hdbscan
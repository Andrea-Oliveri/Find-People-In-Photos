The aim of this package is to provide a pipeline to find and copy in a separate folders all images (and optionally videos) containing a specific person from a large dataset.

The pipeline which is implemented by this package is as follows:

|    | Step                                                            | Script                                             |
| -- | --------------------------------------------------------------- | -------------------------------------------------- |
| 1. | Face detection and extraction from whole dataset                | [extract_faces.py](src/extract_faces.py)           |
| 2. | Computation of face embeddings                                  | [make_embeddings.py](src/make_embeddings.py)       |
| 3. | Clustering of face embeddings                                   | [cluster.py](src/cluster.py)                       |
| 4. | (Optionally) Manually review and add/remove faces from clusters | No script. Use output of previous step.            |
| 5. | Copy images containing the person's face in a new location      | [copy_photos_person.py](src/copy_photos_person.py) |


### Installation:

To create the environment to run all components of this package, runt he following commands:

´´´
conda create -n deepface python=3.10 matplotlib pandas
conda activate deepface
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install tensorflow==2.10 opencv-contrib-python==4.5.5.64 Pillow pillow-heif 
pip install cmake
pip install deepface hdbscan 
´´´

Optionally, you can also install the following packages to have access to additional models for face detection and embeddings:

´´´
pip install mediapipe dlib cmake
´´´
# Find-People-In-Photos

## Project Overview

The aim of this project is to provide a pipeline to find and copy in a separate folders all images (and optionally videos) containing the face of a specific person from a large dataset.

This is achieved via a face detection and unsupervised clustering pipeline, described below:

|    | Step                                                            | Script                                             |
| -- | --------------------------------------------------------------- | -------------------------------------------------- |
| 1. | Face detection and extraction from whole dataset                | [extract_faces.py](src/extract_faces.py)           |
| 2. | Computation of face embeddings                                  | [make_embeddings.py](src/make_embeddings.py)       |
| 3. | Clustering of face embeddings                                   | [cluster.py](src/cluster.py)                       |
| 4. | (Optionally) Manually review and add/remove faces from clusters | No script. Use output of previous step.            |
| 5. | Copy images containing the person's face in a new location      | [copy_photos_person.py](src/copy_photos_person.py) |


## Repository Structure

This directory contains the following files and directories:

* [**src.py**](src): Directory containing all Python scripts.
* [**.gitignore**](.gitignore): The gitignore file of this repository.
* [**README.md**](README.md): The Readme file you are currently reading.


## Installation

To create the environment to run all components of this package, run the following commands:

```
conda create -n people_photo_search python=3.10 matplotlib pandas numpy<2
conda activate people_photo_search
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install tensorflow==2.10 opencv-contrib-python==4.5.5.64 opencv-python==4.10.0.84 Pillow pillow-heif deepface==0.0.93 hdbscan
```

Optionally, you can also install the following packages to have access to additional models for face detection and embeddings:

```
pip install cmake mediapipe dlib
```


## Example usage

Each step in the pipeline has its own arguments which can optionally be tuned to achieve the desired outcome. The default values are what worked reasonably well and fast in experiments on my data (more about this in [Performance](#performance) section).

You can read about all a the arguments by using `-h` when calling the script, for example:
```
python src/extract_faces -h
```

Nonetheless, here is a quick walkthrough on how to run the pipeline providing only required arguments:

1) The first step you would do is to extract faces from your photos and videos. The simplest way to do this is as below:
```
python src/extract_faces.py -i "Your image & video folder" -w "A work folder where each script will write its results"
```

2) You then need to create the embeddings for each extracted face. This can be done as such: 
```
python src/make_embeddings.py -w "The same work folder used in the previous step"
```

3) After this step you will need to cluster your faces to group them by person. This is the step where tuning your parameters can have a dramatic impact on your precision and recall. Nonetheless, the step can be run with default parameters as such:
```
python src/cluster.py -w "The same work folder used in the previous step"
```

4) If accuracy is important to you, in this step I strongly recommend taking a second to review the output of the clustering. It will have grouped faces based on their embeddings' similarity, but you may still get multiple folders for the same person, some faces which are not in a group and occasionally multiple people in the same group. I advise taking a second to fix these manually. You can freely change the names of the folders (not of the photos though), move photos around and even delete folders outright and the next step will still work.

5) Finally, to extract all original images with the face of a person of interest, you can run this:
```
python src/copy_photos_person.py -w "The same work folder used in the previous step" -l "Name of the subfolder containing the cluster" -o "Folder where to save the copied images"
```


## Performance

The default parameters were chosen not to minimize computational time, but rather to increase output quality. Admittedly, improvements could also be made to the code to parallelize calculations and reduce processing time.

Below you can find my performance metrics I observed on my data.

###### [extract_faces.py](src/extract_faces.py)
With default parameters, an average of 2.613 images can be processed per second on a 3070 mobile GPU.

The default RetinaFace model has outstanding face detection performance, to the point that some extractions may be correct but unusable in next steps.

###### [make_embeddings.py](src/make_embeddings.py)
With default parameters, an average of 5.626 face embeddings can be created per second on a 3070 mobile GPU.

###### [cluster.py](src/cluster.py)
With default parameters, it takes 4.5 minutes to cluster 38096 embeddings of dimentionality 128.

After labelling 6604 out of 49885 faces, I ran a grid-search over HDBSCAN hyperparameters with a 5-fold cross validation to get the hyperparameters which showed best validation performance on my data. They are the ones set as defaults for the script.

The scores, averaged across folds, are reported here:

| Metric                      | Validation Score | Training Score |
|-----------------------------|------------------|----------------|
| Adjusted Mutual Information | 0.5727           | 0.6363         |
| Adjusted Rand Index         | 0.2079           | 0.2699         |
| Completeness                | 0.6327           | 0.6449         |
| Homogeneity                 | 0.5949           | 0.6513         |
| V-Measure                   | 0.6131           | 0.6481         |
| Fowlkes-Mallows             | 0.3336           | 0.3687         |

###### [copy_photos_person.py](src/copy_photos_person.py)

The performance of this step will vary greatly depending on the speed of your storage.
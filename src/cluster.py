# -*- coding: utf-8 -*-

import os
import numpy as np
import hdbscan
import shutil
import matplotlib.pyplot as plt

faces_dir = 'faces'
embeddings_save_file = 'faces_embeddings.npz'
cluster_faces_images = 'clustered_faces'


clusterer = hdbscan.HDBSCAN(min_samples = 5, min_cluster_size = 20, memory = 'cluster_cache')

faces_embeddings = np.load(embeddings_save_file)['embeddings']
cluster_labels = clusterer.fit_predict(faces_embeddings)

print(f'{len(set(cluster_labels))} clusters found.')

shutil.rmtree(cluster_faces_images)
for cluster in set(cluster_labels):
    cluster_folder = os.path.join(cluster_faces_images, str(cluster))
    os.makedirs(cluster_folder, exist_ok = False)

    idx_cluster = np.arange(len(cluster_labels))[cluster_labels == cluster]
    for idx_image in idx_cluster:
        image_filename = f"{idx_image}.png"
        shutil.copy(os.path.join(faces_dir, image_filename), os.path.join(cluster_folder, image_filename))


#clusterer.condensed_tree_.plot(label_clusters = True)
#plt.savefig(os.path.join(cluster_faces_images, 'results.png'), dpi = 600)
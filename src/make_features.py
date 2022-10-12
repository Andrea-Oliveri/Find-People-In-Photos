# -*- coding: utf-8 -*-

import os
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd



def get_feature_extractor(model_path):
    model = tf.saved_model.load(model_path)
    
    extractor_func = lambda image: model.signatures['serving_default'](input = tf.convert_to_tensor(image, dtype = tf.float32), phase_train = tf.constant(False, dtype = tf.bool))

    return extractor_func



faces_dir = 'faces'
tsv_path = os.path.join(faces_dir, 'faces.tsv')
model_path = 'facenet_model'
embeddings_save_file = 'faces_embeddings.npz'
extractor_input_size = 160

extractor = get_feature_extractor(model_path)
embeddings = []


df = pd.read_table(tsv_path, encoding = 'ISO-8859-1')
paths = [os.path.join(faces_dir, f"{id_}.png") for id_ in df.id]


for image_path in paths:
    
    image = cv2.imread(image_path)
    image = cv2.resize(image, (extractor_input_size, extractor_input_size))
    image = image.astype(np.float32)
    
    image -= image.mean()
    image /= image.std()
    
    # This works, can also be batched however.
    image = image[None, ...]
    res = extractor(image)['output'].numpy().squeeze()
    
    embeddings.append(res)

    if not len(embeddings) % 10:
        print(f"Processed {len(embeddings)} out of {len(paths)} files")

embeddings = np.stack(embeddings, axis = 0)
np.savez(embeddings_save_file, embeddings = embeddings)
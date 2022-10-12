# -*- coding: utf-8 -*-

import os
import pandas as pd
import shutil

faces_tsv_file = 'faces/faces.tsv'
person_faces_folder = 'faccie_person'
out_folder = 'Photos Person'


try:
    faces_df = pd.read_table(faces_tsv_file, encoding='utf-8')
except UnicodeDecodeError:
    faces_df = pd.read_table(faces_tsv_file, encoding='ISO-8859-1')


faces_person = [int(os.path.splitext(p)[0]) for p in os.listdir(person_faces_folder)]

images_to_copy = faces_df[faces_df.id.isin(faces_person)].image_path.unique()

os.makedirs(out_folder, exist_ok=True)
for image_path in images_to_copy:
    basename = os.path.basename(image_path)

    try:
        shutil.copy(image_path, os.path.join(out_folder, basename))
    except FileNotFoundError:
        print('File not Found:', image_path)
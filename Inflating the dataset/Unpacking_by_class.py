import os
import pandas as pd

class_names = ['Dent', 'Fastener Damage', 'Rupture']
splits = ['train', 'valid', 'test']
base_data_path = os.path.abspath('../data')

for split in splits:
    split_path = os.path.join(base_data_path, split)
    annotations_path = os.path.join(split_path, '_annotations.csv')

    if not os.path.exists(annotations_path):
        continue

    df = pd.read_csv(annotations_path)

    for _, row in df.iterrows():
        filename = row['filename']
        label = row['class']

        src_path = os.path.normpath(os.path.join(split_path, filename))
        dst_dir = os.path.normpath(os.path.join(split_path, label))
        dst_path = os.path.normpath(os.path.join(dst_dir, filename))

        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.exists(src_path):
            continue

        os.replace(src_path, dst_path)

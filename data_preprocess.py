from pathlib import Path
import pandas as pd
from tqdm import tqdm

dataset_path = Path('/self-built-masked-face-recognition-dataset')
masked_path = dataset_path/'AFDB_masked_face_dataset'
non_masked_path = dataset_path/'AFDB_face_dataset'
df_mask = pd.DataFrame()

for subject in tqdm(list(non_masked_path.iterdir()), desc='photos without mask'):
    for img_path in subject.iterdir():
        df_mask = df_mask.append({
            'image': str(img_path),
            'mask': 0
        }, ignore_index=True)

for subject in tqdm(list(masked_path.iterdir()), desc='photos with mask'):
    for img_path in subject.iterdir():
        df_mask = df_mask.append({
            'image': str(img_path),
            'mask': 1
        }, ignore_index=True)


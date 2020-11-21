from pathlib import Path
import pandas as pd
from tqdm import tqdm

dataset_path = Path('/self-built-masked-face-recognition-dataset')
masked_path = dataset_path/'AFDB_masked_face_dataset'
non_masked_path = dataset_path/'AFDB_face_dataset'
df_mask = pd.DataFrame()


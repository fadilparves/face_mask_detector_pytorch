import cv2
import numpy as np
from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

class TorchDataset(Dataset):
    """Masked faces dataset desc
       0 = 'photos without mask'
       1 = 'photos with mask'
    """

    def __init__(self, df):
        self.df = df

        self.transformations = Compose([
            ToPILImage(),
            Resize((100, 100)),
            ToTensor()
        ])

    def __getitem__(self, key):
        row = self.df.iloc[key]
        image = cv2.imdecode(np.fromfile(row['image'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return {
            'image': self.transformations(row['image']),
            'mask': tensor([row['mask']], dtype=long),
        }

    def __len__(self):
        return len(self.df.index)
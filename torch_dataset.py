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
            Resize((100, 100))
            ToTensor()
        ])
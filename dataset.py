import os
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from skimage import io
import pdb

class FrameDataset(Dataset):
    def __init__(self, csv_file, train_dir):
        self.labels = pd.read_csv(csv_file)
        self.train_dir = train_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((66,220)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
            ])

    def show_img(self, img, denormalize = True):
        inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        if denormalize:
            img = inv_normalize(img)

        plt.imshow(np.transpose(img.numpy(), (1,2,0)))
        plt.show()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = os.path.join(self.train_dir, self.labels.iloc[index][0])
        image = io.imread(img_path)
        y_label = torch.tensor(float(self.labels.iloc[index][1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)





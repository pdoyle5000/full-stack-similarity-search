import numpy as np
import torch
from torchvision import transforms
from copy import deepcopy
from torch.utils.data import Dataset
from PIL import Image
from glob import glob


class CifarDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.paths = glob(f"cifar/{dataset}/*.png")
        print(f"Number of files in the {dataset} set: {len(self.paths)}")
        self.transform = transform
        # self.channel_means = (0.4914, 0.4822, 0.4465)
        # self.channel_stds = (0.2023, 0.1994, 0.2010)
        # self.norm = transforms.Compose(
        #    [transforms.Normalize(self.channel_means, self.channel_stds)]
        # )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        raw_img = Image.open(self.paths[idx])
        if self.transform:
            raw_img = self.transform(raw_img)
        img = np.asarray(raw_img) / 255
        img = img.astype(np.float)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        # img = self.norm(img)

        return img, deepcopy(img), self.paths[idx]


# def _get_dataset_loader(data, transform=None, shuffle=False):
#    return torch.utils.data.DataLoader(
#        CifarDataset(data, transform=transform),
#        batch_size=32,
#        shuffle=shuffle,
#        num_workers=8,
#    )

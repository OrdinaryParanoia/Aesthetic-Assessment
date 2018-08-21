import os
import pandas as pd
from PIL import Image
from torch.utils import data

__all__ = ['AVADataset']

class AVADataset(data.Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, index_col=0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name).convert("RGB")
        annotations = self.annotations.iloc[idx, 1:].values
        annotations = annotations / annotations.sum()
        annotations = annotations.astype('float').reshape(-1, 1)
        sample = {'img_id': img_name, 'image': image, 'annotations': annotations}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

#class AVADataset_forLstm(data.Dataset):
#
#    def __init__(self, csv_file1, csv_file2, root_dir1, root_dir2):
#        self.annotations1 = pd.read_csv(csv_file1, index_col=0)
#        self.root_dir1 = root_dir1
#        self.annotations2 = pd.read_csv(csv_file2, index_col=0)
#        self.root_dir2 = root_dir2
#
#    def __len__(self):
#        return len(self.annotations1)
#
#    def __getitem__(self, idx):
#        img_name1 = os.path.join(self.root_dir1, str(self.annotations1.iloc[idx, 0]) + '.jpg')
#        img_name2 = os.path.join(self.root_dir2, str(self.annotations2.iloc[idx, 0]) + '.jpg')
#        image1 = Image.open(img_name1).convert("RGB")
#        image2 = Image.open(img_name2).convert("RGB")
#        annotations1 = self.annotations1.iloc[idx, 1:].values
#        annotations1 = annotations1 / annotations1.sum()
#        annotations1 = annotations1.astype('float').reshape(-1, 1)
#        annotations2 = self.annotations2.iloc[idx, 1:].values
#        annotations2 = annotations2 / annotations2.sum()
#        annotations2 = annotations2.astype('float').reshape(-1, 1)
#
##        sample = {'img_id1': img_name1, 'image1': image1, 'annotations1': annotations1,
##                  'img_id2': img_name2, 'image2': image2, 'annotations2': annotations2}
#        sample = [image1, image2]
#
#        return sample
#
#class AVADataset_foreachImage(data.Dataset):
#
#    def __init__(self, transform=None):
#        self.transform = transform
#    
#    def __getitem__(self, img_name):
#        image = Image.open(img_name).convert("RGB")
#
##        sample = {'img_id1': img_name1, 'image1': image1, 'annotations1': annotations1,
##                  'img_id2': img_name2, 'image2': image2, 'annotations2': annotations2}
#        sample = {'image': image}
#
#        if self.transform:
#            sample['image'] = self.transform(sample['image'])
#
#        return sample
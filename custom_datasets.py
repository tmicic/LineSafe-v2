from collections import OrderedDict
from typing import Callable, Optional
import torch
import torchvision
from torchvision.datasets import VisionDataset
from torchvision.transforms.transforms import ToTensor
import common_functions as common
import pandas as pd
import dicom_processing
import numpy as np
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt

class UnbalancedDataset(VisionDataset):

    def __init__(self, df_path: str, 
                        root: str, 
                        transform=None, 
                        target_transform=None,
                        loader=dicom_processing.auto_loader) -> None:

        super().__init__(root, transforms=None, transform=transform, target_transform=target_transform)

        self.df_path = df_path

        if df_path.endswith('.csv'):
            self.df = pd.read_csv(df_path, index_col=0)
        else:
            self.df = pd.read_feather(df_path)
            del self.df['index']

        self.loader = loader
        
        self.classes = sorted(self.df['category'].unique())                 # make sure that training and validation set have same classes!!! in same order!!
        self.class_to_id = {x:i for i,x in enumerate(self.classes)}

    def __getitem__(self, index: int):
        
        item = self.df.iloc[index]

        X = self.loader(os.path.join(self.root, item['category'],item['filename']))
        if self.transform is not None: X = self.transform(X)
            
        y = torch.tensor(self.class_to_id[item['category']])
        
        if self.target_transform is not None: y = self.target_transform(y)

        return X, y

    def get_class_ratios(self):
        # returns the ratio of each class in the order of class_to_id
        x = self.df.groupby('category')['hash'].nunique()

        rs = torch.tensor([x[r] for r in self.class_to_id])

        return rs / rs.sum()

    def __len__(self) -> int:
        return len(self.df)

class UnbalancedSelfSupervisedRotationalDataset(UnbalancedDataset):
    # 0 is 0 deg, 1 is 90 deg, 2 is 180 deg, 3 is 270 deg 

    def __init__(self, df_path: str, 
                        root: str, 
                        transform=None, 
                        target_transform=None,
                        loader=dicom_processing.auto_loader) -> None:
        super().__init__(df_path, root, transform=transform, target_transform=target_transform, loader=loader)

    def __getitem__(self, index: int):
        
        item = self.df.iloc[index]

        X = self.loader(os.path.join(self.root, item['category'],item['filename']))
        if self.transform is not None: X = self.transform(X)

        rot = torch.tensor(random.randint(0,3))

        if rot > 0:
            y = X.rot90(k=rot.item(), dims=(1,2))
        else:
            y = X

        cat = torch.tensor(self.class_to_id[item['category']])
        

        return y, X, rot, cat           # rotated image is now X, X is the label rot

class UnetDataset(UnbalancedDataset):

    def __init__(self, df_path: str, 
                        root: str,
                        map_root: str,
                        transform=None, 
                        target_transform=None,
                        loader=dicom_processing.auto_loader,
                        target_loader=dicom_processing.segmentation_image_loader,
                        allow_non_segmentated_images=True,      # if true, will return ng maps and films without ng tubes with blank maps
                        blank_map_dims=(512,512)) -> None:      # before any transforms
        super().__init__(df_path, root, transform=transform, target_transform=target_transform, loader=loader)


        self.map_root = map_root
        self.target_loader = target_loader
        self.allow_non_segmentated_images = allow_non_segmentated_images
        self.blank_map_dims = blank_map_dims

        if not self.allow_non_segmentated_images:
            self.df = self.df[self.df['ng_roi_filename'].notnull()]
        else:
            #allow no ng tube and ng tubes with ng_roi_filename
            self.df = self.df[(self.df['ng_roi_filename'].notnull() | (self.df['category'] == 'NO_NG'))]

    def __getitem__(self, index: int):
        
        item = self.df.iloc[index]

        X = self.loader(os.path.join(self.root, item['category'],item['filename']))
        if self.transform is not None: X = self.transform(X)
        
        cat = torch.tensor(self.class_to_id[item['category']])

        if item['ng_roi_filename'] is None:
            # return blank target
            y = np.zeros(self.blank_map_dims)
        else:
            y = self.target_loader(os.path.join(self.map_root,item['ng_roi_filename']))

        if self.target_transform is not None: y = self.target_transform(y)

        return X, y, cat

class SiameseDataset(UnbalancedDataset):

    def __init__(self, df_path: str, 
                        root: str, 
                        transform=None, 
                        target_transform=None,
                        loader=dicom_processing.auto_loader, 
                        return_type='contrastive',   # or triplet
                        ignore_no_tubes=True,
                        ) -> None:
        super().__init__(df_path, root, transform=transform, target_transform=target_transform, loader=loader)

        self.ignore_no_tubes = ignore_no_tubes
        self.return_type = return_type

        if ignore_no_tubes:
            self.df = self.df[self.df['category'] != 'NO_NG']

    def __getitem__(self, index: int):
        item = self.df.iloc[index]

        X = self.loader(os.path.join(self.root, item['category'],item['filename']))
        if self.transform is not None: X = self.transform(X)

        cat_name = item['category']
        cat = torch.tensor(self.class_to_id[cat_name])

        if self.return_type == 'triplet':
            # return X as anchor, Pos, Neg, cat

            pos = self.df[self.df['category']==cat_name].sample(n=1).iloc[0]
            neg = self.df[self.df['category']!=cat_name].sample(n=1).iloc[0]

            pos_image = self.loader(os.path.join(self.root, pos['category'], pos['filename']))
            neg_image = self.loader(os.path.join(self.root, neg['category'], neg['filename']))

            if self.transform is not None:
                pos_image = self.transform(pos_image)
                neg_image = self.transform(neg_image)

            return X, pos_image, neg_image, cat

        elif self.return_type == 'contrastive':
            # return X, Y, same/diff (0,1), cat of X
            same_or_diff = torch.tensor(random.randint(0,1))

            if same_or_diff == 0:   # same = 0
                contrastive = self.df[self.df['category']==cat_name].sample(n=1).iloc[0]
            else:                   # diff = 1
                contrastive = self.df[self.df['category']!=cat_name].sample(n=1).iloc[0]

            contrastive_image = self.loader(os.path.join(self.root, contrastive['category'], contrastive['filename']))

            if self.transform is not None:
                contrastive_image = self.transform(contrastive_image)
            
            return X, contrastive_image, same_or_diff, cat
        
        else:
            raise NotImplementedError(f'Siamese Dataset has not implemented the return_type of {self.return_type}.')






if __name__ == '__main__':
    print('debugging')
    import custom_transforms

    transform = transforms.Compose([
                                
                                transforms.ToTensor(),
                                custom_transforms.ToMultiChannel(3),
                                transforms.Resize((256,256)),
                            ])

    
    train_dataset = SiameseDataset(common.TRAIN_DF_PATH, 
                            root=common.SATO_IMAGES_ROOT_PATH, 
                            loader=dicom_processing.auto_loader,
                            transform=transform,
                            target_transform=None,
                            ignore_no_tubes=True, 
                            return_type='contrastive')


    train_dataset.__getitem__(0)












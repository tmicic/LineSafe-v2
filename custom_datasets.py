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
import torchvision.transforms.functional as f_trans

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
                        blank_map_dims=(256,256),
                        random_v_flip_rate=None,
                        random_h_flip_rate=None,
                        random_4_point_rotation_rate=None,
                        
                        ) -> None:      # before any transforms
        super().__init__(df_path, root, transform=transform, target_transform=target_transform, loader=loader)


        self.map_root = map_root
        self.target_loader = target_loader
        self.allow_non_segmentated_images = allow_non_segmentated_images
        self.blank_map_dims = blank_map_dims
        self.random_v_flip_rate=random_v_flip_rate
        self.random_h_flip_rate=random_h_flip_rate
        self.random_4_point_rotation_rate=None

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

        if self.random_h_flip_rate is not None:
            to_flip = np.random.choice([True, False], p=[self.random_h_flip_rate, 1-self.random_h_flip_rate])
            if to_flip:
                X = f_trans.hflip(X)
                y = f_trans.hflip(y)

        if self.random_v_flip_rate is not None:
            to_flip = np.random.choice([True, False], p=[self.random_v_flip_rate, 1-self.random_v_flip_rate])
            if to_flip:
                X = f_trans.vflip(X)
                y = f_trans.vflip(y)           

        if self.random_4_point_rotation_rate is not None:
            rot = np.random.choice([0.,90.,180.,270.], p=[self.random_4_point_rotation_rate])
            if rot > 0.:
                X = f_trans.rotate(X, rot)
                y = f_trans.rotate(X, rot)     

        return X, y, cat



class LineSafeDataset():

    def __init__(self, 
                    df_path,    # path the csv file
                    dicom_root, # path the dicom images
                    segmentation_map_root=None, # path to the ng tube segmentation maps
                    transform=None,     # transform applied to the dicom_image
                    ignore_no_ng=True,  # removes no_ngs from the dataset
                    loader=dicom_processing.auto_loader,    # loader used for the dicom images
                    target_loader=dicom_processing.segmentation_image_loader,   # loader used for the segmentation maps
                    default_blank_seg_map_size=(256,256),   # if no segmentation map, default blank image size to return
                    seg_map_transform=None,  # transform applied to the segmentation map 
                    only_images_with_segmentations=False, # if true, only images with segmentation maps will be returned, if false, if no segmentation map, blank maps are returned                
                    return_type=None,# see below
                    random_v_flip_rate=None,    # rate of how often a v flip is done, 
                    random_h_flip_rate=None,    # rate of how often a h flip is done - if none, no flipping
                    random_4_point_rotation_rate=None, # list of frequencies of rotations or 0, 90, 180, and 270
                    return_segmentations=True
                    ):

                    # return_type 
                    #   None or 'unbal' = unbalanced (dicom, category, [segmentation])
                    #   'bal' = balanced (dicom, category, [segmentation])  
                    #   'contrastive' = (dicom1, category1, [segmentation1], dicom2, category2, [segmentation2], is_diff)
                    #   'triplet' = (dicom_anchor, category_anchor, [segmentation_anchor], positive_dicom, 
                    #               positive_category, [positive_segmentation], negative_dicom, negative_cat, [negative_segmentation])
        
        self.df_path = df_path
        self.dicom_root = dicom_root
        self.seg_map_root = segmentation_map_root
        self.random_v_flip_rate = random_v_flip_rate
        self.random_h_flip_rate = random_h_flip_rate
        self.random_4_point_rotation_rate=random_4_point_rotation_rate,
        self.transform = transform
        self.seg_map_transform = seg_map_transform
        self._default_map_size = default_blank_seg_map_size
        self.loader = loader
        self.target_loader = target_loader
        self.return_segmentations = return_segmentations
        self.return_type = return_type

        if df_path.endswith('.csv'):
            self.df = pd.read_csv(df_path, index_col=0)
        else:
            self.df = pd.read_feather(df_path)
            del self.df['index']

        if only_images_with_segmentations:
            self.df = self.df[self.df['ng_roi_filename'].notnull()]

        if ignore_no_ng:
            self.df = self.df[self.df['category'] != 'NO_NG']

        self.classes = sorted(self.df['category'].unique())                 # make sure that training and validation set have same classes!!! in same order!!
        self.class_to_id = {x:i for i,x in enumerate(self.classes)}        
        
        if return_type == 'bal':
            min_count = min(self.df.groupby('category')['hash'].count().values)
            self.df = pd.concat([self.df[self.df['category']==c].sample(n=min_count) for c in self.classes])

    def __len__(self) -> int:
        return len(self.df)


    def __getitem__(self, index: int):
        
        item = self.df.iloc[index]

        X = self.loader(os.path.join(self.dicom_root, item['category'], item['filename']))
        if self.transform is not None: X = self.transform(X)

        y = torch.tensor(self.class_to_id[item['category']])

        seg = None
        if self.return_segmentations:
            if item['ng_roi_filename'] is None:
            # return blank target
                seg = np.zeros(self.blank_map_dims)
            else:
                seg = self.target_loader(os.path.join(self.map_root,item['ng_roi_filename']))

        if seg is not None and self.seg_map_transform is not None:
            seg = self.seg_map_transform(seg)

        if self.return_type is None or self.return_type == 'bal' or self.return_type == 'unbal':
            if seg is None:
                return X, y
            else:
                return X, y, seg
        elif self.return_type=='contrastive':
            raise NotImplementedError()
        elif self.return_type=='triplet':
            raise NotImplementedError()
        else:
            raise NotImplementedError() 





        
        # if self.target_transform is not None: y = self.target_transform(y)

        # return X, y




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

    DATASET_PATH = r'datasets\train.csv'
    SATO_IMAGES_ROOT_PATH = r'C:\Users\Tom\Google Drive\Documents\PYTHON PROGRAMMING\AI\data\SATO1'
    NG_ROI_ROOT_PATH = r'C:\Users\Tom\Google Drive\Documents\PYTHON PROGRAMMING\AI\data\ROIS\NG_ROI'
    
    dataset = LineSafeDataset(df_path=DATASET_PATH, 
                            dicom_root=SATO_IMAGES_ROOT_PATH, 
                            segmentation_map_root=NG_ROI_ROOT_PATH,
                            ignore_no_ng=False,
                            only_images_with_segmentations=False,
                            return_type='bal')

    print(dataset.df)

    



    # import custom_transforms

    # transform = transforms.Compose([
                                
    #                             transforms.ToTensor(),
    #                             custom_transforms.ToMultiChannel(3),
    #                             transforms.Resize((256,256)),
    #                         ])

    
    # train_dataset = LineSafeDataset(common.TRAIN_DF_PATH,
    #                         dicom_root=common. 
    #                         root=common.SATO_IMAGES_ROOT_PATH, 
    #                         loader=dicom_processing.auto_loader,
    #                         transform=transform,
    #                         target_transform=None,
    #                         ignore_no_tubes=True, 
    #                         return_type='contrastive')


    # train_dataset.__getitem__(0)












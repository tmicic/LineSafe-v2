import common_functions as common
import os
import pandas as pd
import numpy as np

if common.ENSURE_REPRODUCIBILITY: common.ensure_reproducibility(common.REPRODUCIBILITY_SEED)

# Create a dataframe containing the hash, filename and category of each image in the dataset
images = []

for category in os.listdir(common.SATO_IMAGES_ROOT_PATH):

    for file in os.listdir(os.path.join(common.SATO_IMAGES_ROOT_PATH, category)):
        
        path = os.path.join(common.SATO_IMAGES_ROOT_PATH, category, file)
        hash = file.split('.')[0]

        images.append((hash, file, category))

# now create a dataframe to go through the rois for both ng and hilar points
ng_rois = []

for file in os.listdir(common.NG_ROI_ROOT_PATH):
        
    path = os.path.join(common.NG_ROI_ROOT_PATH, file)
    hash = file.split('.')[0]

    ng_rois.append((hash, file))
        
hilar_rois = []

for file in os.listdir(common.HILAR_POINT_ROI_ROOT_PATH):
        
    path = os.path.join(common.HILAR_POINT_ROI_ROOT_PATH, file)
    hash = file.split('.')[0]

    hilar_rois.append((hash, file))

# create the dataframes from the above lists of tuples
films_df = pd.DataFrame(images, columns=['hash', 'filename', 'category'])
ng_roi_df = pd.DataFrame(ng_rois, columns=['hash', 'ng_roi_filename'])   
hilar_roi_df = pd.DataFrame(hilar_rois, columns=['hash', 'hilar_point_roi_filename'])   

# set index to hash so that we can to left joins
films_df.set_index('hash', inplace=True)
ng_roi_df.set_index('hash', inplace=True)
hilar_roi_df.set_index('hash', inplace=True)

films_df = films_df.join(ng_roi_df)
films_df = films_df.join(hilar_roi_df)

# shuffle the dataset
films_df = films_df.sample(frac=1)
films_df.reset_index(inplace=True)

# create train and validate datasets and save them. The split should be the same as train and validate split
train_df = films_df.iloc[:int(len(films_df) * common.TRAIN_VALIDATE_SPLIT)]
validate_df = films_df.iloc[int(len(films_df) * common.TRAIN_VALIDATE_SPLIT):]

train_df.reset_index(inplace=True)
validate_df.reset_index(inplace=True)

train_df.to_feather('train.df')
validate_df.to_feather('validate.df')

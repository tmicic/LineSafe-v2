from numpy import infty
import common_functions as common
import models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from custom_datasets import UnetDataset
import dicom_processing
import torchvision.transforms as transforms
import custom_transforms 
from contextlib import nullcontext
import matplotlib.pyplot as plt
import torch.optim as optim
import os
from metrics import Metrics
import drawing

### VARIABLES ###
DEBUG = True                        # if true, debug mode is on and things will be written to the screen
REPRODUCIBILITY_SEED = 6000         # Holly's IQ used as seed to ensure reproducibility
ENSURE_REPRODUCIBILITY = True
SUPPRESS_WARNINGS = True
NUMBER_OF_WORKERS = 4
USE_CUDA = True

TRAIN_BATCH_SIZE = 32
VALIDATE_BATCH_SIZE = 64

LR = 0.0001
TRAIN_EPOCHS = 100

TRAIN_SHUFFLE = True
VALIDATE_SHUFFLE = False

# SATO_IMAGES_ROOT_PATH = r'C:\Users\Tom\Google Drive\Documents\PYTHON PROGRAMMING\AI\data\SATO1'
# NG_ROI_ROOT_PATH = r'C:\Users\Tom\Google Drive\Documents\PYTHON PROGRAMMING\AI\data\ROIS\NG_ROI'
# HILAR_POINT_ROI_ROOT_PATH = r'C:\Users\Tom\Google Drive\Documents\PYTHON PROGRAMMING\AI\data\ROIS\HILAR_POINTS_ROI'

# on academy server
SATO_IMAGES_ROOT_PATH = r'C:\Users\rigwa\Desktop\LineSafeV2\Data\SATO'
NG_ROI_ROOT_PATH = r'C:\Users\rigwa\Desktop\LineSafeV2\Data\NG_ROI'
HILAR_POINT_ROI_ROOT_PATH = r'C:\Users\rigwa\Desktop\LineSafeV2\Data\HILAR_POINTS_ROI'

TRAIN_DF_PATH = r'datasets\train.csv'
VALIDATE_DF_PATH = r'datasets\validate.csv'

MODEL_PATH = r'unet_only_ngs_flip_and_rot.model'

ALWAYS_VALIDATE_MODEL_FIRST = True

TRAIN_VALIDATE_SPLIT = 0.8

POSITIVE_CLASS = 0          # NG_NOT_OK is the positive class

DEVICE = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')
      
if __name__ == '__main__':

    common.ensure_reproducibility(ENSURE_REPRODUCIBILITY)  

    train_transform = transforms.Compose([
                                
                                transforms.ToTensor(),
                                transforms.Resize((256,256)),
                                transforms.Normalize([0.5],[0.225]),
                            ])

    train_target_transform = transforms.Compose([  
                                transforms.ToTensor(),
                                transforms.Resize((256,256)),
                            ])
    
    # no random flipping in val transforms
    validate_transform = transforms.Compose([
                                
                                transforms.ToTensor(),
                                transforms.Resize((256,256)),
                                transforms.Normalize([0.5],[0.225]),
                            ])

    validate_target_transform = transforms.Compose([  
                                transforms.ToTensor(),
                                transforms.Resize((256,256)),
                            ])


    train_dataset = UnetDataset(TRAIN_DF_PATH, 
                            root=SATO_IMAGES_ROOT_PATH, 
                            map_root=NG_ROI_ROOT_PATH,
                            loader=dicom_processing.auto_loader,
                            transform=train_transform,
                            target_transform=train_target_transform,
                            allow_non_segmentated_images=False,
                            blank_map_dims=(256,256),
                            random_v_flip_rate=0.5,
                            random_h_flip_rate=0.5,
                            random_4_point_rotation_rate=[0.25,0.25,0.25, 0.25])

    validate_dataset = UnetDataset(VALIDATE_DF_PATH, 
                            root=SATO_IMAGES_ROOT_PATH, 
                            map_root=NG_ROI_ROOT_PATH,
                            loader=dicom_processing.auto_loader,
                            transform=validate_transform,
                            target_transform=validate_target_transform,
                            allow_non_segmentated_images=False)

    train_dataloader = DataLoader(train_dataset, TRAIN_BATCH_SIZE, shuffle=TRAIN_SHUFFLE, num_workers=NUMBER_OF_WORKERS)
    validate_dataloader = DataLoader(validate_dataset, VALIDATE_BATCH_SIZE, shuffle=VALIDATE_SHUFFLE, num_workers=NUMBER_OF_WORKERS)
    
    

    model = models.UNet(n_channels=1, n_classes=1).to(DEVICE) # 0 = 0 deg, 1 = 90 deg, 2 = 180 deg, 3 = 270 deg
    optimizer = optim.Adam(model.parameters(), lr=0.001)#, weight_decay=1e-8, momentum=0.9) # lr = 0.001
    criterion = nn.BCELoss().to(DEVICE) # if one channel else cross entropy
    
    # load model
    if MODEL_PATH != '':
        if os.path.exists(MODEL_PATH):
            print(f'Loading model ({MODEL_PATH})...')
            model = common.load_model_state(model, MODEL_PATH)
        else:
            print(f'Model file does not exist.')

    best_loss = None     # stores the best metrics

    for epoch in range(0 if ALWAYS_VALIDATE_MODEL_FIRST else 1, TRAIN_EPOCHS + 1):    # do an epoch 0 if pre-val required

        if epoch > 0:
            print(f'Epoch {epoch} of {TRAIN_EPOCHS}:')
        else:
            print(f'Pre-evaluating model...')

        for training, dataset, dataloader in [(True, train_dataset, train_dataloader), 
                                                (False, validate_dataset, validate_dataloader)]:

            if epoch == 0 and training:
                # pre-eval therefore skip training
                continue

            if training:
                context_manager = nullcontext()
                model.train()
                print('\tTraining:')
            else:
                context_manager = torch.no_grad()
                model.eval()
                print('\tEvaluating:')

            
            with context_manager:
                
                total_loss = 0.

                for i, (X, y, cat) in enumerate(dataloader):
                    
                    X = X.to(DEVICE)
                    y = y.to(DEVICE)

                    if training: optimizer.zero_grad()

                    output = model(X)

                    loss = criterion(output, y)
                    
                    total_loss += loss.item()                    
                    
                    if training:
                        loss.backward()
                        optimizer.step()

                    print(f'\r\tBatch: {i+1} of {len(dataloader)}: loss: {loss.item():.4f}', end='')
                    
            print()
            
            if not training: print('---------------------------------------------------')
            print(f'\t Total loss: {total_loss:.4f}')
            if not training: print('---------------------------------------------------')

            if not training:

                # draw the last dataset
                drawing.update_figure_unet(X.detach().cpu(), y.detach().cpu(), output.detach().cpu())

                
                # update stats, save model
                if best_loss is None:
                    best_loss = total_loss
                else:
                    if total_loss <= best_loss and epoch > 0:   # don't resave the model if its a pre-evaluation
                        print(f'Current model ({total_loss}) out-performed previous best model ({best_loss}). Saving new model...')
                        best_loss = total_loss
                        common.save_model_state(model, MODEL_PATH)

    drawing.prevent_figure_close()




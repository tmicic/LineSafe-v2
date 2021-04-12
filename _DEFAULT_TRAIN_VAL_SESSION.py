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
      
if __name__ == '__main__':

    common.ensure_reproducibility(common.ENSURE_REPRODUCIBILITY)  

    transform = transforms.Compose([
                                
                                transforms.ToTensor(),
                                transforms.Resize((256,256)),
                                transforms.Normalize([0.5],[0.225])
                            ])

    target_transform = transforms.Compose([  
                                transforms.ToTensor(),
                                transforms.Resize((256,256)),
                            ])


    train_dataset = UnetDataset(common.TRAIN_DF_PATH, 
                            root=common.SATO_IMAGES_ROOT_PATH, 
                            map_root=common.NG_ROI_ROOT_PATH,
                            loader=dicom_processing.auto_loader,
                            transform=transform,
                            target_transform=target_transform,
                            allow_non_segmentated_images=False)

    validate_dataset = UnetDataset(common.VALIDATE_DF_PATH, 
                            root=common.SATO_IMAGES_ROOT_PATH, 
                            map_root=common.NG_ROI_ROOT_PATH,
                            loader=dicom_processing.auto_loader,
                            transform=transform,
                            target_transform=target_transform,
                            allow_non_segmentated_images=False)

    train_dataloader = DataLoader(train_dataset, 32, shuffle=common.TRAIN_SHUFFLE, num_workers=common.NUMBER_OF_WORKERS)
    validate_dataloader = DataLoader(validate_dataset, common.VALIDATE_BATCH_SIZE, shuffle=common.VALIDATE_SHUFFLE, num_workers=common.NUMBER_OF_WORKERS)
    
    MODEL_PATH = r'unet_256x256_sing_ch_sigmoid_normalised.model'

    model = models.UNet(n_channels=1, n_classes=1).to(common.DEVICE) # 0 = 0 deg, 1 = 90 deg, 2 = 180 deg, 3 = 270 deg
    optimizer = optim.Adam(model.parameters(), lr=0.0001)#, weight_decay=1e-8, momentum=0.9) # lr = 0.001
    criterion = nn.BCELoss().to(common.DEVICE) # if one channel else cross entropy

    # load model
    if MODEL_PATH != '':
        if os.path.exists(MODEL_PATH):
            print(f'Loading model ({MODEL_PATH})...')
            model = common.load_model_state(model, MODEL_PATH)
        else:
            print(f'Model file does not exist.')

    best_loss = None     # stores the best metrics

    for epoch in range(0 if common.ALWAYS_VALIDATE_MODEL_FIRST else 1, common.TRAIN_EPOCHS + 1):    # do an epoch 0 if pre-val required

        if epoch > 0:
            print(f'Epoch {epoch} of {common.TRAIN_EPOCHS}:')
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
                    
                    X = X.to(common.DEVICE)
                    y = y.to(common.DEVICE)


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
                import matplotlib.pyplot as plt

                plt.show()
                
                # update stats, save model
                if best_loss is None:
                    best_loss = total_loss
                else:
                    if total_loss <= best_loss and epoch > 0:   # don't resave the model if its a pre-evaluation
                        print(f'Current model ({total_loss}) out-performed previous best model ({best_loss}). Saving new model...')
                        best_loss = total_loss
                        common.save_model_state(model, MODEL_PATH)

    drawing.prevent_figure_close()




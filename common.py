import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#!pip install torchsummary
from torchsummary import summary
#!pip install tensorboard
from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import ReduceLROnPlateau



import os, glob

import csv
import matplotlib.pyplot as plt
import random
#!pip install scikit-image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

#!pip install lpips
import lpips



from archithectures import *
from testing import *
from loading import *



def plot_hyperspectral_images_false_color_train(lr_img, hr_img, pred_img, idx, bands=[30, 50, 70], cmap='terrain'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))


    lr_color = lr_img[bands, :, :].transpose(1, 2, 0)
    hr_color = hr_img[bands, :, :].transpose(1, 2, 0)
    pred_color = pred_img[bands, :, :].transpose(1, 2, 0)


    lr_color = (lr_color - lr_color.min()) / (lr_color.max() - lr_color.min())
    hr_color = (hr_color - hr_color.min()) / (hr_color.max() - hr_color.min())
    pred_color = (pred_color - pred_color.min()) / (pred_color.max() - pred_color.min())

    lr_vmin, lr_vmax = lr_color.min(), lr_color.max()
    hr_vmin, hr_vmax = hr_color.min(), hr_color.max()
    pred_vmin, pred_vmax = pred_color.min(), pred_color.max()

    vmin = min(lr_vmin, hr_vmin, pred_vmin)
    vmax = min(lr_vmax, hr_vmax, pred_vmax)

    #axes[0].imshow(lr_color)
    axes[0].imshow(lr_color,  vmin=vmin, vmax=vmax)
    axes[0].set_title(f'LR Image (Bands {bands})')
    axes[0].axis('off')


    #axes[1].imshow(hr_color)
    axes[1].imshow(hr_color,  vmin=vmin, vmax=vmax)
    axes[1].set_title(f'HR Image (Bands {bands})')
    axes[1].axis('off')


    #axes[2].imshow(pred_color)
    axes[2].imshow(pred_color,  vmin=vmin, vmax=vmax)
    axes[2].set_title(f'Predicted Image (Bands {bands})')
    axes[2].axis('off')

    fig.canvas.draw()
    return fig









class CustomLoader:
    def __init__(self, data, mean,std, *args, **kwargs):
        print("len of data:", len(data))
        self.loader = DataLoader(data, *args, **kwargs)
        """self.mean = mean
        self.std = std"""
        """self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)"""
        self.mean = torch.tensor(mean,dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(std,dtype=torch.float32).view(1, -1, 1, 1)


def S5_DSCR_S_train(args,train_loader,valid_loader,num_bands,device,correct_relu = True,same_kernel = False, bias = False,compression="no",last_conv = False,min_val_stopping = False,mean=torch.tensor(0.0), std=torch.tensor(1.0)):
    model = S5_DSCR_S(in_channels=num_bands, 
                            out_channels=num_bands, 
                            num_spectral_bands=num_bands, 
                            depth_multiplier=1, 
                            upsample_scale=4, 
                            mode='convtranspose',
                            correct_relu=correct_relu,
                            same_kernel=same_kernel,
                            bias=bias,
                            compression=compression,
                            last_conv=last_conv,
                            mean=mean,
                            std=std).to(device)
    model_name = 'DSC2'
    return generic_train(model,model_name,args,train_loader,valid_loader,num_bands,device,min_val_stopping=min_val_stopping)



def S5_DSCR_train(args,train_loader,valid_loader,num_bands,device,correct_relu = True, same_kernel = False, bias = False,compression="no",last_conv = False,min_val_stopping = False,mean=torch.tensor(0.0), std=torch.tensor(1.0)):

    model = S5_DSCR(
        in_channels=num_bands,
        out_channels=num_bands,
        num_spectral_bands=num_bands,
        depth_multiplier=3,
        num_layers=5,
        kernel_size=5,
        upsample_scale=4,
        correct_relu=correct_relu,
        same_kernel=same_kernel,
        bias=bias,
        compression=compression,
        last_conv=last_conv,
        mean=mean,
        std=std
    ).to(device)
    model_name = 'DSC_residual2'
    return generic_train(model,model_name,args,train_loader,valid_loader,num_bands,device,min_val_stopping=min_val_stopping)


def generic_train(model,model_name,args,train_loader,valid_loader,num_bands,device,min_val_stopping=False):
    model = model.to(device)
    summary(model, input_size=(num_bands, 64, 64))

    log_dir = os.path.join(args.save_dir, args.save_prefix, model_name)
    writer_tensor = SummaryWriter(log_dir=log_dir)

    criterion = nn.L1Loss() if args.net_loss == 'L1norm' else ( psnr_loss  if args.net_loss == 'PSNR' else nn.MSELoss() )
    optimizer = optim.Adam(model.parameters(), lr=args.net_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    train_losses, val_losses = [], []
    min_loss_val = float('inf')
    model_to_save = None
    stopped_epoch = -1
    for epoch in range(args.nepochs):
        #print the epoch number and the learning rate but not returning to the line so next prints will overwrite it
        print("epoch", epoch+1, "/", args.nepochs, " lr:", optimizer.param_groups[0]['lr'], "\r", end="")
        model.train()
        epoch_loss = 0
        mean, std = train_loader.mean, train_loader.std
        for batch_idx, (lr, hr) in enumerate(train_loader.loader):
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            output = model(lr)
            #normalize output and hr to [0, 1] based on hr min and max on each datapoint and each channel
            #shape of output and hr is (batch_size,channels, height, width)
            """max_hr = hr.max()
            output = output / max_hr
            hr = hr / max_hr"""
            #compute the max of each image in the batch
            max_lr = lr.view(lr.size(0), -1).max(1)[0].view(-1, 1, 1, 1)
            output = output / max_lr
            hr = hr / max_lr
            
            loss = criterion(output, hr)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            
            if epoch % 2 == 0 and batch_idx == 0:  
                for i in range(min(5, len(lr))):  
                    lr_img = lr.cpu().numpy()[i]
                    hr_img = hr.cpu().numpy()[i]
                    pred_img = output.cpu().detach().numpy()[i]
                    fig = plot_hyperspectral_images_false_color_train(lr_img, hr_img, pred_img, idx=epoch * len(train_loader.loader) + batch_idx * len(lr) + i)
                    writer_tensor.add_figure(f'Predictions vs Actuals/Epoch_{epoch}', fig, global_step=epoch * len(train_loader.loader) + batch_idx * len(lr) + i)
                pass
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            mean, std = valid_loader.mean, valid_loader.std
            for lr, hr in valid_loader.loader:
                lr, hr = lr.to(device), hr.to(device)
                output = model(lr)
                #normalize output and hr to [0, 1] based on hr min and max on each datapoint and each channel
                """max_hr = hr.max()
                output = output / max_hr
                hr = hr / max_hr"""
                #compute the max of each image in the batch
                max_lr = lr.view(lr.size(0), -1).max(1)[0].view(-1, 1, 1, 1)
                output = output / max_lr
                hr = hr / max_lr
                val_loss += criterion(output, hr).item()
        if min_val_stopping:
            if val_loss/len(valid_loader.loader) < min_loss_val:
                min_loss_val = val_loss/len(valid_loader.loader)
                model_to_save = model.state_dict()
                stopped_epoch = epoch
        train_losses.append(epoch_loss / len(train_loader.loader))
        val_losses.append(val_loss / len(valid_loader.loader))
        print(f"Epoch {epoch+1}/{args.nepochs}, Train Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}")
        writer_tensor.add_scalar('Loss/Train', train_losses[-1], epoch)
        writer_tensor.add_scalar('Loss/Validation', val_losses[-1], epoch)
        scheduler.step(val_losses[-1])

    writer_tensor.close()
    if min_val_stopping:
        if stopped_epoch != args.nepochs - 1:
            print(f"Training stopped at epoch {stopped_epoch+1} with minimum validation loss: {min_loss_val}")
        else:
            print("Minimum of validation not reached during training.")

    try:
        if min_val_stopping and model_to_save is not None:
            torch.save(model_to_save, os.path.join(args.save_dir, f"{args.save_prefix}_{model_name}_updated_hyperspectral_model.pth"))
            model.load_state_dict(model_to_save)
        else:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.save_prefix}_{model_name}_updated_hyperspectral_model.pth"))
        print('Model saved successfully.')
    except Exception as e:
        print(f"Error saving model: {e}")
    
    try:
        with open(os.path.join(args.save_dir, f"{args.save_prefix}_{model_name}_updated_losses.csv"), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
            for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
                writer.writerow([epoch, train_loss, val_loss])
        print('Loss file saved successfully.')
    except Exception as e:
        print(f"Error saving loss file: {e}")
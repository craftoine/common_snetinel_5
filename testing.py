import torch
import numpy as np
from scipy.ndimage import sobel
import torch.nn as nn
from sklearn.decomposition import PCA
import os
from archithectures import S5_DSCR, S5_DSCR_S
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import csv
from archi_com import *


def psnr_gpu(sr, hr_img, device, data_range=1):
    if len(sr.shape)>=3:
        #print(sr.shape, hr_img.shape)
        sr = sr.view(-1, sr.size(-2), sr.size(-1))
        hr_img = hr_img.view(-1, hr_img.size(-2), hr_img.size(-1))
        #comput the mse on the las 2 coordinates and not the batch one
        mse = torch.mean((sr -hr_img) ** 2, dim=(-2,-1))
        #print(mse.shape)
        #return the mean psnr over the batch
        return torch.mean(10* torch.log10(torch.tensor(data_range).to(device)**2 /mse))
    else:
        mse = torch.mean((sr -hr_img) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * torch.log10(torch.tensor(data_range).to(device)**2 / mse)
def psnr_loss(sr, hr_img):
    #print(sr.shape, hr_img.shape)
    if len(sr.shape)>=3:
        sr = sr.view(-1, sr.size(-2), sr.size(-1))
        hr_img = hr_img.view(-1, hr_img.size(-2), hr_img.size(-1))
        #comput the mse on the las 2 coordinates and not the batch one
        mse = torch.mean((sr -hr_img) ** 2, dim=(-2,-1))
        #return the mean psnr over the batch
        return torch.mean(-(10* torch.log10(1.0 /mse)))
    else:
        mse = nn.MSELoss()(sr, hr_img)
        return -(10* torch.log10(1.0 /mse))
def scc(sr, hr):
    sr = sr.astype(np.float64)
    hr = hr.astype(np.float64)
    
    if sr.ndim == 2:
        sr_lap_x = sobel(sr, axis=1)
        sr_lap_y = sobel(sr, axis=0)
        sr_lap = np.sqrt(sr_lap_x**2 + sr_lap_y**2)

        hr_lap_x = sobel(hr, axis=1)
        hr_lap_y = sobel(hr, axis=0)
        hr_lap = np.sqrt(hr_lap_x**2 + hr_lap_y**2)

        scc_map = (sr_lap * hr_lap) / (np.sqrt(np.sum(sr_lap**2)) * np.sqrt(np.sum(hr_lap**2)))
    else:
        sr_lap = np.zeros(sr.shape)
        hr_lap = np.zeros(hr.shape)
        
        for idim in range(sr.shape[2]):  # Loop over spectral bands
            sr_lap_x = sobel(sr[:, :, idim], axis=1)
            sr_lap_y = sobel(sr[:, :, idim], axis=0)
            sr_lap[:, :, idim] = np.sqrt(sr_lap_x**2 + sr_lap_y**2)

            hr_lap_x = sobel(hr[:, :, idim], axis=1)
            hr_lap_y = sobel(hr[:, :, idim], axis=0)
            hr_lap[:, :, idim] = np.sqrt(hr_lap_x**2 + hr_lap_y**2)

        scc_map = np.sum(sr_lap * hr_lap, axis=2) / (np.sqrt(np.sum(sr_lap**2, axis=2)) * np.sqrt(np.sum(hr_lap**2, axis=2)))
    
    scc_value = np.sum(sr_lap * hr_lap)
    scc_value /= np.sqrt(np.sum(sr_lap**2))
    scc_value /= np.sqrt(np.sum(hr_lap**2))

    return scc_value, scc_map
def calculate_lpips_bandwise(sr, hr, loss_fn_gpu, device):
    lpips_bandwise = []
    num_bands = sr.shape[2]  # Assume shape (H, W, C) for hyperspectral


    for band in range(num_bands):
        """sr_band = torch.tensor(sr[:, :, band]).unsqueeze(0).unsqueeze(0).float()  # Shape (1, 1, H, W)
        hr_band = torch.tensor(hr[:, :, band]).unsqueeze(0).unsqueeze(0).float()"""
        """sr_band = torch.tensor(sr[:, :, band]).unsqueeze(0).unsqueeze(0).float().to(device)  # Shape (1, 1, H, W)
        hr_band = torch.tensor(hr[:, :, band]).unsqueeze(0).unsqueeze(0).float().to(device)"""
        sr_band = sr[:, :, band].unsqueeze(0).unsqueeze(0).float().to(device)
        hr_band = hr[:, :, band].unsqueeze(0).unsqueeze(0).float().to(device)
        

        
        """lpips_value = loss_fn(sr_band, hr_band)"""
        lpips_value = loss_fn_gpu(sr_band, hr_band).detach().cpu()
        lpips_bandwise.append(lpips_value.item())

    return np.mean(lpips_bandwise)


def plot_hyperspectral_images_false_color_global2(lr_img, hr_img, pred_img, idx,network_name,save_dir, bands=[30, 50, 70], cmap='terrain', vmin=None, vmax=None): 

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    lr_color = lr_img[bands, :, :].transpose(1, 2, 0)
    hr_color = hr_img[bands, :, :].transpose(1, 2, 0)
    pred_color = pred_img[bands, :, :].transpose(1, 2, 0)

    lr_color_flat = lr_color.flatten()
    hr_color_flat = hr_color.flatten()
    pred_color_flat = pred_color.flatten()
    all_images = np.concatenate([lr_color_flat, hr_color_flat, pred_color_flat]) 

    global_min = all_images.min()
    global_max = all_images.max()

    lr_color_normalized = (lr_color - global_min) / (global_max - global_min)
    hr_color_normalized = (hr_color - global_min) / (global_max - global_min)
    pred_color_normalized = (pred_color - global_min) / (global_max - global_min)

    lr_vmin, lr_vmax = lr_color_normalized.min(), lr_color_normalized.max()
    hr_vmin, hr_vmax = hr_color_normalized.min(), hr_color_normalized.max()
    pred_vmin, pred_vmax = pred_color_normalized.min(), pred_color_normalized.max()

    
    vmin = min(lr_vmin, hr_vmin, pred_vmin)
    vmax = min(lr_vmax, hr_vmax, pred_vmax)
    
    axes[0].imshow(lr_color_normalized, vmin=vmin, vmax=vmax, cmap=cmap) 
    axes[0].set_title(f'LR Image (Bands {bands})')
    axes[0].axis('off')

    axes[1].imshow(hr_color_normalized, vmin=vmin, vmax=vmax, cmap=cmap) 
    axes[1].set_title(f'HR Image (Bands {bands})')
    axes[1].axis('off')

    axes[2].imshow(pred_color_normalized, vmin=vmin, vmax=vmax, cmap=cmap) 
    axes[2].set_title(f'Predicted Image (Bands {bands})')
    axes[2].axis('off')

    plt.tight_layout()
    output_path = f"{save_dir}/{network_name}_{idx}_global.png"
    #plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    #print(f"Image saved at: {output_path}")
    plt.show()
    plt.close(fig)


def metric_s5net(model, test_loader, device, network_name,loss_fn_lpips_gpu,save_dir, csv_filename= None, plot_hyper = True):
    if csv_filename is None:
        assert False, "Please provide a csv filename to save the results"
    model.eval()
    psnr_values, scc_values = [], []
    ssim_values, lpips_values = [], []
    lr_images, hr_images, sr_images = [], [], []
    ii =0
    with torch.no_grad():
        mean,std = test_loader.mean, test_loader.std
        for lr, hr in test_loader.loader:
            lr, hr = lr.to(device), hr.to(device)
            model = model.to(device)  
            output = model(lr,mean = mean, std = std)
            output = nn.ReLU()(output)#clip to [0; infinity[
            max_hr = hr.max().item()
            output = output/max_hr
            hr = hr/max_hr
            lr = lr/max_hr
            for i in range(output.shape[0]):
                sr = output[i].squeeze()
                hr_img = hr[i].squeeze()
                lr_img = lr[i].squeeze()

                #normalise to [0,1] but only using the min and max of the hr image
                """sr = (sr - hr_img.min()) / (hr_img.max() - hr_img.min())
                hr_img = (hr_img - hr_img.min()) / (hr_img.max() - hr_img.min())"""
                

                # LPIPS (normalized to [-1, 1])
                sr_lpips = 2 * sr - 1
                hr_lpips = 2 * hr_img - 1
                lpips_value = calculate_lpips_bandwise(sr_lpips, hr_lpips, loss_fn_lpips_gpu, device)


                
                psnr_value = psnr_gpu(sr, hr_img,device, data_range=1).item()  # [0, 1] range

                #need of numpy array for scc and ssim
                sr = sr.cpu().numpy()
                hr_img = hr_img.cpu().numpy()
                lr_img = lr_img.cpu().numpy()

                scc_value, _ = scc(sr, hr_img)
                ssim_value = ssim(hr_img, sr, data_range=1)  # [0, 1] range
                
                #compute on gpu instead of cpu

                psnr_values.append(psnr_value)
                scc_values.append(scc_value)
                ssim_values.append(ssim_value)
                lpips_values.append(lpips_value)

                lr_images.append(lr_img)
                hr_images.append(hr_img)
                sr_images.append(sr)

                ii= ii+1

                hr_reshaped = hr_img.reshape(hr_img.shape[0], -1).T  
                pca = PCA(n_components=3)
                pca.fit(hr_reshaped)  
                
                hr_pca = pca.transform(hr_reshaped).T.reshape(3, hr_img.shape[1], hr_img.shape[2])  
                lr_reshaped = lr_img.reshape(lr_img.shape[0], -1).T
                lr_pca = pca.transform(lr_reshaped).T.reshape(3, lr_img.shape[1], lr_img.shape[2])  
                sr_reshaped = sr.reshape(sr.shape[0], -1).T
                sr_pca = pca.transform(sr_reshaped).T.reshape(3, sr.shape[1], sr.shape[2])  
                
                lr_pca = (lr_pca - lr_pca.mean()) / lr_pca.std()
                hr_pca = (hr_pca - hr_pca.mean()) / hr_pca.std()
                sr_pca = (sr_pca - sr_pca.mean()) / sr_pca.std()
                if plot_hyper:
                    plot_hyperspectral_images_false_color_global2(lr_pca, hr_pca, sr_pca, ii, network_name, save_dir, bands=[1,0, 2], cmap='viridis')

    avg_psnr = np.mean(psnr_values)
    avg_scc = np.mean(scc_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)

    print(f"PSNR: {avg_psnr:.4f}")
    print(f"SCC: {avg_scc:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"LPIPS: {avg_lpips:.4e}")

    rounded_values = [
    network_name,
    round(avg_psnr, 4),
    round(avg_scc, 4),
    round(avg_ssim, 4),
    f"{avg_lpips:.4e}"]

    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["Network Name", "PSNR", "SCC", "SSIM", "LPIPS"])
        csv_writer.writerow(rounded_values)
    
    return avg_psnr, avg_scc,  avg_ssim, avg_lpips, lr_images, hr_images, sr_images


def S5_DSCR_S_test(params,test_loader,device,num_bands,loss_fn_lpips_gpu,csv_file, correct_relu = True, same_kernel = False,bias=False,plot_hyper = True,compression="no"):
    model = S5_DSCR_S(in_channels=num_bands, 
                            out_channels=497, 
                            num_spectral_bands=num_bands, 
                            depth_multiplier=1, 
                            upsample_scale=4, 
                            mode='convtranspose',
                            correct_relu=correct_relu,
                            same_kernel=same_kernel,
                            bias=bias,
                            compression=compression).to(device)

    model.load_state_dict(torch.load(os.path.join(params.save_dir, f"{params.save_prefix}_DSC2_updated_hyperspectral_model.pth")))

    """avg_psnr, avg_scc, avg_ssim, avg_lpips, lr_images, hr_images, sr_images = metric_s5net(model, test_loader, device, 'S5_DSCR_S',loss_fn_lpips_gpu,params.save_dir,csv_file, plot_hyper=plot_hyper)
    return lr_images, hr_images, sr_images"""
    return generic_testing(model,test_loader,device,'S5_DSCR_S',loss_fn_lpips_gpu,csv_file,params,plot_hyper)


def S5_DSCR_S_test_on_train_set(params,train_loader,device,num_bands,loss_fn_lpips_gpu,csv_file, correct_relu = True, same_kernel = False,bias=False,plot_hyper = True,compression="no"):
    model = S5_DSCR_S(in_channels=num_bands, 
                            out_channels=497, 
                            num_spectral_bands=num_bands, 
                            depth_multiplier=1, 
                            upsample_scale=4, 
                            mode='convtranspose',
                            correct_relu=correct_relu,
                            same_kernel=same_kernel,
                            bias=bias,
                            compression=compression).to(device)

    model.load_state_dict(torch.load(os.path.join(params.save_dir, f"{params.save_prefix}_DSC2_updated_hyperspectral_model.pth")))

    """avg_psnr, avg_scc, avg_ssim, avg_lpips, lr_images, hr_images, sr_images = metric_s5net(model, train_loader, device, 'S5_DSCR_S',loss_fn_lpips_gpu,params.save_dir,csv_file, plot_hyper=plot_hyper)
    return lr_images, hr_images, sr_images"""
    return generic_testing(model,train_loader,device,'S5_DSCR_S_on_train_set',loss_fn_lpips_gpu,csv_file,params,plot_hyper)
def S5_DSCR_S_test_on_val_set(params,valid_loader,device,num_bands,loss_fn_lpips_gpu,csv_file, correct_relu = True, same_kernel = False,bias=False,plot_hyper = True,compression="no"):
    model = S5_DSCR_S(in_channels=num_bands, 
                            out_channels=497, 
                            num_spectral_bands=num_bands, 
                            depth_multiplier=1, 
                            upsample_scale=4, 
                            mode='convtranspose',
                            correct_relu=correct_relu,
                            same_kernel=same_kernel,
                            bias=bias,
                            compression=compression).to(device)

    model.load_state_dict(torch.load(os.path.join(params.save_dir, f"{params.save_prefix}_DSC2_updated_hyperspectral_model.pth")))

    """avg_psnr, avg_scc, avg_ssim, avg_lpips, lr_images, hr_images, sr_images = metric_s5net(model, valid_loader, device, 'S5_DSCR_S',loss_fn_lpips_gpu,params.save_dir,csv_file, plot_hyper=plot_hyper)
    return lr_images, hr_images, sr_images"""
    return generic_testing(model,valid_loader,device,'S5_DSCR_S',loss_fn_lpips_gpu,csv_file,params,plot_hyper)
def S5_DSCR_test(params,test_loader,device,num_bands,loss_fn_lpips_gpu,csv_file, correct_relu = True, same_kernel = False,bias=False,plot_hyper = True,compression="no"):
    model = S5_DSCR(
        in_channels=497,
        out_channels=497,
        num_spectral_bands=497,
        depth_multiplier=3,
        num_layers=5,
        kernel_size=5,
        upsample_scale=4,
        correct_relu=correct_relu,
        same_kernel=same_kernel,
        bias=bias,
        compression=compression)

    model.load_state_dict(torch.load(os.path.join(params.save_dir, f"{params.save_prefix}_DSC_residual2_updated_hyperspectral_model.pth")))
    """avg_psnr, avg_scc, avg_ssim, avg_lpips, lr_images, hr_images, sr_images = metric_s5net(model, test_loader, device,'S5_DSCR',loss_fn_lpips_gpu,params.save_dir,csv_file, plot_hyper=plot_hyper)
    return lr_images, hr_images, sr_images"""
    return generic_testing(model,test_loader,device,'S5_DSCR',loss_fn_lpips_gpu,csv_file,params,plot_hyper)
def S5_DSCR_test_on_train_set(params,train_loader,device,num_bands,loss_fn_lpips_gpu,csv_file, correct_relu = True, same_kernel = False,bias=False,plot_hyper = True,compression="no"):
    model = S5_DSCR(
        in_channels=497,
        out_channels=497,
        num_spectral_bands=497,
        depth_multiplier=3,
        num_layers=5,
        kernel_size=5,
        upsample_scale=4,
        correct_relu=correct_relu,
        same_kernel=same_kernel,
        bias=bias,
        compression=compression)

    model.load_state_dict(torch.load(os.path.join(params.save_dir, f"{params.save_prefix}_DSC_residual2_updated_hyperspectral_model.pth")))
    """avg_psnr, avg_scc, avg_ssim, avg_lpips, lr_images, hr_images, sr_images = metric_s5net(model, train_loader, device,'S5_DSCR_on_train_set',loss_fn_lpips_gpu,params.save_dir,csv_file, plot_hyper=plot_hyper)
    return lr_images, hr_images, sr_images"""
    return generic_testing(model,train_loader,device,'S5_DSCR_on_train_set',loss_fn_lpips_gpu,csv_file,params,plot_hyper)
def S5_DSCR_test_on_val_set(params,valid_loader,device,num_bands,loss_fn_lpips_gpu,csv_file, correct_relu = True, same_kernel = False,bias=False,plot_hyper = True,compression="no"):
    model = S5_DSCR(
        in_channels=497,
        out_channels=497,
        num_spectral_bands=497,
        depth_multiplier=3,
        num_layers=5,
        kernel_size=5,
        upsample_scale=4,
        correct_relu=correct_relu,
        same_kernel=same_kernel,
        bias=bias,
        compression=compression)
    
    model.load_state_dict(torch.load(os.path.join(params.save_dir, f"{params.save_prefix}_DSC_residual2_updated_hyperspectral_model.pth")))

    """avg_psnr, avg_scc, avg_ssim, avg_lpips, lr_images, hr_images, sr_images = metric_s5net(model, valid_loader, device, 'S5_DSCR_on_val_set',loss_fn_lpips_gpu,params.save_dir,csv_file, plot_hyper=plot_hyper)
    return lr_images, hr_images, sr_images"""
    return generic_testing(model,valid_loader,device,'S5_DSCR_on_val_set',loss_fn_lpips_gpu,csv_file,params,plot_hyper)

def bicubic_upsample_test(params,test_loader,device,loss_fn_lpips_gpu,csv_file,plot_hyper = True):
    class BicubicUpsample(nn.Module):
        def __init__(self, scale_factor=4):
            super(BicubicUpsample, self).__init__()
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)

        def forward(self, x,mean=torch.tensor(0.0),std=torch.tensor(1.0)):
            mean, std = mean.to(device), std.to(device)
            x = (x-mean)/std
            x = self.upsample(x)
            x = x * std + mean
            return x

    model = BicubicUpsample(scale_factor=4).to(device)
    """avg_psnr, avg_scc, avg_ssim, avg_lpips, lr_images, hr_images, sr_images = metric_s5net(model, test_loader, device, 'Bicubic_Upsample',loss_fn_lpips_gpu,params.save_dir,csv_file, plot_hyper=plot_hyper)
    return lr_images, hr_images, sr_images"""
    return generic_testing(model,test_loader,device,'Bicubic_Upsample',loss_fn_lpips_gpu,csv_file,params,plot_hyper)
def bicubic_upsample_test_on_train_set(params,train_loader,device,loss_fn_lpips_gpu,csv_file,plot_hyper = True):
    class BicubicUpsample(nn.Module):
        def __init__(self, scale_factor=4):
            super(BicubicUpsample, self).__init__()
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)

        def forward(self, x,mean=torch.tensor(0.0),std=torch.tensor(1.0)):
            mean, std = mean.to(device), std.to(device)
            x = (x-mean)/std
            x = self.upsample(x)
            x = x * std + mean
            return x

    model = BicubicUpsample(scale_factor=4).to(device)
    """avg_psnr, avg_scc, avg_ssim, avg_lpips, lr_images, hr_images, sr_images = metric_s5net(model, train_loader, device, 'Bicubic_Upsample_on_train_set',loss_fn_lpips_gpu,params.save_dir,csv_file, plot_hyper=plot_hyper)
    return lr_images, hr_images, sr_images"""
    return generic_testing(model,train_loader,device,'Bicubic_Upsample_on_train_set',loss_fn_lpips_gpu,csv_file,params,plot_hyper)
def bicubic_upsample_test_on_val_set(params,valid_loader,device,loss_fn_lpips_gpu,csv_file,plot_hyper = True):
    class BicubicUpsample(nn.Module):
        def __init__(self, scale_factor=4):
            super(BicubicUpsample, self).__init__()
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)

        def forward(self, x,mean=torch.tensor(0.0),std=torch.tensor(1.0)):
            mean, std = mean.to(device), std.to(device)
            x = (x-mean)/std
            x = self.upsample(x)
            x = x * std + mean
            return x

    model = BicubicUpsample(scale_factor=4).to(device)
    """avg_psnr, avg_scc, avg_ssim, avg_lpips, lr_images, hr_images, sr_images = metric_s5net(model, valid_loader, device, 'Bicubic_Upsample_on_val_set',loss_fn_lpips_gpu,params.save_dir,csv_file, plot_hyper=plot_hyper)
    return lr_images, hr_images, sr_images"""
    return generic_testing(model,valid_loader,device,'Bicubic_Upsample_on_val_set',loss_fn_lpips_gpu,csv_file,params,plot_hyper)

def generic_testing(model,test_loader,device,network_test_name,loss_fn_lpips_gpu,csv_file,params,plot_hyper = True):
    avg_psnr, avg_scc, avg_ssim, avg_lpips, lr_images, hr_images, sr_images = metric_s5net(model, test_loader, device, network_test_name,loss_fn_lpips_gpu,params.save_dir,csv_file, plot_hyper=plot_hyper)
    return lr_images, hr_images, sr_images
def generic_test_on_train_set(model,train_loader,device,network_name,loss_fn_lpips_gpu,csv_file,params,plot_hyper = True):
    return generic_testing(model,train_loader,device,network_name+'_on_train_set',loss_fn_lpips_gpu,csv_file,params,plot_hyper)
def generic_test_on_val_set(model,valid_loader,device,network_name,loss_fn_lpips_gpu,csv_file,params,plot_hyper = True):
    return generic_testing(model,valid_loader,device,network_name+'_on_val_set',loss_fn_lpips_gpu,csv_file,params,plot_hyper)
def generic_test(model,test_loader,device,network_name,loss_fn_lpips_gpu,csv_file,params,plot_hyper = True):
    return generic_testing(model,test_loader,device,network_name,loss_fn_lpips_gpu,csv_file,params,plot_hyper)

#plot Custom_point_wise_conv weight matrices from ther trained model
def plot_custom_point_wise_conv_weights(model, history=""):
    history += model.__class__.__name__ + "->"
    #recursively go through the model to find all instances of Custom_point_wise_conv
    if isinstance(model, Custom_point_wise_conv):
        print("Found Custom_point_wise_conv in", history)
        model.compression_model.plot_weight()
        return
    if isinstance(model, (nn.Sequential)):
        for layer in model:
            plot_custom_point_wise_conv_weights(layer, history=history)
            return
    #get all the atributes of the model and check if their are list, nn.Module or nn.Sequential
    for attr in dir(model):
        if not attr.startswith('_'):
            #print("found", attr, "in", history)
            module = getattr(model, attr)
            if isinstance(module, nn.ModuleList):
                print("found nn.ModuleList", attr, "in", history)
                for item in module:
                    plot_custom_point_wise_conv_weights(item, history=history+attr+":")
            elif isinstance(module, nn.Sequential):
                print("found nn.Sequential", attr, "in", history)
                for item in module:
                    plot_custom_point_wise_conv_weights(item, history=history+attr+":")
            elif isinstance(module, nn.Module):
                if not isinstance(module, (nn.ReLU, nn.BatchNorm2d, nn.Conv2d, nn.Upsample, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                    print("found nn.Module", attr, "in", history)
                    plot_custom_point_wise_conv_weights(module, history=history+attr+":")
            elif isinstance(module, (list)):
                print("found list", attr, "in", history)
                for item in module:
                    plot_custom_point_wise_conv_weights(item, history=history+attr+":")
    return
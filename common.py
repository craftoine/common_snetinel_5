import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#!pip install torchsummary
from torchsummary import summary
#!pip install tensorboard
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import scipy.io as sio
import os, glob

import csv
import matplotlib.pyplot as plt
import random
#!pip install scikit-image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import sobel
#!pip install lpips
import lpips
from sklearn.decomposition import PCA






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




def compute_global_metrics(lr_files, hr_files):
    all_intensity_values = []

    for lr_file in lr_files:
        corresponding_hr_file = lr_file.replace('_LR4', '')
        if corresponding_hr_file in hr_files:
            lr_image = sio.loadmat(lr_file)['radiance']
            hr_image = sio.loadmat(corresponding_hr_file)['radiance']

            """all_intensity_values.append(lr_image.flatten())
            all_intensity_values.append(hr_image.flatten())"""
            #keep the channel separately
            all_intensity_values.append(lr_image.reshape(-1, lr_image.shape[2]))
            all_intensity_values.append(hr_image.reshape(-1, hr_image.shape[2]))
            print("nb: channels =", lr_image.shape[2])

    all_intensity_values = np.concatenate(all_intensity_values)

    global_min = np.min(all_intensity_values, axis=0)
    global_max = np.max(all_intensity_values, axis=0)
    global_mean = np.mean(all_intensity_values, axis=0)
    global_median = np.median(all_intensity_values, axis=0)
    global_std = np.std(all_intensity_values, axis=0)
    print("shape of all mean, std, min, max, median:", global_mean.shape, global_std.shape, global_min.shape, global_max.shape, global_median.shape)

    return all_intensity_values, global_min, global_max, global_mean, global_median, global_std

def convert_normalise_meanSTD(image, global_mean, global_std):
    min, max = image.min(), image.max()
    diff = max - min
    image = torch.tensor(image, dtype=torch.float32)
    image = (image - global_mean) / global_std
    norm_min, norm_max = image.min().item(), image.max().item()
    return image, min, max, diff, norm_min, norm_max



def extract_patches(image, patch_size, stride=16):
    img_h, img_w, bands = image.shape  
    patch_h, patch_w = patch_size

    patches = []
    for i in range(0, img_h - patch_h + 1, stride):
        for j in range(0, img_w - patch_w + 1, stride):
            patch = image[i:i + patch_h, j:j + patch_w, :]  
            patches.append(patch)
    return np.array(patches)

def load_data_with_patches(data_path, patch_size, BAND, global_mean=None, global_std=None):
    lr_data, hr_data, global_mean, global_std = load_normalise_data(data_path, BAND, global_mean, global_std)

    lr_patches = []
    hr_patches = []
    print(f'LR Data Shape: {lr_data.shape}' )
    print(f'HR Data Shape: {hr_data.shape}')

    for lr_img, hr_img in zip(lr_data, hr_data):
        lr_img_patches = extract_patches(lr_img, patch_size)  # lr_img (spectral bands, H, W)
        hr_img_patches = extract_patches(hr_img, (patch_size[0] * 4, patch_size[1] * 4), stride=64)
        lr_patches.extend(lr_img_patches)
        hr_patches.extend(hr_img_patches)

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    print(f'LR Patch Shape: {lr_patches.shape}')
    print(f'HR Patch Shape: {hr_patches.shape}')
    
    return lr_patches, hr_patches, global_mean, global_std




def load_normalise_data(data_dir, BAND, global_mean=None, global_std=None):
    lr_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('_cropped_hyper_LR4.mat') and BAND in f ])
    hr_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('_cropped_hyper.mat') and BAND in f])

    if global_mean is None or global_std is None:
        all_intensity_values, global_min, global_max, global_mean, global_median, global_std = compute_global_metrics(lr_files, hr_files)
        """
        output_csv = os.path.join(params.save_dir, params.save_prefix + '_global_metrics.csv')
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Global Min", global_min])
            writer.writerow(["Global Max", global_max])
            writer.writerow(["Global Mean", global_mean])
            writer.writerow(["Global Median", global_median])

        #plot_global_histogram(all_intensity_values, global_min, global_max, global_mean, global_median, global_std)
        """    
    print(f"Using Global Mean: {global_mean}, Global Std: {global_std}")

    lr_data = []
    hr_data = []

    for lr_file in lr_files:
        corresponding_hr_file = lr_file.replace('_LR4', '')
        if corresponding_hr_file in hr_files:
            lr_image = sio.loadmat(lr_file)['radiance']
            hr_image = sio.loadmat(corresponding_hr_file)['radiance']

            if len(lr_image.shape) == 2:
                lr_image = lr_image[np.newaxis, :, :]
            if len(hr_image.shape) == 2:
                hr_image = hr_image[:, :, np.newaxis]

            """lr_image, lr_min, lr_max, lr_diff, lr_norm_min, lr_norm_max = convert_normalise_meanSTD(lr_image, global_mean, global_std)
            hr_image, hr_min, hr_max, hr_diff, hr_norm_min, hr_norm_max = convert_normalise_meanSTD(hr_image, global_mean, global_std)"""

            lr_data.append(lr_image)
            hr_data.append(hr_image)

    """lr_data = np.array([img.numpy() for img in lr_data])
    hr_data = np.array([img.numpy() for img in hr_data])        """
    lr_data = np.array(lr_data)
    hr_data = np.array(hr_data)
    return lr_data, hr_data, global_mean, global_std




def apply_mult(x,weight,bias):
    if bias is None:
        return torch.matmul(x.permute(0,2,3,1), weight.squeeze()).permute(0,3,1,2)
    else:
        return torch.matmul(x.permute(0,2,3,1), weight.squeeze()).permute(0,3,1,2) + bias.view(1, -1, 1, 1)
def apply_mult_SVD(x,U,S,bias):
    # x: (batch, in_channels, H, W)
    # U: (in_channels, rank)
    # S: (rank, out_channels)
    x = x.permute(0,2,3,1)
    batch, H, W, in_channels = x.shape
    x = x.reshape(-1, x.size(-1))  # Shape (batch*H*W, in_channels)
    if bias is None:
        #return torch.chain_matmul(x.permute(0,2,3,1), U, S).permute(0,3,1,2)
        return torch.chain_matmul(x, U, S).view(batch, H, W, -1).permute(0,3,1,2)
    else:
        return torch.chain_matmul(x, U, S).view(batch, H, W, -1).permute(0,3,1,2) + bias.view(1, -1, 1, 1)
class weight_no_compression(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(weight_no_compression, self).__init__()
        # Custom weights for pointwise convolution
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels) * 0.01)

    def forward(self,x,bias):
        weight= self.weight
        x = apply_mult(x,weight,bias)
        return x
    def plot_weight(self):
        weight = self.weight.detach().cpu().numpy()
        plt.figure(figsize=(6,5))
        plt.imshow(weight, aspect='equal', cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title('Weight Matrix without Compression')
        plt.tight_layout()
        plt.show()
        #plot the svd decomposition
        U, S, Vt = np.linalg.svd(weight, full_matrices=False)
        #M = (U*S)@Vt
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(U, aspect='auto', cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title('U Matrix')
        plt.subplot(1,2,2)
        plt.imshow(Vt.T, aspect='auto', cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title('Vt.T Matrix')
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(15,5))
        plt.plot(S)
        plt.yscale('log')
        plt.title('Singular Values (log scale)')
        plt.tight_layout()
        plt.show()
        #plot cumulative explained variance
        explained_variance = np.cumsum(S**2) / np.sum(S**2)
        plt.figure(figsize=(15,5))
        plt.plot(explained_variance)
        plt.title('Cumulative Explained Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance')
        plt.grid()
        plt.tight_layout()
        plt.show()


class weight_svd_compression(nn.Module):
    def __init__(self, in_channels, out_channels, rank =20):
        super(weight_svd_compression, self).__init__()
        # Custom weights for pointwise convolution
        #self.U = nn.Parameter(torch.randn(in_channels, rank) * 0.01)
        #self.S = nn.Parameter(torch.randn(rank, out_channels) * 0.01)
        #for torchsummary name of parameters need to be "weight"
        self.weight = nn.Parameter(torch.randn(in_channels*rank+out_channels*rank) * 0.01)
        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.buffered_M = None
    def get_U(self):
        return self.weight[:self.in_channels*self.rank].view(self.in_channels, self.rank)
    def get_S(self):
        return self.weight[self.in_channels*self.rank:].view(self.rank, self.out_channels)
    def forward(self,x,bias):
        #return(self.weight, self.bias)
        #reconstruct the weight
        if self.buffered_M is not None:
            """weight = self.buffered_M
            x = apply_mult(x,weight,bias)
            return x"""
            #using the buffered weight is not as computationally efficient as recomputing it with matrix multiplications in the right order
            x = apply_mult_SVD(x,self.get_U(),self.get_S(),bias)
            return x
        else:
            """U = self.get_U()
            S = self.get_S()
            weight = torch.matmul(U,S)
            x = apply_mult(x,weight,bias)"""
            x = apply_mult_SVD(x,self.get_U(),self.get_S(),bias)
            return x
    def plot_weight(self):
        U = self.get_U().detach().cpu().numpy()
        S = self.get_S().detach().cpu().numpy()
        M = np.matmul(U,S)
        plt.figure(figsize=(10,5))
        plt.subplot(1,3,1)
        plt.imshow(U, aspect='auto', cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title('U Matrix')
        plt.subplot(1,3,2)
        plt.imshow(S, aspect='auto', cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title('S Matrix')
        plt.subplot(1,3,3)
        plt.imshow(M, aspect='equal', cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title('Reconstructed Weight Matrix M=US')
        plt.tight_layout()
        plt.show()

        #compute the true svd of M to plot the singular values
        _, S_, _ = np.linalg.svd(M, full_matrices=False)
        plt.figure(figsize=(15,5))
        plt.plot(S_[:self.rank])
        plt.yscale('log')
        plt.title('Singular Values (log scale)')
        plt.tight_layout()
        plt.show()
    #when model is in eval mode, buffer the weight to avoid recomputing it each time
    def eval(self):
        super().eval()
        if not self.possible_buffer:
            U = self.get_U()
            S = self.get_S()
            self.buffered_M = torch.matmul(U,S)
    #when model is in train mode, unbuffer the weight to recompute it each time
    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.buffered_M = None

class embed_to_values(nn.Module):
    def __init__(self, embd_size):
        super(embed_to_values, self).__init__()
        # Use individual layers instead of nn.Sequential for better torchsummary compatibility
        self.linear1 = nn.Linear(2 * embd_size, 50)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(50, 20)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(20, 1)
        
    def forward(self, embd):
        x = self.linear1(embd)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

class weight_nn_compression(nn.Module):
    #predict matrix value with a small nn
    def __init__(self, in_channels, out_channels):
        super(weight_nn_compression, self).__init__()
        #get the index i,j as two inputs throught an embedding layer
        embd_size = 10
        #self.embedding_= custom_embedding(in_channels, out_channels, embd_size)
        self.weight = nn.Parameter(torch.randn(in_channels + out_channels, embd_size) * 0.01)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_to_values = embed_to_values(embd_size)
    def forward(self,x,bias):
        ins = self.weight[:self.in_channels, :]
        outs = self.weight[self.in_channels:, :]
        embd_i = ins  # Shape (in_channels, embd_size)
        embd_j = outs  # Shape (out_channels, embd_size)
        embd_i_exp = embd_i.unsqueeze(1).expand(-1, self.out_channels, -1)  # Shape (in_channels, out_channels, embd_size)
        embd_j_exp = embd_j.unsqueeze(0).expand(self.in_channels, -1, -1)  # Shape (in_channels, out_channels, embd_size)
        embd = torch.cat((embd_i_exp, embd_j_exp), dim=-1)  # Shape (in_channels, out_channels, 2*embd_size)
        embd = embd.view(-1, embd.size(-1))  # Shape (in_channels*out_channels, 2*embd_size)
        weight = self.emb_to_values(embd).squeeze(-1)  # Shape (in_channels* out_channels)
        weight = weight.view(self.in_channels, self.out_channels)  # Shape (in_channels, out_channels)
        return(apply_mult(x,weight,bias))
    def plot_weight(self):
        ins = self.weight[:self.in_channels, :]
        outs = self.weight[self.in_channels:, :]
        embd_i = ins  # Shape (in_channels, embd_size)
        embd_j = outs  # Shape (out_channels, embd_size)
        embd_i_exp = embd_i.unsqueeze(1).expand(-1, self.out_channels, -1)  # Shape (in_channels, out_channels, embd_size)
        embd_j_exp = embd_j.unsqueeze(0).expand(self.in_channels, -1, -1)  # Shape (in_channels, out_channels, embd_size)
        embd = torch.cat((embd_i_exp, embd_j_exp), dim=-1)  # Shape (in_channels, out_channels, 2*embd_size)
        weight = self.emb_to_values(embd).squeeze(-1).detach().cpu().numpy()  # Shape (in_channels, out_channels)
        plt.figure(figsize=(6,5))
        plt.imshow(weight, aspect='equal', cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title('Weight Matrix from NN Compression')
        plt.tight_layout()
        plt.show()
        #compute the svd of weight to plot the singular values
        _, S_, _ = np.linalg.svd(weight, full_matrices=False)
        plt.figure(figsize=(15,5))
        plt.plot(S_)
        plt.yscale('log')
        plt.title('Singular Values (log scale)')
        plt.tight_layout()
        plt.show()

class Custom_point_wise_conv(nn.Module):
    #redefined conv but with custom parameters
    def __init__(self, in_channels, out_channels, bias=False):
        super(Custom_point_wise_conv, self).__init__()
        # Custom weights for pointwise convolution
        compression = "no"#"no","svd","nn"
        self.compression_model = None
        if compression == "no":
            self.compression_model = weight_no_compression(in_channels, out_channels)
        elif compression == "svd":
            self.compression_model = weight_svd_compression(in_channels, out_channels, rank=20)#20)
        elif compression == "nn":
            self.compression_model = weight_nn_compression(in_channels, out_channels)
            #raise a warning as summary doesn't work well with this model
            from warnings import warn
            warn("Custom_point_wise_conv with nn compression not work well with torchsummary weight counting")
        else:
            raise ValueError("Unknown compression type")
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x):
        #no need of conv because it's a 1x1 conv, use matrix multiplication instead
        return(self.compression_model(x,self.bias))


class DSC(nn.Module):
    def __init__(self, in_channels, out_channels, num_spectral_bands, depth_multiplier=1, upsample_scale=2, mode='bilinear',correct_relu = True,same_kernel = False,bias=False):
        super(DSC, self).__init__()
        self.same_kernel = same_kernel
        if self.same_kernel:
            self.depthwise_conv = nn.Conv2d(in_channels=1, 
                                            out_channels=depth_multiplier,  
                                            kernel_size=3,  
                                            stride=1,
                                            padding=1,
                                            groups=1,#num_spectral_bands,               !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                            bias=bias)
        else:
            self.depthwise_conv = nn.Conv2d(in_channels=num_spectral_bands, 
                                            out_channels=num_spectral_bands * depth_multiplier,  
                                            kernel_size=3,  
                                            stride=1,
                                            padding=1,
                                            groups=num_spectral_bands,  
                                            bias=bias)
        
        """self.pointwise_conv = nn.Conv2d(in_channels=num_spectral_bands * depth_multiplier, 
                                        out_channels=out_channels,  
                                        kernel_size=1,  
                                        bias=bias)"""
        self.pointwise_conv = Custom_point_wise_conv(in_channels=num_spectral_bands * depth_multiplier,
                                                    out_channels=out_channels,
                                                    bias=bias)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.correct_relu = correct_relu
        if not(self.correct_relu):
            self.relu = nn.ReLU()
        
    def forward(self, x):
        #print("dfjnrgfjk")
        # shape (batch_size, num_spectral_bands, height, width)
        #!!!!!!!!!!!!!!!!!!!!!!!
        if self.same_kernel:
            shape_0,shape_1 = x.shape[0], x.shape[1]
            x = x.view(-1, 1, x.size(2), x.size(3))  # Ensure correct shape"""
        x = self.depthwise_conv(x)
        if self.same_kernel:
            x = x.view(shape_0, shape_1 * x.size(1), x.size(2), x.size(3))  # Reshape back
        x = self.pointwise_conv(x)
        x = self.bn(x)
        if not(self.correct_relu):
            x = self.relu(x)
        return x


class S5_DSCR_S(nn.Module):
    def __init__(self, in_channels, out_channels, num_spectral_bands, depth_multiplier=1, upsample_scale=2, mode='bilinear',correct_relu = True,same_kernel = False,bias=False):
        super(S5_DSCR_S, self).__init__()
        self.interpolation = nn.Upsample(scale_factor=upsample_scale, mode='bicubic', align_corners=False)
        self.dsc_block = DSC(in_channels, out_channels, num_spectral_bands, depth_multiplier, correct_relu=correct_relu, same_kernel=same_kernel, bias=bias)
        self.relu = nn.ReLU()
    def forward(self, x, target_size=None,mean=torch.tensor(0.0), std=torch.tensor(1.0)):
        mean, std = mean.to(x.device), std.to(x.device)
        x = (x-mean)/std
        interpolated = self.interpolation(x)
        refined = self.dsc_block(interpolated)
        
        if target_size is not None:
            interpolated = F.interpolate(interpolated, size=target_size, mode='bicubic', align_corners=False)
            refined = F.interpolate(refined, size=target_size, mode='bicubic', align_corners=False)
        else:
            refined = F.interpolate(refined, size=interpolated.shape[2:], mode='bicubic', align_corners=False)
        output = refined + interpolated
        output = output * std + mean
        output = nn.ReLU()(output)
        return output
    


class ReshapeLayer(nn.Module):
    def __init__(self, target):
        super(ReshapeLayer, self).__init__()
        self.target = target

    def forward(self, x):
        return x.view(-1,self.target, x.size(2), x.size(3))
class ImprovedDSC_2(nn.Module):
    def __init__(self, in_channels, out_channels, num_spectral_bands, depth_multiplier=1, num_layers=3, kernel_size=3, correct_relu = True, same_kernel = False,bias=False):
        super(ImprovedDSC_2, self).__init__()
        
        layers = []
        for ly in range(num_layers):
            if same_kernel:
                depthwise_conv = nn.Conv2d(
                    in_channels=1,  
                    out_channels= depth_multiplier,  
                    kernel_size=kernel_size,  
                    stride=1,
                    padding=kernel_size // 2, 
                    groups=1,
                    bias=bias
                )
                layers.append(ReshapeLayer(1))  # Reshape before depthwise conv
                layers.append(depthwise_conv)
                layers.append(ReshapeLayer(num_spectral_bands * depth_multiplier))
            else:
                depthwise_conv = nn.Conv2d(
                    in_channels=num_spectral_bands,  
                    out_channels=num_spectral_bands * depth_multiplier,  
                    kernel_size=kernel_size,  
                    stride=1,
                    padding=kernel_size // 2, 
                    groups=num_spectral_bands,  
                    bias=bias
                )
                layers.append(depthwise_conv)
            """if correct_relu:
                layers.append(nn.ReLU())#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!test"""
            """pointwise_conv = nn.Conv2d(
                in_channels=num_spectral_bands * depth_multiplier,  
                out_channels=out_channels,  
                kernel_size=1,  
                bias=bias
            )"""
            pointwise_conv = Custom_point_wise_conv(
                in_channels=num_spectral_bands * depth_multiplier,
                out_channels=out_channels,
                bias=bias
            )
            layers.append(pointwise_conv)
            layers.append(nn.BatchNorm2d(out_channels))
            if ly != num_layers - 1 or not(correct_relu):  # No ReLU after the last layer
                layers.append(nn.ReLU())

        self.conv_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.conv_layers(x)


class S5_DSCR(nn.Module):
    def __init__(self, in_channels, out_channels, num_spectral_bands, depth_multiplier=1, num_layers=3, kernel_size=3, upsample_scale=2, correct_relu = True, same_kernel = False,bias=False):
        super(S5_DSCR, self).__init__()
        self.interpolation = nn.Upsample(scale_factor=upsample_scale, mode='bicubic', align_corners=False)
        self.dsc_block = ImprovedDSC_2(in_channels, out_channels, num_spectral_bands, depth_multiplier, num_layers, kernel_size,correct_relu=correct_relu,same_kernel=same_kernel,bias=bias)

    def forward(self, x, mean=torch.tensor(0), std=torch.tensor(1)):
        mean,std = mean.to(x.device), std.to(x.device)
        x = (x-mean)/std
        interpolated = self.interpolation(x)
        refined = self.dsc_block(interpolated)
        output = refined + interpolated
        output = output * std + mean
        output = nn.ReLU()(output)
        return output

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

def metric_s5net(model, test_loader, device, network_name,loss_fn_lpips_gpu,save_dir, csv_filename= None):
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


class CustomLoader:
    def __init__(self, data, mean,std, *args, **kwargs):
        self.loader = DataLoader(data, *args, **kwargs)
        """self.mean = mean
        self.std = std"""
        """self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)"""
        self.mean = torch.tensor(mean,dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(std,dtype=torch.float32).view(1, -1, 1, 1)


def S5_DSCR_S_train(args,train_loader,valid_loader,num_bands,device,correct_relu = True,same_kernel = False, bias = False):
    model = S5_DSCR_S(in_channels=num_bands, 
                            out_channels=497, 
                            num_spectral_bands=num_bands, 
                            depth_multiplier=1, 
                            upsample_scale=4, 
                            mode='convtranspose',
                            correct_relu=correct_relu,
                            same_kernel=same_kernel,
                            bias=bias).to(device)  
    
    model = model.to(device)     
    summary(model, input_size=(num_bands, 64, 64))

    log_dir = os.path.join(args.save_dir, args.save_prefix, 'DSC')
    writer_tensor = SummaryWriter(log_dir=log_dir)

    criterion = nn.L1Loss() if args.net_loss == 'L1norm' else ( psnr_loss  if args.net_loss == 'PSNR' else nn.MSELoss() )
    optimizer = optim.Adam(model.parameters(), lr=args.net_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    train_losses, val_losses = [], []
    
    for epoch in range(args.nepochs):
        model.train()
        epoch_loss = 0
        mean, std = train_loader.mean, train_loader.std
        for batch_idx, (lr, hr) in enumerate(train_loader.loader):
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            output = model(lr, mean=mean, std=std)
            #normalize output and hr to [0, 1] based on hr min and max on each datapoint and each channel
            #shape of output and hr is (batch_size,channels, height, width)
            max_hr = hr.max()
            output = output / max_hr
            hr = hr / max_hr
            
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
                output = model(lr, mean=mean, std=std)
                #normalize output and hr to [0, 1] based on hr min and max on each datapoint and each channel
                max_hr = hr.max()
                output = output / max_hr
                hr = hr / max_hr
                val_loss += criterion(output, hr).item()

        train_losses.append(epoch_loss / len(train_loader.loader))
        val_losses.append(val_loss / len(valid_loader.loader))
        
        print(f"Epoch {epoch+1}/{args.nepochs}, Train Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}")
        writer_tensor.add_scalar('Loss/Train', train_losses[-1], epoch)
        writer_tensor.add_scalar('Loss/Validation', val_losses[-1], epoch)
        scheduler.step(val_losses[-1])

    writer_tensor.close()

    try:
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.save_prefix}_DSC2_updated_hyperspectral_model.pth"))
        print('Model saved successfully.')
    except Exception as e:
        print(f"Error saving model: {e}")
    
    try:
        with open(os.path.join(args.save_dir, f"{args.save_prefix}_DSC2_updated_losses.csv"), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
            for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
                writer.writerow([epoch, train_loss, val_loss])
        print('Loss file saved successfully.')
    except Exception as e:
        print(f"Error saving loss file: {e}")



def S5_DSCR_train(args,train_loader,valid_loader,num_bands,device,correct_relu = True, same_kernel = False, bias = False):
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
        bias=bias).to(device) 

    model = model.to(device) 
    summary(model, input_size=(num_bands, 64, 64))

    log_dir = os.path.join(args.save_dir, args.save_prefix, 'DSC_residual2')
    writer_tensor = SummaryWriter(log_dir=log_dir)

    criterion = nn.L1Loss() if args.net_loss == 'L1norm' else ( psnr_loss  if args.net_loss == 'PSNR' else nn.MSELoss() )
    optimizer = optim.Adam(model.parameters(), lr=args.net_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    train_losses, val_losses = [], []

    
    for epoch in range(args.nepochs):
        model.train()
        epoch_loss = 0
        mean, std = train_loader.mean, train_loader.std
        for batch_idx, (lr, hr) in enumerate(train_loader.loader):
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            output = model(lr,mean = mean, std=std)
            #normalize output and hr to [0, 1] based on hr min and max on each datapoint and each channel
            #shape of output and hr is (batch_size,channels, height, width)
            max_hr = hr.max()
            output = output / max_hr
            hr = hr / max_hr
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
                output = model(lr, mean=mean, std=std)
                #normalize output and hr to [0, 1] based on hr min and max on each datapoint and each channel
                max_hr = hr.max()
                output = output / max_hr
                hr = hr / max_hr
                val_loss += criterion(output, hr).item()

        train_losses.append(epoch_loss / len(train_loader.loader))
        val_losses.append(val_loss / len(valid_loader.loader))
        
        print(f"Epoch {epoch+1}/{args.nepochs}, Train Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}")
        writer_tensor.add_scalar('Loss/Train', train_losses[-1], epoch)
        writer_tensor.add_scalar('Loss/Validation', val_losses[-1], epoch)
        scheduler.step(val_losses[-1])

    writer_tensor.close()

    try:
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.save_prefix}_DSC_residual2_updated_hyperspectral_model.pth"))
        print('Model saved successfully.')
    except Exception as e:
        print(f"Error saving model: {e}")
    
    try:
        with open(os.path.join(args.save_dir, f"{args.save_prefix}_DSC_residual2_updated_losses.csv"), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
            for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
                writer.writerow([epoch, train_loss, val_loss])
        print('Loss file saved successfully.')
    except Exception as e:
        print(f"Error saving loss file: {e}")



def S5_DSCR_S_test(params,test_loader,device,num_bands,loss_fn_lpips_gpu,csv_file, correct_relu = True, same_kernel = False,bias=False):
    model = S5_DSCR_S(in_channels=num_bands, 
                            out_channels=497, 
                            num_spectral_bands=num_bands, 
                            depth_multiplier=1, 
                            upsample_scale=4, 
                            mode='convtranspose',
                            correct_relu=correct_relu,
                            same_kernel=same_kernel,
                            bias=bias).to(device)  
    
    model.load_state_dict(torch.load(os.path.join(params.save_dir, f"{params.save_prefix}_DSC2_updated_hyperspectral_model.pth")))

    avg_psnr, avg_scc, avg_ssim, avg_lpips, lr_images, hr_images, sr_images = metric_s5net(model, test_loader, device, 'S5_DSCR_S',loss_fn_lpips_gpu,params.save_dir,csv_file)
    return lr_images, hr_images, sr_images


def S5_DSCR_S_test_on_train_set(params,train_loader,device,num_bands,loss_fn_lpips_gpu,csv_file, correct_relu = True, same_kernel = False,bias=False):
    model = S5_DSCR_S(in_channels=num_bands, 
                            out_channels=497, 
                            num_spectral_bands=num_bands, 
                            depth_multiplier=1, 
                            upsample_scale=4, 
                            mode='convtranspose',
                            correct_relu=correct_relu,
                            same_kernel=same_kernel,
                            bias=bias).to(device)  
    
    model.load_state_dict(torch.load(os.path.join(params.save_dir, f"{params.save_prefix}_DSC2_updated_hyperspectral_model.pth")))

    avg_psnr, avg_scc, avg_ssim, avg_lpips, lr_images, hr_images, sr_images = metric_s5net(model, train_loader, device, 'S5_DSCR_S',loss_fn_lpips_gpu,params.save_dir,csv_file)
    return lr_images, hr_images, sr_images
def S5_DSCR_S_test_on_val_set(params,valid_loader,device,num_bands,loss_fn_lpips_gpu,csv_file, correct_relu = True, same_kernel = False,bias=False):
    model = S5_DSCR_S(in_channels=num_bands, 
                            out_channels=497, 
                            num_spectral_bands=num_bands, 
                            depth_multiplier=1, 
                            upsample_scale=4, 
                            mode='convtranspose',
                            correct_relu=correct_relu,
                            same_kernel=same_kernel,
                            bias=bias).to(device)  
    
    model.load_state_dict(torch.load(os.path.join(params.save_dir, f"{params.save_prefix}_DSC2_updated_hyperspectral_model.pth")))

    avg_psnr, avg_scc, avg_ssim, avg_lpips, lr_images, hr_images, sr_images = metric_s5net(model, valid_loader, device, 'S5_DSCR_S',loss_fn_lpips_gpu,params.save_dir,csv_file)
    return lr_images, hr_images, sr_images
def S5_DSCR_test(params,test_loader,device,num_bands,loss_fn_lpips_gpu,csv_file, correct_relu = True, same_kernel = False,bias=False):
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
        bias=bias)

    model.load_state_dict(torch.load(os.path.join(params.save_dir, f"{params.save_prefix}_DSC_residual2_updated_hyperspectral_model.pth")))
    avg_psnr, avg_scc, avg_ssim, avg_lpips, lr_images, hr_images, sr_images = metric_s5net(model, test_loader, device,'S5_DSCR',loss_fn_lpips_gpu,params.save_dir,csv_file)
    return lr_images, hr_images, sr_images
def S5_DSCR_test_on_train_set(params,train_loader,device,num_bands,loss_fn_lpips_gpu,csv_file, correct_relu = True, same_kernel = False,bias=False):
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
        bias=bias)

    model.load_state_dict(torch.load(os.path.join(params.save_dir, f"{params.save_prefix}_DSC_residual2_updated_hyperspectral_model.pth")))
    avg_psnr, avg_scc, avg_ssim, avg_lpips, lr_images, hr_images, sr_images = metric_s5net(model, train_loader, device,'S5_DSCR_on_train_set',loss_fn_lpips_gpu,params.save_dir,csv_file)
    return lr_images, hr_images, sr_images
def S5_DSCR_S_test_on_val_set(params,valid_loader,device,num_bands,loss_fn_lpips_gpu,csv_file, correct_relu = True, same_kernel = False,bias=False):
    model = S5_DSCR_S(in_channels=num_bands, 
                            out_channels=497, 
                            num_spectral_bands=num_bands, 
                            depth_multiplier=1, 
                            upsample_scale=4, 
                            mode='convtranspose',
                            correct_relu=correct_relu,
                            same_kernel=same_kernel,
                            bias=bias).to(device)  
    
    model.load_state_dict(torch.load(os.path.join(params.save_dir, f"{params.save_prefix}_DSC2_updated_hyperspectral_model.pth")))

    avg_psnr, avg_scc, avg_ssim, avg_lpips, lr_images, hr_images, sr_images = metric_s5net(model, valid_loader, device, 'S5_DSCR_S_on_val_set',loss_fn_lpips_gpu,params.save_dir,csv_file)
    return lr_images, hr_images, sr_images

def bicubic_upsample_test(params,test_loader,device,loss_fn_lpips_gpu,csv_file):
    class BicubicUpsample(nn.Module):
        def __init__(self, scale_factor=4):
            super(BicubicUpsample, self).__init__()
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)

        def forward(self, x,mean=torch.tensor(0.0),std=torch.tensor(1.0)):
            mean, std = mean.to(device), std.to(device)
            x = (x-mean)/std
            x = self.upsample(x)
            x = x * std + mean
            x = nn.ReLU()(x)
            return x

    model = BicubicUpsample(scale_factor=4).to(device)
    avg_psnr, avg_scc, avg_ssim, avg_lpips, lr_images, hr_images, sr_images = metric_s5net(model, test_loader, device, 'Bicubic_Upsample',loss_fn_lpips_gpu,params.save_dir,csv_file)
    return lr_images, hr_images, sr_images
def bicubic_upsample_test_on_train_set(params,train_loader,device,loss_fn_lpips_gpu,csv_file):
    class BicubicUpsample(nn.Module):
        def __init__(self, scale_factor=4):
            super(BicubicUpsample, self).__init__()
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)

        def forward(self, x,mean=torch.tensor(0.0),std=torch.tensor(1.0)):
            mean, std = mean.to(device), std.to(device)
            x = (x-mean)/std
            x = self.upsample(x)
            x = x * std + mean
            x = nn.ReLU()(x)
            return x

    model = BicubicUpsample(scale_factor=4).to(device)
    avg_psnr, avg_scc, avg_ssim, avg_lpips, lr_images, hr_images, sr_images = metric_s5net(model, train_loader, device, 'Bicubic_Upsample_on_train_set',loss_fn_lpips_gpu,params.save_dir,csv_file)
    return lr_images, hr_images, sr_images
def bicubic_upsample_test_on_val_set(params,valid_loader,device,loss_fn_lpips_gpu,csv_file):
    class BicubicUpsample(nn.Module):
        def __init__(self, scale_factor=4):
            super(BicubicUpsample, self).__init__()
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=False)

        def forward(self, x,mean=torch.tensor(0.0),std=torch.tensor(1.0)):
            mean, std = mean.to(device), std.to(device)
            x = (x-mean)/std
            x = self.upsample(x)
            x = x * std + mean
            x = nn.ReLU()(x)
            return x

    model = BicubicUpsample(scale_factor=4).to(device)
    avg_psnr, avg_scc, avg_ssim, avg_lpips, lr_images, hr_images, sr_images = metric_s5net(model, valid_loader, device, 'Bicubic_Upsample_on_val_set',loss_fn_lpips_gpu,params.save_dir,csv_file)
    return lr_images, hr_images, sr_images
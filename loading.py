import scipy.io as sio
import numpy as np
import torch
import os



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

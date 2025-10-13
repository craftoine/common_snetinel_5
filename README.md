usage:

devise should be defined

loss_fn_lpips = lpips.LPIPS(net='alex')
loss_fn_lpips_gpu = loss_fn_lpips.to(device)

csv_file = os.path.join(args.save_dir, args.save_prefix, "results.csv")






args = params


lr_patches, hr_patches, global_mean, global_std  = load_data_with_patches(args.ftrain, (64, 64), 'BAND4')


lr_patches = lr_patches.transpose(0, 3, 1, 2)  # to (batch_size, channels, height, width)
hr_patches = hr_patches.transpose(0, 3, 1, 2)

train_data = [(torch.tensor(lr, dtype=torch.float32), torch.tensor(hr, dtype=torch.float32)) for lr, hr in zip(lr_patches, hr_patches)]
#train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
train_loader = CustomLoader(train_data, global_mean, global_std, batch_size=args.batch_size, shuffle=True)
    
num_bands = lr_patches.shape[1]  # num of spectral channels





lr_patches, hr_patches,_ , _  = load_data_with_patches(params.fvalid, (64, 64), 'BAND4', global_mean=global_mean, global_std=global_std)
    
lr_patches = lr_patches.transpose(0, 3, 1, 2)  
hr_patches = hr_patches.transpose(0, 3, 1, 2)

valid_data = [(torch.tensor(lr, dtype=torch.float32), torch.tensor(hr, dtype=torch.float32)) for lr, hr in zip(lr_patches, hr_patches)]
#valid_loader = DataLoader(valid_data, batch_size=params.batch_size, shuffle=True)
valid_loader = CustomLoader(valid_data, global_mean, global_std, batch_size=params.batch_size, shuffle=True)

num_bands = lr_patches.shape[1]





lr_patches, hr_patches,_ , _  = load_data_with_patches(params.ftest, (64, 64), 'BAND4', global_mean=global_mean, global_std=global_std)
    
lr_patches = lr_patches.transpose(0, 3, 1, 2)  
hr_patches = hr_patches.transpose(0, 3, 1, 2)

test_data = [(torch.tensor(lr, dtype=torch.float32), torch.tensor(hr, dtype=torch.float32)) for lr, hr in zip(lr_patches, hr_patches)]
#test_loader = DataLoader(test_data, batch_size=params.batch_size, shuffle=False)
test_loader = CustomLoader(test_data, global_mean, global_std, batch_size=params.batch_size, shuffle=False)

num_bands = lr_patches.shape[1]





bias = False
same_kernel = False
correct_relu = True


example of training and testing:

S5_DSCR_train(params,train_loader,valid_loader,num_bands,device,correct_relu, same_kernel, bias)
_,_,_ = S5_DSCR_S_test_on_train_set(params,train_loader,device,num_bands,loss_fn_lpips_gpu,csv_file, correct_relu, same_kernel,bias)


import torchvision.transforms as transforms

# # once the images are loaded, how do we pre-process them before being passed into the network
# # by default, we resize the images to 64 x 64 in size
# # and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from ImageNet

data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        #mean=[0.485, 0.456, 0.406],
        #std=[0.229, 0.224, 0.225]
        mean = 0.9822,
        std = 0.0538
    )
])


data_transforms_densenet = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    #transforms.Grayscale(1),
    transforms.Normalize(
        mean = 0.9767,
        std = 0.1078
    )
])


data_transforms_resnet = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    #transforms.Grayscale(1),
    transforms.Normalize(
        mean = 0.9767,
        std = 0.1078
    )
])

data_transforms_resnet50 = transforms.Compose([
    transforms.Resize((232,232)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    #transforms.Grayscale(1),
    transforms.Normalize(
        mean = 0.9809,
        std = 0.0949
    )
])
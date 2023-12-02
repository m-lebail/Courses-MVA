import torchvision.transforms as transforms

# # once the images are loaded, how do we pre-process them before being passed into the network
# # by default, we resize the images to 64 x 64 in size
# # and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from ImageNet

data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


data_transforms_resnet = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop((224,224)),
    #transforms.Resize((224,224)),
    transforms.ToTensor(),
    #transforms.Grayscale(1),
    transforms.Normalize(
        mean = 0.485,
        std = 0.229
    )
])

data_transforms_resnet_rotation_flip = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
    transforms.RandomRotation(degrees=30),  # Randomly rotate by up to 15 degrees
    transforms.Resize((256,256)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    #transforms.Grayscale(1),
    transforms.Normalize(
        mean = 0.485,
        std = 0.229
    )
])



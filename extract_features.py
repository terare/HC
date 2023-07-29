'''
This is a program that inputs Imagenet test data to encoder 
and calculates the average vector of features obtained for each class.
'''
import os
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)
import pickle
import argparse
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import clip
from pytorch_pretrained_vit import ViT
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


parser = argparse.ArgumentParser()
parser.add_argument('-m','--model',choices=['clip','vit','wrn'],default='clip')
args = parser.parse_args()


if __name__ == '__main__':

    # Specify GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    
    # Specify model architecture and preprocessing
    if args.model == 'clip':
        model, transform = clip.load("ViT-B/32", device=device)
        save_file = 'features/clip.pickle'
    elif args.model == 'vit':
        model = ViT('B_32_imagenet1k', pretrained=True).to(device)
        transform = transforms.Compose([
            transforms.Resize([384,384]),
            transforms.ToTensor(),
            transforms.Normalize(0.5,0.5)
        ])
        save_file = 'features/vit.pickle'
    elif args.model == 'wrn':
        model = timm.create_model('wide_resnet101_2',pretrained=True).to(device)
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        save_file = 'features/wrn.pickle'
    else:
        print('Warning: The specified model does not exist.')
    model.eval()

    # Create a dataset(test)
    valroot = './../../imagenet-1k/ILSVRC2012_img_val_for_ImageFolder'
    valset = ImageFolder(root=valroot,transform=transform)

    # Imagenet-1k consists of 1000 classes of data, and its validation dataset contains 50 images for each class.
    num_class = 1000
    class_size = 50
    
    # Extract features and take the average for each class.
    features = [0]*num_class
    for i in tqdm(range(len(valset))):
        image,label = valset[i]
        image = image.unsqueeze(0).to(device)
        if args.model == 'clip':
            output = model.encode_image(image)
        else:
            output = model(image)
        output = output.view(-1).to('cpu').detach().numpy().copy()
        features[label] += output/class_size

    # Save features
    if not os.path.isdir('features'):
        os.makedirs('features')
    f = open(save_file,'wb')
    pickle.dump(features,f)
    
    print("Done!")
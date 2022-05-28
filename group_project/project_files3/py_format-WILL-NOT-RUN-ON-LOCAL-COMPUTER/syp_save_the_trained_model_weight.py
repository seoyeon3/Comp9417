#!/usr/bin/env python
# coding: utf-8
##################################################################
# This code saves the trained model's weight so as to reuse later.
##################################################################
#################################
# access to google drive
#################################
from google.colab import drive

drive.mount('/content/drive')

#################################
# import ViT pretrained model
#################################
get_ipython().system('pip install pytorch_pretrained_vit')

#################################
# import modules
#################################
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
import pandas as pd
from torchvision.io import read_image, ImageReadMode
import numpy as np

from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy

from pytorch_pretrained_vit import ViT
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


#################################
# function for model training
#################################
def train_model(model, dataloaders, criterion, optimizer, num_epochs, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    #seoyeon added
    best_epoch = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc

                #seoyeon added
                best_epoch = epoch
                #min_loss = min(val_loss_array)
                #min_loss_epoch = val_loss_array.index(min_loss)

                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    #seoyeon added
    print("Best epoch: {}".format(best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


#################################
# function to set the requires_grad parameter
#################################
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


#################################
# function to get the model from pytorch models
#################################
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299
        
    # *************
    elif model_name == "vit":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        # config = dict(hidden_size=512, num_heads=8, num_layers=6)
        # model = ViT.from_config(config)
        model_ft = ViT('B_16_imagenet1k', pretrained=use_pretrained)

        # model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # # Handle the auxilary net
        # num_ftrs = model_ft.AuxLogits.fc.in_features
        # model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 384

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size


#################################
# training with samples_x384_all1: 
# 383x383 sized jpg including all samples but random selection among train, valid and test
#################################
data_dir = "/content/drive/MyDrive/shared/samples_x384_all"
model_name = "vit"
num_classes = 4
num_epochs = 20
feature_extract = False
batch_size = 8


#################################
# Initialize the model for this run
#################################
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
# Print the model we just instantiated
#print(model_ft)

#################################
# Data augmentation and normalization for training
# Just normalization for validation
#################################
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#################################
# Create training and validation datasets
# Just normalization for validation
#################################
#print("Initializing Datasets and Dataloaders...")

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}
#image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid', 'test']}
#dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid']}

#################################
# Send the model to GPU
#################################
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Send the model to GPU
model_ft = model_ft.to(device)

#################################
# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
#################################
params_to_update = model_ft.parameters()
#print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            #print("\t",name)
'''
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            #print("\t",name)
'''

#################################
# Observe that all parameters are being optimized
#################################

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

#################################
# Setup the loss fxn
#################################

criterion = nn.CrossEntropyLoss()

#################################
# Train and evaluate
#################################

model1, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))


#################################
# save the weight of this ViT model
#################################
PATH = '/content/drive/MyDrive/shared/save_models3/vit1.pth'

torch.save(model1.state_dict(), PATH)

model1_ = torch.load(PATH)
model1_


##################################################################
# training with samples_x384_all1: 
# 383x383 sized jpg including all samples but random selection among train, valid and test
##################################################################
data_dir = "/content/drive/MyDrive/shared/samples_x384_all_2"
model_name = "vit"
num_classes = 4
num_epochs = 20
feature_extract = False
batch_size = 8

#tensor([0.8476, 0.6748, 0.7568])
#tensor([0.1283, 0.2436, 0.1882])

#################################
# Initialize the model for this run
#################################
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
# Print the model we just instantiated
#print(model_ft)

#################################
# Data augmentation and normalization for training
# Just normalization for validation
#################################
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#print("Initializing Datasets and Dataloaders...")

#################################
# Create training and validation datasets
# Just normalization for validation
#################################

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}
#image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid', 'test']}
#dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid']}

#################################
# Send the model to GPU
#################################
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Send the model to GPU
model_ft = model_ft.to(device)

#################################
# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
#################################
params_to_update = model_ft.parameters()
#print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            #print("\t",name)
'''
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            #print("\t",name)
'''

#################################
# Observe that all parameters are being optimized
#################################

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

#################################
# Setup the loss fxn
#################################

criterion = nn.CrossEntropyLoss()

#################################
# Train and evaluate
#################################

model2, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))


#################################
# save the weight of this ViT model
#################################
PATH = '/content/drive/MyDrive/shared/save_models3/vit2.pth'
torch.save(model2.state_dict(), PATH)

model2_ = torch.load(PATH)
model2_


##################################################################
# training with samples_x384_all3: 
# 383x383 sized jpg including all samples but random selection among train, valid and test
##################################################################
data_dir = "/content/drive/MyDrive/shared/samples_x384_all_3"
model_name = "vit"
num_classes = 4
num_epochs = 20
feature_extract = False
batch_size = 8

#tensor([0.8484, 0.6749, 0.7561])
#tensor([0.1275, 0.2433, 0.1892])

#################################
# Initialize the model for this run
#################################
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
# Print the model we just instantiated
#print(model_ft)

#################################
# Data augmentation and normalization for training
# Just normalization for validation
#################################
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#print("Initializing Datasets and Dataloaders...")

#################################
# Create training and validation datasets
# Just normalization for validation
#################################
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}
#image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid', 'test']}
#dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid']}

#################################
# Send the model to GPU
#################################
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Send the model to GPU
model_ft = model_ft.to(device)

#################################
# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
#################################
params_to_update = model_ft.parameters()
#print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            #print("\t",name)
'''
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            #print("\t",name)
'''

#################################
# Observe that all parameters are being optimized
#################################

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

#################################
# Setup the loss fxn
#################################

criterion = nn.CrossEntropyLoss()

#################################
# Train and evaluate
#################################

model3, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

#################################
# save the weight of this ViT model
#################################
PATH = '/content/drive/MyDrive/shared/save_models3/vit3.pth'
torch.save(model3.state_dict(), PATH)
model3_ = torch.load(PATH)
model3_


##################################################################
# training with samples_x384_all1: 
# 383x383 sized jpg including all samples but random selection among train, valid and test
##################################################################
data_dir = "/content/drive/MyDrive/shared/samples_x384_all_4"
model_name = "vit"
num_classes = 4
num_epochs = 20
feature_extract = False
batch_size = 8

#tensor([0.8481, 0.6741, 0.7554])
#tensor([0.1284, 0.2439, 0.1896])

#################################
# Initialize the model for this run
#################################
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
# Print the model we just instantiated
#print(model_ft)

#################################
# Data augmentation and normalization for training
# Just normalization for validation
#################################
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#print("Initializing Datasets and Dataloaders...")

#################################
# Create training and validation datasets
# Just normalization for validation
#################################
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}
#image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid', 'test']}
#dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid']}

#################################
# Send the model to GPU
#################################
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Send the model to GPU
model_ft = model_ft.to(device)

#################################
# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
#################################
params_to_update = model_ft.parameters()
#print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            #print("\t",name)
'''
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            #print("\t",name)
'''

#################################
# Observe that all parameters are being optimized
#################################

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

#################################
# Setup the loss fxn
#################################

criterion = nn.CrossEntropyLoss()

#################################
# Train and evaluate
#################################

model4, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))


#################################
# save the weight of this ViT model
#################################
PATH = '/content/drive/MyDrive/shared/save_models3/vit4.pth'
torch.save(model4.state_dict(), PATH)
model4_ = torch.load(PATH)
model4_

##################################################################
# training with samples_x384_all5: 
# 383x383 sized jpg including all samples but random selection among train, valid and test
##################################################################
data_dir = "/content/drive/MyDrive/shared/samples_x384_all_5"
model_name = "vit"
num_classes = 4
num_epochs = 20
feature_extract = False
batch_size = 8

#tensor([0.8478, 0.6723, 0.7548])
#tensor([0.1271, 0.2422, 0.1877])

#################################
# Initialize the model for this run
#################################
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
# Print the model we just instantiated
#print(model_ft)

#################################
# Data augmentation and normalization for training
# Just normalization for validation
#################################
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#print("Initializing Datasets and Dataloaders...")

#################################
# Create training and validation datasets
# Just normalization for validation
#################################
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}
#image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid', 'test']}
#dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid']}

#################################
# Send the model to GPU
#################################
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Send the model to GPU
model_ft = model_ft.to(device)

#################################
# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
#################################
params_to_update = model_ft.parameters()
#print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            #print("\t",name)
'''
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            #print("\t",name)
'''

#################################
# Observe that all parameters are being optimized
#################################

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


#################################
# Setup the loss fxn
#################################

criterion = nn.CrossEntropyLoss()

#################################
# Train and evaluate
#################################

model5, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

#################################
# save the weight of this ViT model
#################################
#PATH = '/content/drive/MyDrive/shared/save_models/vit5.pth'
PATH = '/content/drive/MyDrive/shared/save_models3/vit5.pth'
torch.save(model5.state_dict(), PATH)
model5_ = torch.load(PATH)
model5_


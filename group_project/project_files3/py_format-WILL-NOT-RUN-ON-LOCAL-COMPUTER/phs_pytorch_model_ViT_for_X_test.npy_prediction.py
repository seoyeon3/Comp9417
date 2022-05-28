#!/usr/bin/env python
# coding: utf-8
##################################################################
# This code gets the predictions on real test sample by one ViT model
# and save the result as numpy file.
##################################################################
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
import os
import copy
from pytorch_pretrained_vit import ViT
from PIL import Image
import glob
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

#################################
# sample paths
# Top level data directory. 
# Here we assume the format of the directory conforms to the ImageFolder structure
#################################

# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
# data_dir = "/content/drive/MyDrive/shared/samples"
# data_dir = "/content/drive/MyDrive/shared/samples_small"
# data_dir = "/content/drive/MyDrive/shared/samples_x299"  # for indeption
# data_dir = "/content/drive/MyDrive/shared/samples_x384"  # for vit
# data_dir = "/content/drive/MyDrive/shared/samples_x384_all"  # for vit
# data_dir = "/content/drive/MyDrive/shared/samples_x384_all_5"  # for vit
# data_dir = "/content/drive/MyDrive/shared/samples_x384_all_4"  # for vit
# data_dir = "/content/drive/MyDrive/shared/samples_x384_all_3"  # for vit
# data_dir = "/content/drive/MyDrive/shared/samples_x384_all_2"  # for vit
data_dir = "/content/drive/MyDrive/shared/samples_x384_all_1"  # for vit
# data_dir = "/content/drive/MyDrive/shared/samples_x384_small_5"  # for vit
# data_dir = "/content/drive/MyDrive/samples2"  # for vgg
# data_dir = "/content/drive/MyDrive/shared/samples_x384_small_4"  # for vit


#################################
# various model to test
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
#################################

# model_name = "resnet"
# model_name = "alexnet"
# model_name = "vgg"
# model_name = "squeezenet"
# model_name = "densenet"
# model_name = "inception"
model_name = "vit"

#################################
# Number of classes in the dataset
#################################

num_classes = 4


#################################
# Batch size for training (change depending on how much memory you have)
#################################
# batch_size = 43
batch_size = 5  # for ViT


#################################
# Number of epochs to train for
#################################
num_epochs = 20


#################################
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
#################################

# feature_extract = True
feature_extract = False


#################################
# function for model training
#################################
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_acc_test = 0.0

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
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    time_result = 'Time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
    acc_result = 'Acc: {:4f}'.format(best_acc)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, time_result, acc_result


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
# Initialize the model for this run
#################################
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)


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

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}
# Create training and validation dataloaders
# dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid', 'test']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#################################
# Send the model to GPU
#################################

model_ft = model_ft.to(device)


#################################
# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
#################################

params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

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

# model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
model_ft, hist, time_result, acc_result = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))


#################################
# function to test model on test sample
#################################

def test_model(model, dataloaders):
  # Iterate over data.
  phase = 'test'
  running_corrects = 0
  for inputs, labels in dataloaders[phase]:
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      running_corrects += torch.sum(preds == labels.data)

  test_acc = running_corrects.double() / len(dataloaders[phase].dataset)

  test_acc_result = 'Test: {:4f}'.format(test_acc)

  return test_acc_result, preds

test_acc_result, preds = test_model(model_ft, dataloaders_dict)
print(test_acc_result)
print("preds: ", preds )


#################################
# function to test model on real test sample ( X_test.npy)
#################################
def real_test_model(model, real_test_samples_folder, images_folder, input_size, save_filename):
    
#     with open('imagenet_classes.txt') as f:
#       labels = [line.strip() for line in f.readlines()]
    labels = [0, 1, 2, 3]

    model.eval()

    data_transforms = {
        'real_test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    } 

    test_img_list = []
    predictions = []
    for filename in glob.glob(f'{real_test_samples_folder}/{images_folder}/*.jpg'): # assuming jpg
        img=Image.open(filename)
        img_t = data_transforms['real_test'](img)
        batch_t = torch.unsqueeze(img_t, 0)
        batch_t = batch_t.to(device)
        out = model(batch_t)
        _, index = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        print(labels[index[0]], percentage[index[0]].item())
        test_img_list.append(filename.split('/')[-1])
        predictions.append(labels[index[0]])

    print('image list:', test_img_list)
    print('predictions:', predictions)
    np.save(f'{real_test_samples_folder}/{save_filename}', np.array(predictions))

real_test_samples_folder = '/content/drive/MyDrive/shared/samples_x384_all_real_test'
images_folder = 'real_test'

# real_test_samples_folder = '/content/drive/MyDrive/shared/samples_x384_all/valid'
# images_folder = '3'

# input_size = 224 # for other models
input_size = 384 # for ViT
save_filename = 'X_test_preds1.npy'

real_test_model(model_ft, real_test_samples_folder, images_folder, input_size, save_filename)
read_again = np.load(f'{real_test_samples_folder}/{save_filename}')
print('read_again', read_again)


#################################
# For comparison purpose between non-trained version vs pretrained version.
# Initialize the non-pretrained version of the model used for this run
#################################

scratch_model,_ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
scratch_model = scratch_model.to(device)
scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
scratch_criterion = nn.CrossEntropyLoss()
# _,scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))
_,scratch_hist, _, _ = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))

# Plot the training curves of validation accuracy vs. number 
#  of training epochs for the transfer learning method and
#  the model trained from scratch
ohist = []
shist = []

ohist = [h.cpu().numpy() for h in hist]
shist = [h.cpu().numpy() for h in scratch_hist]

# plt.title(f"Validation Accuracy vs. Number of Training Epochs: {model_name}")
plt.title(f"Model: {model_name}, {time_result}, {acc_result}, {test_acc_result}")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
plt.plot(range(1,num_epochs+1),shist,label="Scratch")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()




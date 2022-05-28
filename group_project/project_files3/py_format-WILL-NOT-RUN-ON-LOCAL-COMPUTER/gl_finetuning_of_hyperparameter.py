#!/usr/bin/env python
# coding: utf-8
##################################################################
# This code tests various hyperparameters: learning rate and batch size
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
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

#additional modules
from datetime import datetime #for saving files
import pandas as pd
import torch
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


#################################
# function for model training
#################################
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False , saved_model=None):
    since = time.time()
    val_acc_history = []
    val_loss_history = []
    #best_model_wts = weights
    #if (saved_model == None):
      #best_model_wts = copy.deepcopy(model.state_dict())
    #else: 
      #best_model_wts = copy.deepcopy(saved_model)
    best_acc_valid = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

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
            if phase == 'valid' and epoch_acc > best_acc_valid:
                best_acc_valid = epoch_acc
                best_loss_valid = epoch_loss

                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
        print()

    time_elapsed = time.time() - since
    time_result = 'Time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc_valid))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, val_loss_history, time_result, best_acc_valid, best_loss_valid


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
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "vit":
        """ 
        vit
        """
        model_ft = ViT('B_16_imagenet1k', pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 384
    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size#, copy.deepcopy(model_ft.state_dict()) #added deep model weights 
        #best_model_wts = copy.deepcopy(model.state_dict())


#################################
# INITIALISE GLOBAL VARIABLES
#################################

# Data
# data_dir = "/content/drive/MyDrive/shared/samples_x384"  # for vit
data_dir = "/content/drive/MyDrive/shared/samples_x384_all"  # for vit
# data_dir = "/content/drive/MyDrive/shared/samples_x384_small_5"  # for vit
# data_dir = "/content/drive/MyDrive/shared/samples" # for vgg
# data_dir = "/content/drive/MyDrive/shared/samples_x384_small_4"  # for vit

# Variables
num_classes = 4
batch_size = 6 # vit <- this is Gordon's sys limit from hyperparameter tuning
num_epochs = 10
feature_extract = False # if true, then we only update the reshaped layer param of the model

# Model
model_name = "vit"
# Initialize the model for this run
#model_ft, input_size, init_weights = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)


#################################
# Data normalisation
# Gordon Add Seoyeon's part
#################################

def get_mean_std(loader,phase):
  channels_sum, channels_squared_sum, num_batches = 0, 0, 0
  
  for data,_ in loader[phase]:
    #print(data[0])
    channels_sum += torch.mean(data, dim=[0,2,3])
    channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
    num_batches += 1
  mean = channels_sum/num_batches
  std = (channels_squared_sum/num_batches - mean**2)**0.5

  return mean, std


#################################
#Get training data into tensorflow
#################################

phase = 'train'
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
}
train_datasets = {phase: datasets.ImageFolder(os.path.join(data_dir, phase), data_transforms[phase])}
train_loader = {phase: torch.utils.data.DataLoader(train_datasets[phase], batch_size=batch_size, shuffle=True, num_workers=4)}


mean,std = get_mean_std(train_loader,'train')


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
        #transforms.Normalize(mean, std)
    ]),
    'valid': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize(mean, std)


    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize(mean, std)
    ])
}

print("Initializing Datasets and Dataloaders...")
# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'valid', 'test']}


#################################
# Visualise sample dataset
#################################

#image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}
train_c = len(image_datasets['train']) #680 #144 vit
valid_c = len(image_datasets['valid']) #89 #22
test_c = len(image_datasets['test']) #89 #21

data_c = {i:len(image_datasets[i]) for i in ['train','valid','test']}

def data_bar_plt(data_c):
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  ax.title.set_text("Sample data split")
  x_axis = list(data_c.keys())
  y_axis = list(data_c.values())
  ax.bar(x_axis,y_axis,color = '#1f77b4')
  for x,y in zip(x_axis,y_axis):
    label = f"{y}"
    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,3), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
  plt.show()

data_bar_plt(data_c)


#################################
# **Single Run**
# Execute this if you only want to do a single run of the data, else, skip to the hyperparameter testing section
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

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()


#################################
# Using a saved model to save the training time
# Send the model to GPU
#################################
PATH = '/content/drive/MyDrive/shared/save_models/vit5.pth'
#torch.save(model_ft.state_dict(), PATH)
model = torch.load(PATH)

# Send the model to GPU - model is still fresh at this point
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

#Load the model
model_ft.load_state_dict(model)
#model_ft.eval() if not training


#################################
# Train one model and evaluate
#################################

model_ft, val_hist_acc, val_hist_loss, time_result, best_acc_valid, best_loss_valid = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,  num_epochs=num_epochs, is_inception=(model_name=="inception"))


#Save model if required:
#torch.save(model_ft.state_dict(), PATH)


#################################
# function to test model on test sample
# This section runs the pre-trained model against a set of test images to predict the class.
# Accuracy of the procedure is then computed on the basis of number of total correct classifications 
# over the total number of images classified.
#################################

import sklearn.metrics as sk

# test
def test_model_eval(model, dataloaders, phase):
  # Iterate over data.
  running_corrects = 0
  len_pred = 0
  predicts = []
  true_vals = []
  date = datetime.today().strftime('%Y%m%d')
  time = datetime.today().strftime('%H%M%S')
  
  for inputs, labels in dataloaders[phase]:
      inputs = inputs.to(device)
      labels = labels.to(device) #truths
      model.eval()
      outputs = model(inputs)
      _, prediction = torch.max(outputs, 1) 
      running_corrects += torch.sum(prediction == labels.data)

      #np.save(prediction, f"/content/drive/MyDrive/shared/results/gl_X_test_pred_{time}.npy")
      
      #Store prediction and lables into cpu list
      for i in prediction.cpu().numpy():
        predicts.append(i)
      for i in labels.data.cpu().numpy():
        true_vals.append(i)
        len_pred += 1
      
  test_acc = running_corrects.double() / len(dataloaders[phase].dataset)
  
  predicts_transformed = predicts.copy()
  true_vals_transformed = true_vals.copy()
  #Process 1,2,3 classes to value of 1 to identify tumor
  for i in range(0,len_pred):
    if (predicts[i] in [1,2,3]):
      predicts_transformed[i] = 1
    if (true_vals[i] in[1,2,3]):
      true_vals_transformed[i] = 1

  #Gordon's recall:
  recall = sk.recall_score(true_vals_transformed, predicts_transformed)
  f1 = sk.f1_score(true_vals, predicts,average='weighted')
  f1_all = sk.f1_score(true_vals, predicts, average=None)
  
  #View sample of data (remove in final code)
  if len(predicts) < 100:
    print("Predictions:", predicts)
    print("True values:", true_vals) 
    print()

  return test_acc, prediction, recall, f1, f1_all#, true_recall
#Run test model


#Against test dataset from cross-validation 
test_acc, prediction, recall, f1, f1_all = test_model_eval(model_ft, dataloaders_dict , phase = 'test')
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"Weighted f1: {f1*100:.2f}%")
for i in range(0,len(f1_all)):
  print(f"f1 for class {i}: {f1_all[i]*100:.2f}%")


#################################
# Model evaluation - against ALL data (real_test)
#################################

# Data augmentation and normalization for training
data_dir2 = "/content/drive/MyDrive/shared/samples_x384_all_real_test" # for vit
#data_dir = "/content/drive/MyDrive/shared/samples_x384_all"  # for vit

#Get training data into tensorflow
phase = 'real_test'
data_transforms2 = {
    'real_test': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
}
test_datasets = {phase: datasets.ImageFolder(data_dir2, data_transforms2[phase])}
test_loader = {phase: torch.utils.data.DataLoader(train_datasets[phase], batch_size=batch_size, shuffle=True, num_workers=1)}

print("Initializing Datasets and Dataloaders...")
# Create training and validation datasets
#img2 = {x: datasets.ImageFolder(os.path.join(data_dir2, x), data_transforms2[x]) for x in ['real_test']}
# Create training and validation dataloaders
#dict2 = {x: torch.utils.data.DataLoader(img2[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['real_test']}


test_acc, prediction, recall, f1, f1_all = test_model_eval(model_ft, test_loader, phase = 'real_test')
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"Weighted f1: {f1*100:.2f}%")
for i in range(0,len(f1_all)):
  print(f"f1 for class {i}: {f1_all[i]*100:.2f}%")


#################################
# train multiple models for testing e.g. learning rate
#################################
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

#hyperparameters
learning_list = [0.01, 0.0015, 0.001, 0.0005] #define arbitrary learning elements
# momentum = [0.95, 0.9, 0.8]
# feature_extract
# use_pretrained

#################################
# Hyper
#################################
def hpt_learning(learning_list, model_ft):
  #initialise storage
  hist_loss_dict = {}
  hist_acc_dict = {}
  time_results = {}

  for i in learning_list.copy():
    #clear memory
    torch.cuda.empty_cache()

    #reinit model_ft:
    model_ft,_ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
    model_ft = model_ft.to(device)

    #init optimizer
    params_to_update = model_ft.parameters()
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    optimizer_ft = optim.SGD(params_to_update, lr = i, momentum = 0.90)
    
    #train
    since = time.time()
    model_ft, val_hist_acc, val_hist_loss, time_result, best_acc_valid, best_loss_valid =     train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,  num_epochs=num_epochs, is_inception=(model_name=="inception"))

    print(f"Training param {i} complete\n")
    hist_loss_dict[i] = val_hist_loss
    hist_acc_dict[i] = [h.cpu().numpy() for h in val_hist_acc]
    time_results[i] = time_result
  return hist_loss_dict, hist_acc_dict, time_results

hloss_d_b, hacc_d_b, time_result = hpt_learning(learning_list, model_ft)


#################################
# loss
#################################
def hp_plt_loss(loss_val, param_tuned, short_name):
  plt.title(f"Hyperparameter: {param_tuned}")
  plt.xlabel("Training Epochs")
  plt.ylabel("Loss")

  for i in hloss_d:
    plt.plot(range(1,len(loss_val[i])+1), loss_val[i], label = f"{short_name} = {i}")
    #plt.plot(range(1,len(hloss_d[i])+1), hloss_d[i], label = f"{param_tuned} = {i}")

  plt.ylim((0,1.))
  plt.xticks(np.arange(1, num_epochs+1, 1.0))
  plt.legend()
  plt.show()

hp_plt_loss(hloss_d_b,'Learning Rate','lr')

#################################
# accuracy
#################################
def hp_plt_acc(accuracy_val, param_tuned , short_name):
  plt.title(f"Hyperparameter: {param_tuned}")
  plt.xlabel("Training Epochs")
  plt.ylabel("Accuracy(%)")
  for i in hacc_d:
    plt.plot(range(1,len(hacc_d[i])+1), hacc_d[i], label = f"{short_name} = {i}")
  plt.ylim((0,1.))
  plt.xticks(np.arange(1, num_epochs+1, 1.0))
  plt.legend()
  plt.show()

hp_plt_acc(hacc_d_b, "Learning Rate","lr")


#################################
# Batch Size
#################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

#hyperparameters
batch_sizes = [4,5,7]
num_epochs = 5
#Hyper
def hpt_batch(batch_sizes, model_ft):
  #initialise storage
  hist_loss_dict = {}
  hist_acc_dict = {}
  time_results = {}

  for i in batch_sizes.copy():
    #clear memory
    torch.cuda.empty_cache()
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=i, shuffle=True, num_workers=4) for x in ['train', 'valid', 'test']}

    #reinit model_ft:
    model_ft,_ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
    model_ft = model_ft.to(device)

    #init optimizer
    params_to_update = model_ft.parameters()
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    optimizer_ft = optim.SGD(params_to_update, lr = 0.001, momentum = 0.90)
    
    #train
    since = time.time()
    model_ft, val_hist_acc, val_hist_loss, time_result, best_acc_valid, best_loss_valid =     train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,  num_epochs=num_epochs, is_inception=(model_name=="inception"))

    print(f"Training param {i} complete\n")
    hist_loss_dict[i] = val_hist_loss
    hist_acc_dict[i] = [h.cpu().numpy() for h in val_hist_acc]
    time_results[i] = time_result
  return hist_loss_dict, hist_acc_dict, time_results

hloss_d, hacc_d, time_result = hpt_batch(batch_sizes, model_ft)


hp_plt_acc(hacc_d_b, "Batch size","b_size")


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
shist = [h.cpu().numpy() for h in scratch_hist]

# plt.title(f"Validation Accuracy vs. Number of Training Epochs: {model_name}")
plt.title(f"Model: {model_name}, {time_result}")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs+1),hloss_d,label="Pretrained")
plt.plot(range(1,num_epochs+1),shist,label="Scratch")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()

#################################
# **Model Performance Diagnostics**
# This section visualises the performance of code onto matplots charts to be used in the reports.
#
# Part 1: Visualise performance of model when running against data for Vgg, densenet and Vit
#
# Part 2: Visualise the chart into plots of 3 distinct model performance 
#################################


def plot_performance(model_name, time_result, best_acc_valid, test_acc, num_epochs, val_hist_loss, val_hist_acc):
    fig,ax = plt.subplots()

    ax_color = '#ff7f0e'
    label_colour = 'black'
    ax.title.set_text(f"Model: {model_name}, {time_result}, Acc: {best_acc_valid*100.0:2.2f}%, Test: {test_acc*100.0:2.2f}%")
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Loss", color = label_colour)
    l1 = ax.plot(np.arange(1, num_epochs+1, 1.0), val_hist_loss, color = ax_color, label = "Loss")
    
    ax2_color = '#1f77b4'
    ax2 = ax.twinx()
    ax2.set_ylabel("Accuracy(%)", color = label_colour)
    val_hist_acc_np = [h.cpu().numpy() for h in val_hist_acc]
    l2 = ax2.plot(np.arange(1, num_epochs+1, 1.0), val_hist_acc_np, color = ax2_color, label = "Accuracy")

    date = datetime.today().strftime('%Y%m%d')
    time = datetime.today().strftime('%H%M%S')
    plt.ylim((0,1.))
    ax.set_xticks(np.arange(1, num_epochs+1, 1))
    
    fig.legend([l1,l2],labels=["Loss","Accuracy"], bbox_to_anchor = [0.885,0.895])
    fig.savefig(f"/content/drive/MyDrive/shared/results/{model_name}_{date}_{time}.png")
    fig.show()


##Part 1:
plot_performance(model_name, time_result,best_acc_valid, test_acc, num_epochs, val_hist_loss, val_hist_acc)


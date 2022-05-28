#!/usr/bin/env python
# coding: utf-8
##################################################################
# This code creates various sample datasets as images 
# needed for pretrained pytorch models
##################################################################
#################################
# access to google drive
#################################
from google.colab import drive

drive.mount('/content/drive')


#################################
# import modules
#################################
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


#################################
# file path
#################################

base_dir = "/content/drive/MyDrive/shared"
# base_dir = "."
# np.random.seed(1)
# sample_dir = 'samples'

# np.random.seed(1)
# sample_dir = 'samples_small'

# np.random.seed(2)
# sample_dir = 'samples_small2'

# np.random.seed(3)
# sample_dir = 'samples_small3'

# np.random.seed(4)
# sample_dir = 'samples_small4'

# np.random.seed(5)
# sample_dir = 'samples_small5'
# # img_size = 224

# np.random.seed(1)
# sample_dir = 'samples_x384_all'
# img_size = 384 # for vit

# np.random.seed(1)
# sample_dir = 'samples_x384_all_1'
# img_size = 384 # for vit

# np.random.seed(2)
# sample_dir = 'samples_x384_all_2'
# img_size = 384 # for vit

# np.random.seed(3)
# sample_dir = 'samples_x384_all_3'
# img_size = 384 # for vit

# np.random.seed(4)
# sample_dir = 'samples_x384_all_4'
# img_size = 384 # for vit

# np.random.seed(5)
# sample_dir = 'samples_x384_all_5'
# img_size = 384 # for vit

# np.random.seed(1)
# sample_dir = 'samples_x384_small_1'
# img_size = 384 # for vit

# np.random.seed(2)
# sample_dir = 'samples_x384_small_2'
# img_size = 384 # for vit

# np.random.seed(3)
# sample_dir = 'samples_x384_small_3'
# img_size = 384 # for vit

# np.random.seed(4)
# sample_dir = 'samples_x384_small_4'
# img_size = 384 # for vit

np.random.seed(5)
sample_dir = 'samples_x384_small_5'
img_size = 384 # for vit

# np.random.seed(1)
# sample_dir = 'samples_x384_all_test_1'
# img_size = 384 # for vit

# np.random.seed(1)
# sample_dir = 'samples_x299'
# img_size = 299 # for inception


#################################
# load train samples from npy file
#################################

X_train=np.load(f"{base_dir}/X_train.npy", mmap_mode='r')


y_train = np.load(f"{base_dir}/y_train.npy")


img_indexes = list(range(len(y_train)))


total_zips =list(zip(X_train, y_train, img_indexes))


#################################
# aggregate the samples into 2 groups:
# class0 contains class0, class1, class2
# class3 contains only class3
#################################


class0 = 0
class1 = 0
class2 = 0
class3 = 0
class0_zips = []
class3_zips = []

for i in range(len(total_zips)):
    class_val = total_zips[i][1]
    if class_val == 0:
        class0 += 1
        class0_zips.append(total_zips[i])
    elif class_val == 1:
        class1 += 1
        class0_zips.append(total_zips[i])
    elif class_val == 2:
        class2 += 1
        class0_zips.append(total_zips[i])
    elif class_val == 3:
        class3 += 1
        class3_zips.append(total_zips[i])
        
print(class0, class1, class2, class3, len(class3_zips)) # 407 225 189 37 37


#################################
# train , valid and test sample ratio
# class0_sample_size = len(class0_zips) # to make a whole X_train.npy dataset
#################################

class0_sample_size = 150 # to make smaller dataset
class3_sample_size = len(class3_zips) # lets use whole class3 data because they are rare.

class0_ratio = {"train": 0.8, "valid": 0.1, "test": 0.1}
class3_ratio = {"train": 0.65, "valid": 0.2, "test": 0.15}


class0_train_size = int(class0_sample_size * class0_ratio["train"])
class0_valid_size = int(class0_sample_size * class0_ratio["valid"])
class0_test_size = class0_sample_size - class0_train_size - class0_valid_size
# class0_train_size


class3_train_size = int(class3_sample_size * class3_ratio["train"])
class3_valid_size = int(class3_sample_size * class3_ratio["valid"])
class3_test_size = class3_sample_size - class3_train_size - class3_valid_size
# class3_sample_size # 37
# class3_train_size # 24
# class3_valid_size # 7
# class3_test_size # 6


#################################
# random sampling to make a train , valid and test sample
#################################
class0_indexes = np.arange(len(class0_zips))
class0_train_indexes = np.random.choice(class0_indexes, class0_train_size, replace=False)
class0_valid_indexes = np.random.choice(class0_indexes, class0_valid_size, replace=False)
class0_test_indexes = np.random.choice(class0_indexes, class0_test_size, replace=False)

class3_indexes = np.arange(len(class3_zips))
class3_train_indexes = np.random.choice(class3_indexes, class3_train_size, replace=False)
class3_valid_indexes = np.random.choice(class3_indexes, class3_valid_size, replace=False)
class3_test_indexes = np.random.choice(class3_indexes, class3_test_size, replace=False)

#################################
# clean up the existing folder before saving samples 
#################################
def remove_samples_folder(sample_dir):
    base_path = base_dir + "/" + sample_dir
    for root, dirs, files in os.walk(base_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


remove_samples_folder(sample_dir)


#################################
# create sample folders.
# a train , valid and test folder should be created.
# Under each folder, each class folder should be created.
#################################
def create_samples_folder(sample_dir):
    base_path = base_dir + "/" + sample_dir
    
    for target_dir in ["train", "valid", "test"]:
        os.mkdir(base_path + "/" + target_dir)
        os.mkdir(base_path + "/" + target_dir + "/0")
        os.mkdir(base_path + "/" + target_dir + "/1")
        os.mkdir(base_path + "/" + target_dir + "/2")
        os.mkdir(base_path + "/" + target_dir + "/3")


create_samples_folder(sample_dir)


#################################
# we will create image based on these 3 zips.
# Created sample images are distributed into each class folder.
#################################

def create_samples(sample_type, class0_sample_indexes, class3_sample_indexes, img_size):
    # [train, valid, test]
    base_path = base_dir + "/" + sample_dir + "/" + sample_type


    # common settings
    class0_dir = "0"
    class1_dir = "1"
    class2_dir = "2"
    class3_dir = "3"

    img_path = ""
    # img_size = 224
    img_size = img_size
    img_type = '.jpg'

    for i in class0_sample_indexes:
        img_npy = class0_zips[i][0]
        img_class = class0_zips[i][1]
        img_index = class0_zips[i][2]

        if img_class == 0:
            img_path = class0_dir 
        elif img_class == 1:
            img_path = class1_dir 
        elif img_class == 2:
            img_path = class2_dir 
        elif img_class == 3:
            img_path = class3_dir  

        img_resized = cv2.resize(img_npy, (img_size, img_size), interpolation=cv2.INTER_AREA)
        plt.imsave(f"{base_path}/{img_path}/{img_index}{img_type}", img_resized)
    
    for i in class3_sample_indexes:
        img_npy = class3_zips[i][0]
        img_class = class3_zips[i][1]
        img_index = class3_zips[i][2]

        if img_class == 0:
            img_path = class0_dir 
        elif img_class == 1:
            img_path = class1_dir 
        elif img_class == 2:
            img_path = class2_dir 
        elif img_class == 3:
            img_path = class3_dir  

        img_resized = cv2.resize(img_npy, (img_size, img_size), interpolation=cv2.INTER_AREA)
        plt.imsave(f"{base_path}/{img_path}/{img_index}{img_type}", img_resized)


#################################
# Create sample images for training
#################################
create_samples("train", class0_train_indexes, class3_train_indexes, img_size)


#################################
# Create sample images for validation
#################################
create_samples("valid", class0_valid_indexes, class3_valid_indexes, img_size)


#################################
# Create sample images for testing
#################################
create_samples("test", class0_test_indexes, class3_test_indexes, img_size)






### 0. File desciption

##### About our project codes
   Our project code is based on the finetuning torchvision modles
   in [Pytorch documents](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)   
   We modified it to suit our project purpose.

   Our code is intended to work on Google Colab and need to have access to the sample dataset on The Google drive also.
   Thus .py files will not work on a standalone local computer.

   Below is how to run our project code on Google Colab.
   And Google Colab link is shown at each file.

##### You can access to our sample datasets on Google drive with below link: 
   The base path to this shared folder is "/content/drive/MyDrive/shared"

   https://drive.google.com/drive/folders/1-7Ja45Xld5gbabnvNnDiQiaZFqlqFJwi?usp=sharing

##### phs_pytorch_dataset_preparation.ipynb
   This code creates various sample datasets as images needed for pretrained pytorch models.

   https://colab.research.google.com/drive/1CCBNQh59iwieCOD1qZ4v-SAoTVXYq_JC?usp=sharing

##### phs_pytorch_model_ViT_for_X_test.npy_prediction.ipynb
   This code gets the predictions on real test sample (X_test.npy) 
   by a ViT model and save the result as numpy file.

   https://colab.research.google.com/drive/1wLz6EaKIcmrjH5gZqhAB_iCM1KO6huxW?usp=sharing


##### phs_pytorch_model_ViT_for_X_test.npy_prediction_with_ensemble_method.ipynb
   This code gets the predictions on real test sample (X_test.npy)
   by voting ensemble by 5 ViT models trained on 5 different datasets.

   https://colab.research.google.com/drive/1_Iexvn_9gnkiZuLAAzi3zYRgAxvzi0Et?usp=sharing

##### syp_CNN_model_from_scratch.ipynb
   This code shows a scratch CNN model tested as initial trial.

   https://colab.research.google.com/drive/1JApUdk6LbMAQhIkgeLidGC6KylGI5pj3?usp=sharing

##### syp_combining_VGG_DenseNet_ViT_models_in_one_code.ipynb
   This code compares the accuracies of various pretrained models
   such as VGG, DenseNet and ViT

   https://colab.research.google.com/drive/1GC5Z3v3Ar1XrBi2I_RBSR3wi-0fNA9iv?usp=sharing

##### syp_save_the_trained_model_weight.ipynb
   This code saves the trained model's weight after training 
   so as to reuse later without re-training.

   https://colab.research.google.com/drive/10c8GtriHZ4iCs0Rrb35H-ezWpnb6R6Dv?usp=sharing

##### gl_finetuning_of_hyperparameter.ipynb
   This code analyze and tests various hyperparameters such as learning rate and batch size

   https://colab.research.google.com/drive/1EFRhAG9HmE_-_kQ3DdQoeEP8ue4YifIm?usp=sharing

### 1. Running environment 

We used [Google Colab](https://colab.research.google.com) for our group project.
Thus all the codes are in .ipynb format that are intended to run on Colab in GPU mode.

And our sample dataset was prepared on [Google Drive](https://drive.google.com) and 
shared among the members.

The attached .py format files are just files extracted from .ipynb format
and not intended to run on standalone local computer.

### 2. Sample preparation
We used an JPG images as the sample dataset and these sample images should be placed
in a folder according to their classes, which the pytorch pretrained models require.

Thus sample dataset should be prepared before running our project code.
One example of sample dataset was attached in sample dataset folder.

With our "phs_pytorch_dataset_preparation.ipynb", 
the sample dataset is automatically created from X_train.npy or X_test.npy. 

### 3. Quick Running on Google Colab

a. Upload the sample dataset on the Google Drive.
b. Upload the .ipynb file on Google Colab.
c. Make sure that Google Colab can access the Google Drive files.
d. Run the Google Colab in GPU mode.
e. If the case is to save a result as a file, 
   make sure to create a folder that saves a file before running.

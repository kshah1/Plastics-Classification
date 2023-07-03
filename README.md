# Plastics Classifcation

The purpose of this project is to be able to use transfer learning of some state of the art models (Residual Network's and Dense Net) on classifying plastics.

## Author: Karan Shah

If any issues, contact me at shah.karan3@northeastern.edu 

## Environment  
OS: MacOS M1 Pro  
IDE: Visual Studio Code  
Python version: 3.9.12  

# How to run the code

## To wrangle the data into proper train, validation, and test sets. The first argument refers to where the Plastics dataset belongs which is organized into heavy_plastic, no_image, no_plastic, and some_plastic. The second argument refers to the location where the trainValTest folder will be created which modifies the data structure from the first argument  
python createTrainValTest.py /Users/karanshah/Spring2023/CV/Karan/FinalProject/Plastics-Classification/Plastics\ Classification/ /Users/karanshah/Spring2023/CV/Karan/FinalProject/trainValTest

## To train and save the models, change the dataroot to where the trainValTest folder is located from the previous data wrangle step.   
python transferLearning.py (dataroot in transferLearning.py is /Users/karanshah/Spring2023/CV/Karan/FinalProject/Plastics-Classification/trainValTest)

## To plot the precision recall curves and evaluate the mdoels
python Evaluation.py (dataroot in Evaluation.py is /Users/karanshah/Spring2023/CV/Karan/FinalProject/Plastics-Classification/Evaluation.py

# Fracture Detection using Ghost Imaging and Neural Networks
Welcome to our project focusing on Fracture Detection through the innovative combination of Ghost Imaging (GI) techniques and advanced machine learning.
In this project, we explore the potential of using machine learning algorithms to identify bone fractures from X-rays captured using the Ghost imaging method.
Our goal is to investigate whether this approach can contribute to accurate medical diagnoses while minimizing radiation exposure, offering a safer alternative
for patients undergoing X-ray evaluations.

## Installation
### Dataset
We utilized a publicly available dataset of wrist X-ray images which we later preprocessed to generate GI measurements for training and evaluating our neural network model. You can download the dataset from [here](https://www.nature.com/articles/s41597-022-01328-z#Sec9).  
Please refer to the dataset's documentation for usage terms, licensing, and any specific instructions provided by the dataset creators.  
(You can instead use the measurements found in the "Processed Data" folder as explained below)
### Packages required
Install the required packages using the command:  
`pip install torch torchvision pandas numpy tqdm Pillow wandb`
### Code Parts
Our project consists of three distinct code parts, each contained in its respective folder:  
1. **GI_MINST**: This folder contains the code for reproducing the results of [this article](https://pubmed.ncbi.nlm.nih.gov/34624000/).
2. **Wrist_original_architecture**: In this folder, you'll find the code where we adapted the architecture from the article to work with wrist fracture images.
3. **Wrist_transfer_learning**: This folder holds the code for the third part, where we applied transfer learning to the wrist fracture dataset.
please download/clone the folder you are interested in. In addition, download the file train_and_test.py and for the second and third folders also the file preprocessing_wrist.py.

## How to use
### MNIST
### article
### transfer learning

## examples


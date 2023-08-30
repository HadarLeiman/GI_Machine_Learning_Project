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
   
Please download/clone the folder you are interested in. In addition, download/clone the file **train_and_test.py** and for the second and third folders also the file **preprocessing_wrist.py**.

## How to use
### First Part - GI MNIST
Run the main.py file.
### Second and third Part - Wrist original architecture & transfer learning
##### preprocess
Use the preprocessing_wrist.py file to create processed data in the form of GI measurements from the X-ray image dataset to use by the model.
Pass:
   - Number of Ghost Imaging measurements
   - Resize size applied to the data during preprocessing
   - Path to the original dataset
##### Running with wandb
If you'd like to run the project using WandB for experiment tracking, execute the `sweep_conf.py` script in the Wrist_original_architecture folder or `main_sweep` in the Wrist_transfer_learning folder. This will initiate training runs with different configurations, and the results will be logged to your WandB account.
##### Running with Custom Configuration
To run the project with a specific configuration, follow these steps:
1. Open the `main_for_wrist.py` file in Wrist_original_architecture folder or `model_pipeline.py` in Wrist_transfer_learning folder.
2. Locate the `main` function in `main_for_wrist.py` or the `model_pipeline` function in `model_pipeline.py`.
3. Pass your desired configuration. The configuration should include:
   - Learning rate
   - Number of epochs
   - Resize size applied to the data during preprocessing
   - Number of Ghost Imaging measurements
   - Batch size (in `main`)
   - model name (in `model_pipeline`)
   - num_of_layers_unfreeze (in `model_pipeline`)

   For example: `main((0.001, 10, (64, 128), 10, 32))` or `model_pipeline((5, 2, ResNet, 0.0001, 512, 64_128))`
This approach allows you to fine-tune the project's parameters according to your preferences. Feel free to experiment with different configurations to observe their effects on the training process and results.

## examples


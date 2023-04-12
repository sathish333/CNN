This repository contains solutions for Assignment-2. 

# Part A:

Objective is to a build convolutional network using pytorch which can classify given image to one of the target classes.

Dataset: This dataset is subset of original *iNaturalist Dataset* which includes various living organisms images.
* Number of classes: 10
* Number of samples in train: 10K
* Number fo samples in test: 2K

20% of the data from train is kept aside as validation data.
## Folder structure

* *PartA_CNN.ipynb:*  Containes the code for Hyperparameters search using WandB.
* *CNN_Best_Model.ipynb:* Building the final model based on best configuration obtained.
* *train.py*: Builds and trains CNN based on arguments passed from the command line. Default values are set as per best configuration.



## How to use:
* In the first line, change the data path accordingly.<br><br>
Then run the command 
  ``` python train.py``` <br>  

One can overwrite the default values by passing command line aruguments.

**Arguments supported:**
<br>


| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | CS22M080 | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | CS22M080 | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-e`, `--epochs` | 10 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 16 | Batch size used to train neural network. |
| `-lr`, `--learning_rate` | 0.0001 | Learning rate used to optimize model parameters |
| `-fs`, `--num_layers` | 5 | Number of conv layers |
| `-fs`, `--filters_size` | 32 | Number of filters in first conv layer |
| `-fo`, `--filter_organization` | double | Specifies how the number of filters in each conv is going to be. |
| `-ks`, `--kernel_size` | 3 | Each of conv kernel size |
| `-a`, `--activation` | Mish | choices:  ["RelU", "GELI", "SiLU", "Mish"] |
| `-do`, `--data_augumentation` | Yes | To enable data augumentation or not. |
| `-bn`, `--batch_normalisation` | Yes | To include batch norm or not. |
| `-ps`, `--pool_size` | 2 | each of the max pool window |
| `-do`, `--dropout` | 0 | Dropout percentage at FC layer|

<br>


# Part B:

In this part, goal is to make use of pretained models which were already been trained large datasets. Since these pre-trained models have learnt the important features to idenify, we can leverage it by fixing the initial layers weights and training only last few layers. 

File:  ```PartB_Resnet.ipynb```

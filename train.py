data_prefix='../../Downloads/inaturalist_12K/'


import wandb
wandb.login()

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
import shutil
import os
import random
import pytorch_lightning as pl
from types import SimpleNamespace
import random
import argparse



# tranform for test data( resize and normalize)

test_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
                        ])

# loading data

def get_data_loader(path,batch_size,transform,shuffle=False):#based on path returns dataset as well as data loader
    dataset=torchvision.datasets.ImageFolder(root=data_prefix+'train', transform=transform)
    return dataset,DataLoader(dataset, batch_size=16, shuffle=shuffle)
    

def getActivation(function): # activation functions
    if function=='ReLU':
        return nn.ReLU()
    if function=='GELU':
        return nn.GELU()
    if function=='SiLU':
        return nn.SELU()
    if function=='Mish':
        return nn.Mish()
    return nn.ReLU() # if no match

def soft_max(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0) #softmax normalize the probabilites
    
    

class Model(pl.LightningModule): # Building the model by extending lightning module
    def __init__(self,config):
        
        super().__init__()
        self.learning_rate=config.learning_rate
        layers=[]
        input_channels=3
        num_layers=5
        kernel_size=config.kernel_size # reading values from config
        kernel_stride=1
        max_pool_size=config.pool_size
        max_pool_stride=max_pool_size
        filters=[]
         # based on filter org,determining number fo filters at each layer
        if(config.filter_organization=='same'):
            filters=[config.filters_size]*num_layers
        elif(config.filter_organization=='double'):
            filters.append(config.filters_size)
            for i in range(4):
                filters.append(filters[-1]*2)
        elif(config.filter_organization=='halve'):
            filters.append(config.filters_size)
            for i in range(4):
                filters.append(filters[-1]//2)

        filters.insert(0,input_channels)
        out_height=224
        for i in range(num_layers):
            layers.append(nn.Conv2d(filters[i],filters[i+1],kernel_size = kernel_size))  # conv layer
            out_height=(out_height-kernel_size)//kernel_stride+1  # output shape of conv layer
            layers.append(nn.MaxPool2d(kernel_size = max_pool_size,stride = max_pool_stride))   # max poolinglayer 
            out_height=out_height//max_pool_stride # output size of maxpool layer
            layers.append(getActivation(config.activation)) # activation 
            if(config.batch_normalisation=='Yes'): # if batch norm is enabled then add accordingly.
                layers.append(nn.BatchNorm2d(filters[i+1]))
        layers.append(nn.Flatten())
        layers.append(nn.Dropout(config.dropout)) # dropout at FC layer
        layers.append(nn.Linear(out_height*out_height*filters[-1],config.dense_layer_size))
        layers.append(nn.Linear(config.dense_layer_size,10)) # out put layer
        self.net = nn.Sequential(*layers)
        self.loss = nn.CrossEntropyLoss()
        self.train_loss=[]
        self.train_acc=[]
        
        
    def forward(self,x):
        return self.net(x) # passes input x to sequentially through all the layers and output is obtained from last layer
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr= self.learning_rate) # setting up adam optimizer

    def training_step(self,batch,batch_idx): # After every train batch, computes it's loss/acc and store it.
        X,Y = batch
        output = self(X)
        loss = self.loss(output,Y)
        acc = (output.argmax(dim = 1) == Y).float().mean()
        self.train_loss.append(loss)
        self.train_acc.append(acc)
        return loss
    def predict_step(self, batch, batch_idx):
        X, Y = batch
        preds = self.net(X)
        return preds
    
   

    def on_train_epoch_end(self): #once an epoch is completed, print and log the metrics to WandB
        train_loss=sum(self.train_loss)/len(self.train_loss)
        train_acc=sum(self.train_acc)/len(self.train_acc)
        self.train_acc=[]
        self.train_loss=[]
        print(f"Epoch: {self.current_epoch} train accuracy :{train_acc:.2f} train_loss :{train_loss:.2f}")
        wandb.log({'train_acc':train_acc,'train_loss':train_loss})
        
def calc_acc(data_loader,targets,trainer,model):# computes the accuracy of data loader and returns it
  preds = trainer.predict(model, data_loader)
  preds = torch.concat(preds)
  preds = preds.argmax(axis=1)
  preds=preds.numpy()
  targets=np.array(targets)
  return np.sum(preds==targets)/len(targets)

def model_fit(params):
    print("Building the model with provided hyper parameters...",params)
    
    if(params.data_augumentation=='Yes'): # loading only train data based on data augumentation is enabled or not.
        train_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                   ])
    else:
        train_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
                    ])
        
    
    _,train_loader=get_data_loader(data_prefix+'train',params.batch_size,train_transform,True)
    model = Model(params) 
    trainer = pl.Trainer(max_epochs=params.epochs,accelerator="auto",devices='auto') 
    trainer.fit(model,train_loader) # fitting the model
    return model,trainer




if __name__=="__main__": 
    parser = argparse.ArgumentParser(description = 'Set Hyper Parameters')
    parser.add_argument('-wp'   , '--wandb_project'  , type = str  , default='CS22M080',metavar = '', help = 'WandB Project Name (Non-Empty String)')
    parser.add_argument('-we'   , '--wandb_entity'   , type = str  , default='CS22M080',metavar = '', help = 'WandB Entity Name (Non-Empty String)')
    parser.add_argument('-e'    , '--epochs'         , type = int  , default=10,metavar = '', help = 'Number of Epochs (Positive Integer)')
    parser.add_argument('-b'    , '--batch_size'     , type = int  , default=16,metavar = '', help = 'Batch Size (Positive Integer)')
    parser.add_argument('-lr'   , '--learning_rate'  , type = float, default=0.0001,metavar = '', help = 'Learning Rate (Positive Float)')
    parser.add_argument('-nl'  , '--num_layers'     , type = int  , default=5,metavar = '', help = '')
    parser.add_argument('-a'    , '--activation'     , type = str  , default='Mish',metavar = '', help = '',choices=["ReLU", "GELU",'SiLU','Mish'] )
    parser.add_argument('-fs'   , '--filters_size'  , type = int  , default=32,metavar = '', help = 'number of filters')
    parser.add_argument('-bn'   , '--batch_normalisation'  , type = str  , default='Yes',metavar = '', help = '')
    parser.add_argument('-da'   , '--data_augumentation'  , type = str  , default='Yes',metavar = '', help = '')
    parser.add_argument('-fo'   , '--filter_organization'  , type = str  , default='same',metavar = '', help = '',choices=['same','double','half'])
    parser.add_argument('-dl'   , '--dense_layer_size'  , type = int  , default=256,metavar = '', help = '')
    parser.add_argument('-do'   , '--dropout'  , type = float  , default=0.0,metavar = '', help = '')
    parser.add_argument('-ks'   , '--kernel_size'  , type = int  , default=3,metavar = '', help = 'Each of the filter size')
    parser.add_argument('-ps'   , '--pool_size'  , type = int  , default=2,metavar = '', help = 'each of max pool size')
    
    
  
    
    # Parse the Input Args
    params = vars(parser.parse_args())
    wandb.init(project=params['wandb_project'],config=params)
    params=SimpleNamespace(**params)
    test_dataset,test_loader=get_data_loader(data_prefix+'train',params.batch_size,test_transform)
    run_name=f'FZ-{params.filters_size} AF - {params.activation} filter_org- {params.filter_organization} batch_norm -{params.batch_normalisation} data_aug -{params.data_augumentation} dropout- {params.dropout}'
    wandb.run.name=run_name
    model,trainer=model_fit(params)
    print("training completed")
    test_accuracy=calc_acc(test_loader,test_dataset.targets,trainer,model)
    print("*"*50)
    print(f"Final Test accuracy: {test_accuracy:.2f}")
    print("*"*50)
    wandb.log({'test accuracy':test_accuracy})
    wandb.finish()

 
#import datetime
import numpy as np
import os
from swarm import SwarmCallback
#import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules.data import SessionDataset
from modules.model import GRU4REC

default_max_epochs = 5
default_min_peers = 2
# maxEpochs = 2
# tell swarm after how many batches
# should it Sync. We are not doing 
# adaptiveRV here, its a simple and quick demo run
swSyncInterval = 1
        
def loadData(dataDir, trainName='train.csv', testName='test.csv'):
    # load data from npz format to numpy 
    train_dataset = SessionDataset(path=os.path.join(dataDir,trainName))
    test_dataset = SessionDataset(path=os.path.join(dataDir,testName), itemmap=train_dataset.itemmap)
    
    return train_dataset, test_dataset   

def main():
    dataDir = os.getenv('DATA_DIR', './data')
    modelDir = os.getenv('MODEL_DIR', './model')
    max_epochs = int(os.getenv('MAX_EPOCHS', str(default_max_epochs)))
    min_peers = int(os.getenv('MIN_PEERS', str(default_min_peers)))
    train_dataset, test_dataset = loadData(dataDir, trainName='small.csv')

    #Init some parameters
    input_size = len(train_dataset.items)
    if_embedding = False
    embedding_size = 16384
    hidden_size = 100
    num_layers = 1
    output_size = input_size
    batch_size = 50
    optimizer_type = 'Adagrad'
    lr = .01
    weight_decay = 0
    momentum = 0
    eps = 1e-6
    loss_types = ['CrossEntropy','BPR','TOP1']
    n_epochs = int(max_epochs)
    use_cuda = torch.cuda.is_available()
    cuda_id = 0
    loss_type = 'CrossEntropy'
    useCuda = torch.cuda.is_available()
    device = torch.device("cuda" if useCuda else "cpu")  
    model = GRU4REC(input_size, if_embedding, embedding_size, hidden_size, output_size,
                num_layers=num_layers,
                batch_size=batch_size,
                optimizer_type=optimizer_type,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                eps=eps,
                loss_type=loss_type,
                use_cuda=use_cuda,
                cuda_id=cuda_id)

    model_name = 'GRU4REC'

    print('#'*10 , 'A'*5, '#'*10)

    # Create Swarm callback
    swarmCallback = SwarmCallback(sync_interval=swSyncInterval,
                                  min_peers=min_peers,
                                  val_data=test_dataset,
                                  val_batch_size=batch_size,
                                  model_name=model_name,use_adaptive_sync=False,
                                  model=model)
    # initalize swarmCallback and do first sync 
    print('#'*10 , 'B'*5, '#'*10)
    swarmCallback.on_train_begin()
    print('#'*10 , 'C'*5, '#'*10)

    model.train(train_dataset, n_epochs=n_epochs, model_name=model_name, save=False, swarmCallback=swarmCallback)
    print('#'*10 , 'D'*5, '#'*10)
    
    # handles what to do when training ends        
    swarmCallback.on_train_end()
    print('#'*10 , 'E'*5, '#'*10)
    
    # Save model and weights
    model_path = os.path.join(modelDir, model_name, 'saved_model.pt')
    # Pytorch model save function expects the directory to be created before hand.
    os.makedirs(os.path.join(modelDir, model_name), exist_ok=True)
    torch.save(model, model_path)
    print('Saved the trained model!')
  
if __name__ == '__main__':
    main()

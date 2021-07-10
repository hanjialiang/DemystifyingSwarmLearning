import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import warnings
from densenet import densenet49, densenet121
import os
import time
from split import ruifeng_by_format, ruifeng_by_date
from collections import OrderedDict

label_index = {
        "BiaoMianBuLiang": 0,
        "DaoXian": 1,
        "LouGu": 2,
        "LouHan": 3,
        "Ok": 4,
        "QiPao": 5,
        "QiePian": 6,
        "QueJiao": 7,
        "YiJiao": 8,
        "ZaZhi": 9
    }

config = {
    "data_path": "./../data/",
    "validation_rate": 0.9,
    "del_uncertain": True
}

def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))

def model_snapshot(model, new_file, old_file=None, verbose=False):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if old_file and os.path.exists(expand_user(old_file)):
        if verbose:
            print("Removing old model {}".format(expand_user(old_file)))
        os.remove(expand_user(old_file))
    if verbose:
        print("Saving model to {}".format(expand_user(new_file)))

    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if v.is_cuda:
            v = v.cpu()
        state_dict[k] = v
    torch.save(state_dict, expand_user(new_file))

#Normalizing The Dataset
transformed = transforms.Compose([transforms.ToTensor()])

#Downloading The Trainset
#trainset = dataset.FashionMNIST('FMNIST_TRAIN', download = True, train = True, transform = transformed)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)
train_pos, train_neg, val_pos, val_neg = ruifeng_by_date(config, label_index, training = True)

#Downloading The TestSet
#testset = dataset.FashionMNIST('FMNIST_TEST', download = True, train = False, transform = transformed)
#testloader =  torch.utils.data.DataLoader(testset, batch_size = 64, shuffle = True)

#Spliting The Dataset
#dataiter = iter(trainloader)
#images, labels = dataiter.next()

model = densenet49(num_classes = 2, channels = 3)
model.cuda()
print(model)

#Initializing Loss and Optimizer
# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#Training The Network
epoch = 100
log_interval = 10
test_interval = 1

best_acc, old_file = 0.0, None
t_begin = time.time()
for x in range(epoch):
    for batch_idx, (images, labels) in enumerate(trainloader):
        indx_target = labels.clone()
        #Flattening The Image
        #print(images.shape)
        #images = images.view(images.shape[0], -1)
        images, labels = images.cuda(), labels.cuda()
        #Clearing Previous Gradients
        optimizer.zero_grad()
    
        #-> Forward Pass 
        #-> Calculating Loss 
        #-> Calculating Gradients Through Backward Pass
        #-> Updating the Weights, using optimizer.step()
    
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            pred = output.data.max(1)[1]
            correct = pred.cpu().eq(indx_target).sum()
            acc = float(correct) / float(len(images))
            print('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                    x, batch_idx * len(images), len(trainloader.dataset),
                    loss.data.item(), acc, optimizer.param_groups[0]['lr']))
    
    elapse_time = time.time() - t_begin
    speed_epoch = elapse_time / (x + 1)
    speed_batch = speed_epoch / len(trainloader)
    eta = speed_epoch * epoch - elapse_time
    print("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
            elapse_time, speed_epoch, speed_batch, eta))
    
    model_snapshot(model, os.path.join('./../model/', 'latest.pth'))

    if x % test_interval == 0:
        test_loss = 0
        correct = 0
        for data, target in testloader:
            indx_target = target.clone()
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += criterion(output, target).data.item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.cpu().eq(indx_target).sum()

        test_loss = test_loss / len(testloader) # average over number of mini-batch
        acc = 100. * float(correct) / float(len(testloader.dataset))
        print('\tTest set: Average loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)'.format(
                test_loss, correct, len(testloader.dataset), acc))
        if acc > best_acc:
            new_file = os.path.join('./../model/', 'best-{}.pth'.format(x))
            model_snapshot(model, new_file, old_file=old_file, verbose=True)
            best_acc = acc
            old_file = new_file

print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time()-t_begin, best_acc))

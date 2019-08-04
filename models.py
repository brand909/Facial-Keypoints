## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3) #32x224x224
        #self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2,2) #32x112x112
        #self.drop1 = nn.Dropout2d(0.1)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2) #64x112x112
        #self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2,2) #64x56x56
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2) #128x56x56
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2,2) #128x28x28
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) #256x28x28
        #self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2,2) #256x14x14
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1) #512x14x14
        #self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2,2) #512x7x7
        
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.drop1 = nn.Dropout(0.6)
        
        self.fc2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(4096, 136)  
        
               
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        #x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        
        x = x.view(x.size(0), -1) # flatten
        
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        
        x = self.fc3(x)
    
        return x
    
class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        # input = 1x224x224
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3) #32x224x224
        self.pool1 = nn.MaxPool2d(2,2) #32x112x112
        self.drop1 = nn.Dropout2d(0.1)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2) #64x112x112
        self.pool2 = nn.MaxPool2d(2,2) #64x56x56
        self.drop2 = nn.Dropout2d(0.2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) #128x56x56
        self.pool3 = nn.MaxPool2d(2,2) #128x28x28
        self.drop3 = nn.Dropout2d(0.3)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0) #256x28x28
        self.pool4 = nn.MaxPool2d(2,2) #256x14x14
        self.drop4 = nn.Dropout2d(0.4)
        
        self.fc1 = nn.Linear(256*14*14, 1024)
        self.drop5 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(1024, 1024)
        self.drop6 = nn.Dropout(0.6)
        
        self.fc3 = nn.Linear(1024, 136)  
        
               
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        x = self.drop1(self.pool1(F.elu(self.conv1(x))))
        x = self.drop2(self.pool2(F.elu(self.conv2(x))))
        x = self.drop3(self.pool3(F.elu(self.conv3(x))))
        x = self.drop4(self.pool4(F.elu(self.conv4(x))))
        
        x = x.view(x.size(0), -1) # flatten
        
        x = self.drop5(F.elu(self.fc1(x)))
        x = self.drop6(F.elu(self.fc2(x)))  
        x = self.fc3(x)
    
        return x    


class PaperNet(nn.Module):

    def __init__(self):
        super(PaperNet, self).__init__()
        # input = 1x224x224
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=0) #32x221x221
        self.pool1 = nn.MaxPool2d((2,2), stride=2) #32x110x110
        self.drop1 = nn.Dropout2d(0.1)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0) #64x108x108
        self.pool2 = nn.MaxPool2d((2,2), stride=2) #64x54x54
        self.drop2 = nn.Dropout2d(0.2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0) #128x53x53
        self.pool3 = nn.MaxPool2d((2,2), stride=2) #128x26x26
        self.drop3 = nn.Dropout2d(0.3)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0) #256x26x26
        self.pool4 = nn.MaxPool2d((2,2), stride=2) #256x13x13
        self.drop4 = nn.Dropout2d(0.4)
        
        self.fc1 = nn.Linear(256*13*13, 1024)
        self.drop5 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(1024, 1024)
        self.drop6 = nn.Dropout(0.6)
        
        self.fc3 = nn.Linear(1024, 136)  
        
               
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        x = self.drop1(self.pool1(F.elu(self.conv1(x))))
        x = self.drop2(self.pool2(F.elu(self.conv2(x))))
        x = self.drop3(self.pool3(F.elu(self.conv3(x))))
        x = self.drop4(self.pool4(F.elu(self.conv4(x))))
        
        x = x.view(x.size(0), -1) # flatten
        
        x = self.drop5(F.elu(self.fc1(x)))
        x = self.drop6(F.elu(self.fc2(x)))  
        x = self.fc3(x)
    
        return x


class LongNet(nn.Module):

    def __init__(self):
        super(LongNet, self).__init__()
        # input=1x224x224
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=0) #32x218x218
        self.pool1 = nn.MaxPool2d(2,2) #32x109x109
        self.drop1 = nn.Dropout2d(0.1)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=6, stride=1, padding=0) #64x104x104
        self.pool2 = nn.MaxPool2d(2,2) #64x52x52
        self.drop2 = nn.Dropout2d(0.2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0) #128x48x48
        self.pool3 = nn.MaxPool2d(2,2) #128x24x24
        self.drop3 = nn.Dropout2d(0.3)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0) #256x21x21
        self.pool4 = nn.MaxPool2d(2,2) #256x10x10
        self.drop4 = nn.Dropout2d(0.4)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0) #512x8x8
        self.pool5 = nn.MaxPool2d(2,2) #512x4x4
        self.drop5 = nn.Dropout2d(0.5)
        
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=2, stride=1, padding=0) #1024x3x3
        self.pool6 = nn.MaxPool2d((2,2), padding=1) #1024x2x2
        self.drop6 = nn.Dropout2d(0.6)
        
        self.fc1 = nn.Linear(1024*2*2, 1024)
        self.drop7 = nn.Dropout(0.6)
        
        self.fc2 = nn.Linear(1024, 1024)
        self.drop8 = nn.Dropout(0.6)
        
        self.fc3 = nn.Linear(1024, 136)  
        
               
    def forward(self, x):
        x = self.drop1(self.pool1(F.elu(self.conv1(x))))
        x = self.drop2(self.pool2(F.elu(self.conv2(x))))
        x = self.drop3(self.pool3(F.elu(self.conv3(x))))
        x = self.drop4(self.pool4(F.elu(self.conv4(x))))
        x = self.drop5(self.pool5(F.elu(self.conv5(x))))
        x = self.drop6(self.pool6(F.elu(self.conv6(x))))
        
        x = x.view(x.size(0), -1) # flatten
        
        x = self.drop7(F.elu(self.fc1(x)))
        x = self.drop8(F.elu(self.fc2(x)))
        x = self.fc3(x)
    
        return x
    

    
    
class AltNet(nn.Module):
    def __init__(self):
        super(AltNet, self).__init__()
        # input=1x224x224
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=0) #32x218x218
        self.pool1 = nn.MaxPool2d(2,2) #32x109x109
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0) #64x105x105
        self.pool2 = nn.MaxPool2d(2,2) #64x52x52
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0) #128x48x48
        self.pool3 = nn.MaxPool2d(2,2) #128x24x24
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0) #256x22x22
        self.pool4 = nn.MaxPool2d(2,2) #256x11x11
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0) #512x9x9
        self.pool5 = nn.MaxPool2d((2,2), padding=1) #512x5x5
        
        self.fc1 = nn.Linear(512*5*5, 4096)
        self.drop6 = nn.Dropout(0.6)
        
        self.fc2 = nn.Linear(4096, 4096)
        self.drop7 = nn.Dropout(0.6)
        
        self.fc3 = nn.Linear(4096, 136)  
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        
        x = x.view(x.size(0), -1) # flatten
        
        x = self.drop6(F.relu(self.fc1(x)))
        x = self.drop7(F.relu(self.fc2(x)))
        x = self.fc3(x)
    
        return x
    

import torchvision.models as models

class TransferRes18(nn.Module):
    def __init__(self):
        super(TransferRes18, self).__init__()
        
        res = models.resnet18(pretrained=True)
        for param in res.parameters():
            param.requires_grad_(True)        # freezes pre-trained model weights if set to False
        modules = list(res.children())[:-1]    # deletes the last fully-connected layer
        
        self.res = nn.Sequential(*modules)
        self.fc1 = nn.Linear(res.fc.in_features, 1024)
        self.fc2 = nn.Linear(1024, 136)
        
        self.drop = nn.Dropout(0.6)
    
    def forward(self, x):
        x = self.res(x.repeat(1,3,1,1)) # pretrained models require 3 channel input - stack 3 grayscale
        x = x.view(x.size(0), -1) # flatten
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    
class TransferRes50(nn.Module):
    def __init__(self):
        super(TransferRes50, self).__init__()
        
        res = models.resnet50(pretrained=True)
        for param in res.parameters():
            param.requires_grad_(False)        # freezes pre-trained model weights
        modules = list(res.children())[:-1]    # deletes the last fully-connected layer
        
        self.res = nn.Sequential(*modules)
        self.fc1 = nn.Linear(res.fc.in_features, 1024)
        self.fc2 = nn.Linear(1024, 136)
    
    def forward(self, x):
        x = self.res(x.repeat(1,3,1,1)) # pretrained models require 3 channel input - stack 3 grayscale
        x = x.view(x.size(0), -1) # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    
class TransferRes152(nn.Module):
    def __init__(self):
        super(TransferRes152, self).__init__()
        
        res = models.resnet152(pretrained=True)
        for param in res.parameters():
            param.requires_grad_(False)        # freezes pre-trained model weights
        modules = list(res.children())[:-1]    # deletes the last fully-connected layer
        
        self.res = nn.Sequential(*modules)
        self.fc1 = nn.Linear(res.fc.in_features, 1024)
        self.fc2 = nn.Linear(1024, 136)
    
    def forward(self, x):
        x = self.res(x.repeat(1,3,1,1)) # pretrained models require 3 channel input - stack 3 grayscale
        x = x.view(x.size(0), -1) # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class TransferVGG(nn.Module):
    def __init__(self):
        super(TransferVGG, self).__init__()
        
        vgg = models.vgg16(pretrained=True)
        for param in vgg.parameters():
            param.requires_grad_(False)        # freezes pre-trained model weights
        modules_1 = list(vgg.children())[0]
        modules_2 = list(vgg.children())[1][:-1]  # deletes last fully-connected layer
        
        self.vgg_m1 = nn.Sequential(*modules_1)
        self.vgg_m2 = nn.Sequential(*modules_2)
        self.fc1 = nn.Linear(4096,136)
        
    def forward(self, x):
        x = self.vgg_m1(x.repeat(1,3,1,1)) # pretrained models require 3 channel input - stack 3 grayscale
        x = x.view(x.size(0), -1) # flatten
        x = self.vgg_m2(x)
        x = self.fc1(x)
        return x
        
        
class TransferInceptionV3(nn.Module):
    def __init__(self):
        super(TransferInceptionV3, self).__init__()
        
        v3 = models.inception_v3(pretrained=True)
        for param in v3.parameters():
            param.requires_grad_(False)
        modules = list(v3.children())[:-1]
        
        self.v3 = nn.Sequential(*modules)
        
        
        
        

import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import torch 

class SmallNet(nn.Module):
    def __init__(self,num_classes=3):
        super(SmallNet,self).__init__()
        # define the conv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        
        self.num_flatten = 64 * 8 * 8
        self.fc1 = nn.Linear(self.num_flatten,512)
        self.out = nn.Linear(512,num_classes)
        
    
    def forward(self,images,targets=None):
        x = self.conv1(images)
        x = F.dropout(x, p=0.25)
        x = self.conv2(x)
        x = F.dropout(x, p=0.25)
        x = x.view(-1,self.num_flatten)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.25)
        x = self.out(x)
        if targets is not None:
            loss = nn.CrossEntropyLoss()
            loss = loss(x, targets)
            return x,loss
        return x,None 

if __name__ == "__main__":
    model = SmallNet(num_classes=3)
    img = torch.rand((1,3,32,32))
    x,loss = model(img,torch.tensor([1],dtype=torch.long))
import torch
import torch.nn as nn


class DAVE_2(nn.Module):
    def __init__(self):
        super(DAVE_2,self).__init__()

        # input(3,66,220) 
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 26, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)

        self.fc1 = nn.Linear(64*1*18, 100)
        self.fc2 = nn.Linear(100,10)
        self.fc3 = nn.Linear(10,1)


    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        
        x = torch.relu(self.conv5(x))

        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

if __name__ == '__main__':
    model = DAVE_2()
    print(model)
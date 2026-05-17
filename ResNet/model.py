import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels,  use_1conv = False, strides = 1):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride = strides)
        self.Conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = strides)
        else:
            self.conv3 = None


    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.Conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)
    


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels= 1, out_channels= 64, kernel_size= 7, stride= 2, padding= 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)
        )
        self.b2 = nn.Sequential(
            Residual(64, 64, use_1conv= False, strides= 1),
            Residual(64, 64, use_1conv= False, strides= 1)
        )
        self.b3 = nn.Sequential(
            Residual(64, 128, use_1conv= True, strides= 2),
            Residual(128, 128, use_1conv= False, strides= 1)
        )
        self.b4 = nn.Sequential(
            Residual(128, 256, use_1conv= True, strides= 2),
            Residual(256, 256, use_1conv= False, strides = 1)
        )
        self.b5 = nn.Sequential(
            Residual(256, 512, use_1conv= True, strides= 2),
            Residual(512, 512, use_1conv= False, strides= 1)
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

        '''
        炮哥的说法是如果我们使用了batchnormal可以不用凯明初始化了
        '''
        #尝试和前面的GoodNe类似初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode= "fan_out", nonlinearity= "relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, X):
        X = self.b1(X)
        X = self.b2(X)
        X = self.b3(X)
        X = self.b4(X)
        X = self.b5(X)
        return self.b6(X)

   
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18().to(device)
    print(summary(model, (1, 224, 224)))
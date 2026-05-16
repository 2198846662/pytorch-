import torch
import torch.nn as nn
from torchsummary import summary


class Lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels= 6, kernel_size=5, padding= 2)
        self.pool1 = nn.AvgPool2d(kernel_size= 2, stride= 2)
        self.conv2  = nn.Conv2d(in_channels= 6, out_channels= 16, kernel_size= 5)
        self.pool2 = nn.AvgPool2d(kernel_size= 2, stride= 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features= 16*5*5, out_features= 120)
        self.fc2 = nn.Linear(in_features= 120, out_features= 84)
        self.fc3 = nn.Linear(in_features= 84, out_features= 10)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.sigmoid(self.conv1(x))
        x = self.pool1(x)
        x = self.sigmoid(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # print(torch.cuda.device_count())
    # print(torch.cuda.get_device_name(0))
    model = Lenet()
    model.to(device)
    summary(model, (1, 28, 28))
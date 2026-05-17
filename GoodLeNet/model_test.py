import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from model import GoodLeNet
from torchvision import transforms
from torchvision.datasets import FashionMNIST

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def test_data_process():
    test_data_process_data = FashionMNIST(root= "./data", train= False, download= True, transform= transform)

    test_loader = DataLoader(dataset= test_data_process_data, batch_size= 1, shuffle= False, num_workers= 0)

    return test_loader


def test_model_process(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_loss = 0.0
    test_acc = 0.0
    test_num = 0    

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, pred = torch.max(outputs, 1)
            test_acc += (pred == labels).sum().item()
            test_num += images.size(0)

            #预测值
            result = pred.item()
            #真实值
            real = labels.item()
            print(f"预测值: {result} -------------------------------- 真实值: {real}")


    test_loss = test_loss / test_num
    test_acc = test_acc / test_num

    print(f"Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    test_loader = test_data_process()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoodLeNet()
    model.to(device)   
    model.load_state_dict(torch.load("result/best_model.pth"))
    test_model_process(model, test_loader)


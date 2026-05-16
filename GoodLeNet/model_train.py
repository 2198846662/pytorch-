import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torchvision import transforms
from torchvision.datasets import FashionMNIST 
from torch.utils.data import DataLoader, random_split
import torch
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 2. 解决中文乱码（彻底消除 Glyph 警告）
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


from model import GoodLeNet

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


#数据加载函数
def train_val_data_process():
    train_data = FashionMNIST(root= "./data", train= True, download= True, transform= transform)

    #按照 8:1 进行划分
    train_data, val_data = random_split(train_data, [round(len(train_data)*0.8), round(len(train_data)*0.2)])
 
    train_dataloader =  DataLoader(dataset= train_data, batch_size= 32, shuffle= True, num_workers= 0)
    val_dataloader =  DataLoader(dataset= val_data, batch_size= 32, shuffle= False, num_workers= 0)


    return train_dataloader, val_dataloader


#模型训练函数
def train_model_process(model, train_dataloader, val_dataloader, epochs= 10):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    #定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
    
    #模型放到gpu
    model.to(device)

    #最优准确率
    best_acc = 0.0

    #损失列表
    train_loss_list = []
    val_loss_list = []

    #准确率列表
    train_acc_list = []
    val_acc_list = []

    #保存时间
    since = time.time()

    for epoch in range(epochs):
        print(f"Epoch: {epoch}/{epochs-1}")
        print('-'*10)


        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        train_num = 0
        val_num = 0


        #打开训练模式
        model.train()
        for _, (b_x, b_y) in enumerate(train_dataloader):
            #数据放到gpu
            b_x, b_y = b_x.to(device), b_y.to(device)
            
            outputs = model(b_x)
            predicted = torch.argmax(outputs, dim= 1)


            loss = criterion(outputs, b_y)  #里面包含了softmax函数，所以输出不需要再经过softmax函数了
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_x.size(0) #乘以batch_size是为了得到整个数据集的损失和  
            train_acc += (predicted == b_y).sum().item() #计算正确的数量
            train_num += b_x.size(0) #计算总的数量

        
        #验证模型
        model.eval() 
        with torch.no_grad():
            for _, (b_x, b_y) in enumerate(val_dataloader):
                b_x, b_y = b_x.to(device), b_y.to(device)
                outputs = model(b_x)
                predicted = torch.argmax(outputs, dim= 1)

                loss = criterion(outputs, b_y)

                val_loss += loss.item() * b_x.size(0)
                val_acc += (predicted == b_y).sum().item()
                val_num += b_x.size(0)

        #计算每一轮的loss和acc
        train_loss_list.append(train_loss / train_num)
        val_loss_list.append(val_loss / val_num)
        train_acc_list.append(train_acc / train_num)
        val_acc_list.append(val_acc / val_num)

        #打印训练和验证的loss和acc
        print(f"Train Loss: {train_loss_list[-1]:.4f} Train Acc: {train_acc_list[-1]:.4f}")
        print(f"Val Loss: {val_loss_list[-1]:.4f} Val Acc: {val_acc_list[-1]:.4f}")


        if val_acc_list[-1] > best_acc:
            best_acc = val_acc_list[-1]
            torch.save(model.state_dict(), "./result/best_model.pth")


        time_elapsed = time.time() - since
        print(f"目前训练时长已经距开始继续了：{time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")

    #保存数据
    train_process = pd.DataFrame({
        "epoch":range(1, epochs+1),
        "train_loss": train_loss_list,
        "val_loss": val_loss_list,
        "train_acc": train_acc_list,
        "val_acc": val_acc_list
            
        })

    return train_process

def matplot_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process["train_loss"], label="Train Loss", color="blue")
    plt.plot(train_process["epoch"], train_process["val_loss"], label="Val Loss", color="orange")
    plt.legend()
    plt.title("Loss曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_acc"], label="Train Acc", color="blue")
    plt.plot(train_process["epoch"], train_process["val_acc"], label="Val Acc", color="orange")
    plt.legend() 
    plt.title("Acc曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")

    plt.savefig("./result/image/loss_acc_curve.png", dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    #获取数据
    train_dataloader, val_dataloader = train_val_data_process()
    #实例化模型
    model = GoodLeNet()
    #训练模型
    train_process = train_model_process(model, train_dataloader, val_dataloader, epochs= 2)
    #画图
    matplot_loss(train_process)

     
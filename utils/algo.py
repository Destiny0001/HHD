import torch
import torchvision
import torchvision.transforms as transforms
import hash_model as image_hash_model
import numpy as np
import time
from cac import test_accuracy
from condition import add_label_noise
import torch.optim as optim
import torch.nn as nn

# 参数定义
top_k = 1000
batch_size = 10
epochs = 100
lr = 0.001
weight_decay = 5* 10 ** -4
lambda1 = 0.01
hash_bits = 64
model_name = "resnet34"
device = torch.device("cuda")

# 数据加载
def load_dataset():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

# 模型训练
def train_model(model, trainloader, testloader, label_hash_codes, noise_rate):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.3, last_epoch=-1)

    total_loss = []
    accuracy_list = []
    best_accuracy = 0.0

    for epoch in range(epochs):
        
        if epoch % 5 == 0:
            accuracy = test_accuracy(model, testloader, label_hash_codes, device)
            print(f'Epoch {epoch}, Test Accuracy: {accuracy}%')
            accuracy_list.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), f'./model/nr_{noise_rate}_{model_name}.pth')
                print(f"Model saved with accuracy: {best_accuracy:.2f}%")

        for iter, (inputs, labels) in enumerate(trainloader):
            labels = add_label_noise(labels.numpy(), noise_rate=noise_rate, num_classes=10)
            labels = torch.from_numpy(labels)

            inputs= inputs.to(device)
            outputs = model(inputs)
            cat_codes = label_hash_codes[labels].to(device) 

            criterion = nn.BCELoss().to(device)
            center_loss = criterion(0.5*(outputs+1), 0.5*(cat_codes+1))
            Q_loss = torch.mean((torch.abs(outputs)-1.0)**2)
            loss = center_loss + lambda1*Q_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss.append(loss.item())

            if iter % 500 == 0:
                print(f'Epoch: {epoch}, Iter: {iter}, Loss: {loss.item()}')

    return total_loss, accuracy_list

def main():
    trainloader, testloader = load_dataset()

    # 设定不同的噪声率
    noise_rates = [0.0, 0.2, 0.4, 0.6, 0.8]

    for noise_rate in noise_rates:
        print(f"Training with noise_rate: {noise_rate}")
        
        # 加载标签哈希码
        with open('../labels/64_cifar10_10_class.pkl', 'rb') as f:
            label_hash_codes = torch.load(f)
        label_hash_codes.to(device)

        # 初始化模型
        model = image_hash_model.HASH_Net(model_name, hash_bits).to(device)

        # 进行训练
        total_loss, accuracy_list = train_model(model, trainloader, testloader, label_hash_codes, noise_rate)

        # 输出结果
        print(f'Finished Training with noise_rate: {noise_rate}')
        print(f'Epoch accuracies with noise_rate {noise_rate}:', accuracy_list)
        
        # 可以在这里添加代码来保存每个噪声率的训练结果，例如损失和准确率

if __name__ == '__main__':
   main()

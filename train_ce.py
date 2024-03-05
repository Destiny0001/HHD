import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import logging
import numpy as np
from dataset import CIFAR10Custom  # 确保这里正确地从您的dataset.py文件导入CIFAR10Custom类
from utils.cac import test_accuracy  # 确保test_accuracy函数可以从cac模块导入
import os


# 数据加载函数
def load_dataset(noise_type, noise_rate=0.0, batch_size=128, num_workers=10):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transfrom_train = transforms.Compose([
                                        #transforms.RandomHorizontalFlip(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    transform_test = transforms.Compose([
                
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ])

    train_dataset = CIFAR10Custom(root='./data', train=True, transform=transform, noise_type=noise_type, noise_rate=noise_rate)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = CIFAR10Custom(root='./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

# 模型训练函数
def train_model(model, trainloader, testloader, epochs=100):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        logging.info(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

        # 测试集上评估模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        logging.info(f'Test Accuracy: {100 * correct / total}%')

    print('Finished Training')

# 加载预训练的ResNet34模型
def get_pretrained_resnet(num_classes=10):
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    return model

if __name__ == '__main__':
    noisetype = "sym"
    noise_rates = [0.6,0.4,0.2,0.6,0.8]
    epochs = 100
    lr = 0.02
    weight_decay = 10 ** -5
    model_name = "resnet34base"
    batchsize = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for noiserate in noise_rates:
        logging.basicConfig(filename=f'./logs/{model_name}_{noisetype}_test_nr.log',level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f'Started training with: {noisetype}-{noiserate} ')
        logging.info(f'Training Configuration: batch_size={batchsize}, epochs={epochs}, lr={lr}, weight_decay={weight_decay}, model_name={model_name}, device={device}')
        trainloader, testloader = load_dataset(noise_type=noisetype, batch_size=batchsize, noise_rate=noiserate)  # 替换'your_noise_type'为实际噪声类型
        model = get_pretrained_resnet(10)  # CIFAR-10有10个类别
        train_model(model, trainloader, testloader, epochs)

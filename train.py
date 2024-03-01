import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import logging
from dataset import CIFAR10Custom # 确保这里正确地从您的dataset.py文件导入CIFAR10Custom类
import utils.hash_model as image_hash_model # 确保您的hash_model模块包含了HASH_Net定义
from utils.cac import test_accuracy # 确保test_accuracy函数可以从cac模块导入

# 设置日志
logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# 参数定义
batch_size = 64
epochs = 100
lr = 0.01
weight_decay = 1e-5
lambda1 = 0.01
hash_bits = 64
model_name = "resnet34"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_dataset(noise_type, batch_size=64):
    # 使用CIFAR10Custom类加载数据集
    train_dataset = CIFAR10Custom(root='./data', train=True, transform=transform, noise_type=noise_type, noise_rate=0.2)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CIFAR10Custom(root='./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# 模型训练函数
def train_model(model, trainloader, testloader, epochs=100):
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

        # 计算测试集上的准确率
        accuracy = test_accuracy(model, testloader, device)
        logging.info(f'Epoch {epoch}: Test Accuracy: {accuracy}%')

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Test Accuracy: {accuracy}%')

def main():
    # 定义噪声类型
    noise_types = ['clean_label', 'worse_label', 'aggre_label', 'random_label1', 'random_label2', 'random_label3']

    # 加载模型
    model = image_hash_model.HASH_Net(model_name, hash_bits).to(device)

    for noise_type in noise_types:
        trainloader, testloader = load_dataset(noise_type=noise_type, batch_size=batch_size)
        logging.info(f'Start Training with noise type: {noise_type}')
        train_model(model, trainloader, testloader, epochs=epochs)
        logging.info(f'Finished Training with noise type: {noise_type}')

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import logging
import numpy as np
from dataset import CIFAR10Custom # 确保这里正确地从您的dataset.py文件导入CIFAR10Custom类
import utils.hash_model as image_hash_model # 确保您的hash_model模块包含了HASH_Net定义
from utils.cac import test_accuracy # 确保test_accuracy函数可以从cac模块导入
import time
# 参数定义
batch_size = 128
epochs = 100
lr = 0.01
weight_decay = 1e-5
lambda1 = 0.01
hash_bits = 64
model_name = "resnet34"
device = torch.device("cuda")



# 定义数据转换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_dataset(noise_type, noise_rate=0.0, batch_size=64):
    # 使用CIFAR10Custom类加载数据集
    train_dataset = CIFAR10Custom(root='./data', train=True, transform=transform, noise_type=noise_type, noise_rate=noise_rate)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 80)

    test_dataset = CIFAR10Custom(root='./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers = 80)

    return train_loader, test_loader

# 模型训练函数
def train_model(model, trainloader, testloader,label_hash_codes, epochs=100):
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    best_accuracy = 0.0
  
    for epoch in range(epochs):
        model.train()
      
        for iter, (inputs, labels) in enumerate(trainloader):
         
            labels = torch.from_numpy(np.array(labels))
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

        # 计算测试集上的准确率
        accuracy = test_accuracy(model, testloader, label_hash_codes, device)
        logging.info(f'Epoch {epoch}: Test Accuracy: {accuracy}%')
        print(f'Epoch {epoch}/{epochs}, Test Accuracy: {accuracy}%')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'./model/nt_{trainloader.dataset.noise_type}_{model_name}.pth')
            print(f"Model saved with accuracy: {best_accuracy:.2f}%")
            logging.info(f"Model saved with accuracy: {best_accuracy:.2f}%")
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()

def test_nr(noise_type = 'asym'):
    logging.basicConfig(filename=f'./logs/{model_name}_{noise_type}_test_nr.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info(f'Training Configuration: batch_size={batch_size}, epochs={epochs}, lr={lr}, weight_decay={weight_decay}, lambda1={lambda1}, hash_bits={hash_bits}, model_name={model_name}, device={device}')
     # 加载标签哈希码
    with open('./labels/64_cifar10_10_class.pkl', 'rb') as f:
        label_hash_codes = torch.load(f)
    label_hash_codes.to(device)
    
    #noise_types = ['aggre_label','worse_label', 'random_label1', 'random_label2', 'random_label3','clean_label']
    noise_rates = [0.2,0.4,0.6,0.8,0.0]
    # 加载模型
    model = image_hash_model.HASH_Net(model_name, hash_bits).to(device)
    noise_type = noise_type
    for noise_rate in noise_rates:
        trainloader, testloader = load_dataset(noise_type=noise_type, batch_size=batch_size, noise_rate=noise_rate)
        logging.info(f'Start Training with noise rate: {noise_type}-{noise_rate}')
        train_model(model, trainloader, testloader, label_hash_codes,epochs=epochs)
        logging.info(f'Finished Training with noise rate: {noise_type}-{noise_rate}')


def test_cifarn():
    logging.basicConfig(filename=f'./logs/{model_name}_test_cifarn.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info(f'Training Configuration: batch_size={batch_size}, epochs={epochs}, lr={lr}, weight_decay={weight_decay}, lambda1={lambda1}, hash_bits={hash_bits}, model_name={model_name}, device={device}')
     # 加载标签哈希码
    with open('./labels/64_cifar10_10_class.pkl', 'rb') as f:
        label_hash_codes = torch.load(f)
    label_hash_codes.to(device)
    
    noise_types = ['aggre_label','worse_label', 'random_label1', 'random_label2', 'random_label3','clean_label']
    #noise_rates = [0.2,0.4,0.6,0.8,0.0]
    # 加载模型
    model = image_hash_model.HASH_Net(model_name, hash_bits).to(device)

    for noise_type in noise_types:
        trainloader, testloader = load_dataset(noise_type=noise_type, batch_size=batch_size)
        noise_rate = trainloader.dataset.noise_rate
        logging.info(f'Start Training with: {noise_type}-{noise_rate}')
        train_model(model, trainloader, testloader, label_hash_codes,epochs=epochs)
        logging.info(f'Finished Training with: {noise_type}-{noise_rate}')

if __name__ == '__main__':

    test_cifarn()

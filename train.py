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
from utils.denoise import label_refurb
from utils.lr_scheduler import WarmupLR
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 参数定义
batch_size = 128
epochs = 100
lr = 0.02
weight_decay = 10 ** -5
lambda1 = 0.02
hash_bits = 64
model_name = "resnet34"
device = torch.device("cuda")



def load_dataset(noise_type, noise_rate=0.0, batch_size= 256, num_workers = 40):
    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 使用CIFAR10Custom类加载数据集
    train_dataset = CIFAR10Custom(root='./data', train=True, transform=transform, noise_type=noise_type, noise_rate=noise_rate)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)

    test_dataset = CIFAR10Custom(root='./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers = num_workers)

    return train_loader, test_loader

# 模型训练函数
def train_model(model, trainloader, testloader,label_hash_codes, epochs=epochs, hash_bits = 64):
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8, last_epoch=-1)
    #scheduler = WarmupLR(optimizer,warmup_epochs=20,initial_lr=0.002)
    best_accuracy = 0.0
  
    for epoch in range(epochs):
        model.train()
        scheduler.step()
        # 计算测试集上的准确率
        accuracy = test_accuracy(model, testloader, label_hash_codes, device)
        logging.info(f'Epoch {epoch}: Test Accuracy: {accuracy}%')
        print(f'Epoch {epoch}/{epochs}, Test Accuracy: {accuracy}%')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'./model/nt_{trainloader.dataset.noise_type}_{model_name}.pth')
            print(f"Model saved with accuracy: {best_accuracy:.2f}%")
            logging.info(f"Model saved with accuracy: {best_accuracy:.2f}%")
            
        for iter, (inputs, labels) in enumerate(trainloader):
         
            labels = torch.from_numpy(np.array(labels))
            inputs= inputs.to(device)
            outputs = model(inputs)
            criterion = nn.BCELoss().to(device)
            if(epoch<6):
                outputs,labels = label_refurb(labels,outputs,label_hash_codes,hash_bits,device, False)
            else:
                outputs,labels = label_refurb(labels,outputs,label_hash_codes,hash_bits,device, False)
            cat_codes = label_hash_codes[labels.cpu()].to(device) 
            
            try:
                # 假设outputs和cat_codes是你想要检查的变量
                assert outputs.min() >= -1 and outputs.max() <= 1, "outputs值不在[-1, 1]范围内"
                assert cat_codes.min() >= -1 and cat_codes.max() <= 1, "cat_codes值不在[-1, 1]范围内"
            except AssertionError as e:
                print(f"断言错误: {e}")
                print("outputs的最小值:", outputs.min())
                print("outputs的最大值:", outputs.max())
                print("cat_codes的最小值:", cat_codes.min())
                print("cat_codes的最大值:", cat_codes.max())
                logging.info(f"断言错误: {e}")
                logging.info("outputs的最小值:", outputs.min())
                logging.info("outputs的最大值:", outputs.max())
                logging.info("cat_codes的最小值:", cat_codes.min())
                logging.info("cat_codes的最大值:", cat_codes.max())
  
            center_loss = criterion(0.5*(outputs+1), 0.5*(cat_codes+1))
            Q_loss = torch.mean((torch.abs(outputs)-1.0)**2)
            loss = center_loss + lambda1*Q_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch%1==0:
                torch.cuda.empty_cache()

      

      
def test_nr(noisetype = None):
    device = torch.device("cuda")
    logging.basicConfig(filename=f'./logs/{model_name}_{noisetype}_test_nr.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info(f'Training Configuration: batch_size={batch_size}, epochs={epochs}, lr={lr}, weight_decay={weight_decay}, lambda1={lambda1}, hash_bits={hash_bits}, model_name={model_name}, device={device}')
     # 加载标签哈希码
    with open(f'./labels/{hash_bits}_cifar10_10_class.pkl', 'rb') as f:
        label_hash_codes = torch.load(f)
    label_hash_codes.to(device)
    
    #noise_types = ['aggre_label','worse_label', 'random_label1', 'random_label2', 'random_label3','clean_label']
    noise_rates = [0.2,0.4,0.6,0.8,0.0]
    
    for noise_rate in noise_rates:
         # 加载模型
        model = image_hash_model.HASH_Net(model_name, hash_bits).to(device)
        trainloader, testloader = load_dataset(noise_type=noisetype, batch_size=batch_size, noise_rate=noise_rate)
        logging.info(f'Start Training with: {noisetype}-{noise_rate}')
        train_model(model, trainloader, testloader, label_hash_codes,epochs=epochs)
        logging.info(f'Finished Training with: {noisetype}-{noise_rate}')
    


def test_cifarn():
    device = torch.device("cuda")
    logging.basicConfig(filename=f'./logs/{model_name}_test_cifarn.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info(f'Training Configuration: batch_size={batch_size}, epochs={epochs}, lr={lr}, weight_decay={weight_decay}, lambda1={lambda1}, hash_bits={hash_bits}, model_name={model_name}, device={device}')
    with open(f'./labels/{hash_bits}_cifar10_10_class.pkl', 'rb') as f:
        label_hash_codes = torch.load(f)
    label_hash_codes.to(device)
    
    noise_types = ['aggre_label','random_label1','worse_label','random_label2', 'random_label3','clean_label']
    #noise_rates = [0.2,0.4,0.6,0.8,0.0]
    

    for noise_type in noise_types:
        # 加载模型
        model = image_hash_model.HASH_Net(model_name, hash_bits).to(device)
        trainloader, testloader = load_dataset(noise_type=noise_type, batch_size=batch_size)
        noise_rate = trainloader.dataset.noise_rate
        logging.info(f'Start Training with: {noise_type}-{noise_rate}')
        train_model(model, trainloader, testloader, label_hash_codes,epochs=epochs)
        logging.info(f'Finished Training with: {noise_type}-{noise_rate}')

def test_hashbits():
    epochs=100
    device = torch.device("cuda")
    logging.basicConfig(filename=f'./logs/{model_name}_test_hashbits.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info(f'Training Configuration: batch_size={batch_size}, epochs={epochs}, lr={lr}, weight_decay={weight_decay}, lambda1={lambda1}, hash_bits={hash_bits}, model_name={model_name}, device={device}')
    
    hashbits = [128,32,16,64]
    

    for hashbit in hashbits:
        with open(f'./labels/{hashbit}_cifar10_10_class.pkl', 'rb') as f:
            label_hash_codes = torch.load(f)
        label_hash_codes.to(device)
    
        # 加载模型
        model = image_hash_model.HASH_Net(model_name, hashbit).to(device)
        trainloader, testloader = load_dataset(noise_type='sym', noise_rate=0.4,batch_size=batch_size)
        logging.info(f'Start Training with hash_bits: {hashbit}')
        train_model(model, trainloader, testloader, label_hash_codes,epochs=epochs,hash_bits=hashbit)
        logging.info(f'Finished Training with hash_bits: {hashbit}')

 

if __name__ == '__main__':
    test_nr("sym")

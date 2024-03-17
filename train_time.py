import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import logging
import numpy as np
from dataset import CIFAR10Custom # 确保这里正确地从您的dataset.py文件导入CIFAR10Custom类
import utils.hash_model as image_hash_model # 确保您的hash_model模块包含了HASH_Net定义
from utils.cac import * # 确保test_accuracy函数可以从cac模块导入
import time
from utils.denoise import label_refurb
from utils.lr_scheduler import WarmupLR
import os
from utils.PreResnet import *
os.environ['cpu_LAUNCH_BLOCKING'] = '1'

# 参数定义
batch_size = 128
epochs = 100
lr = 0.01
weight_decay = 10 ** -5
lambda1 = 0.01
hash_bits = 64
model_name = "resnet34"
device = torch.device("cuda")



def load_dataset(noise_type, noise_rate=0.0, batch_size= 256, num_workers = 30):


    # 使用CIFAR10Custom类加载数据集
    train_dataset = CIFAR10Custom(root='./data', train=True, noise_type=noise_type, noise_rate=noise_rate)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers)

    test_dataset = CIFAR10Custom(root='./data', train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers = num_workers)

    return train_loader, test_loader

# 模型训练函数
def train_model(model, trainloader, testloader,label_hash_codes, epochs=epochs, hash_bits = 64):
    device = torch.device("cuda")
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.8, last_epoch=-1)
    #scheduler = WarmupLR(optimizer,warmup_epochs=20,initial_lr=0.002)
    best_accuracy = 0.0
    num_classes = label_hash_codes.size(0) 
    label_hash_codes=label_hash_codes.to(device)

    for epoch in range(epochs):
        epoch_start_time = time.time()  # 记录整个epoch的开始时间
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
        

        for iter, (inputs, labels,is_noise) in enumerate(trainloader):
            iter_start_time = time.time()  # 记录这一次迭代的开始时间
            
            inputs= inputs.to(device)
            labels = labels.to(device)
            is_noise = is_noise.to(device)
            data_to_gpu_time = time.time() - iter_start_time

            forward_start_time = time.time()
            outputs = model(inputs).to(device)
            forward_time = time.time() - forward_start_time
            
            criterion = nn.BCELoss().to(device)
            t2 = time.time()

            label_refurb_start_time = time.time()
            outputs,labels,is_noise = label_refurb(epoch, labels,outputs,is_noise,label_hash_codes,hash_bits,device, iter, True)
            label_refurb_time = time.time() - label_refurb_start_time
            
            prepare_loss_calc_start_time = time.time()

            outputs = torch.clamp(outputs,-1.0 + 1e-4,1.0 - 1e-4)
 
            
            predicted_label = get_predicted_cate(outputs,label_hash_codes,device)
            precodes = label_hash_codes[predicted_label.cpu()].to(device) 
            logits =(outputs.unsqueeze(1) * label_hash_codes.unsqueeze(0)).sum(dim=2)
            logits =logits/(0.0625*hash_bits)
            prior = torch.ones(10)/10
            prior = prior.to(device)        
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior*torch.log(prior/pred_mean))

            cat_codes = label_hash_codes[labels.cpu()].to(device) 
            center_loss = criterion(0.5*(outputs+1), 0.5*(cat_codes+1))
            Q_loss = torch.mean((torch.abs(outputs)-1.0)**2)
            loss = center_loss + lambda1*Q_loss+0.1*penalty
            prepare_loss_calc_time = time.time() - prepare_loss_calc_start_time

            optimizer.zero_grad()

            backward_start_time = time.time()
            loss.backward()
            backward_time = time.time() - backward_start_time 

            step_start_time = time.time()
            optimizer.step()
            step_time = time.time() - step_start_time
            
            iter_time = time.time()-iter_start_time
            if(iter%100==0):
                #print("center_loss:",center_loss)
                #print("Q_loss:",lambda1*Q_loss)
                #print("pred_mean:",pred_mean)
                #print("penalty",penalty)
                print(f"迭代{iter}: Data Load to GPU Time: {data_to_gpu_time:.4f}, forward_time: {forward_time:.4f}, label_refurb_time: {label_refurb_time:.4f}, prepare_loss_calc_time: {prepare_loss_calc_time:.4f}, backward_time {backward_time:.4f}, step_time: {step_time:.4f},iter_time:{iter_time:.4f}")
            
      

def test_nr(noisetype = None, noise_rate = None, epoch = None):
    if epoch!=None:
        epochs = epoch
    logging.basicConfig(filename=f'./logs/{model_name}_{noisetype}_test_nr.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info(f'Training Configuration: batch_size={batch_size}, epochs={epochs}, lr={lr}, weight_decay={weight_decay}, lambda1={lambda1}, hash_bits={hash_bits}, model_name={model_name}, device={device}')
     # 加载标签哈希码
    with open(f'./labels/{hash_bits}_cifar10_10_class.pkl', 'rb') as f:
        label_hash_codes = torch.load(f)
    label_hash_codes.to(device)
    
    #noise_types = ['aggre_label','worse_label', 'random_label1', 'random_label2', 'random_label3','clean_label']
    noise_rates = [0.2,0.4,0.6,0.8,0.0]
    noise_rates = [0.0,0.4,0.6,0.8,0.2]
    noise_rates =[0.4]
    if noise_rate is not None:
        noise_rates = [noise_rate]
    for noise_rate in noise_rates:
        trainloader, testloader = load_dataset(noise_type=noisetype, batch_size=batch_size, noise_rate=noise_rate)
        #model = image_hash_model.HASH_Net(model_name, hash_bits).to(device)
        #model = ResNet34Hash(hash_bits).to(device)
        model = CSQModel(hash_bits).to(device)
        logging.info(f'Start Training with: {noisetype}-{noise_rate}')
        train_model(model, trainloader, testloader, label_hash_codes,epochs=epoch)
        logging.info(f'Finished Training with: {noisetype}-{noise_rate}')
    


def test_cifarn(noise_type = None,epoch = None):
    if epoch!=None:
        epochs = epoch
    logging.basicConfig(filename=f'./logs/{model_name}_test_cifarn.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info(f'Training Configuration: batch_size={batch_size}, epochs={epochs}, lr={lr}, weight_decay={weight_decay}, lambda1={lambda1}, hash_bits={hash_bits}, model_name={model_name}, device={device}')
    with open(f'./labels/{hash_bits}_cifar10_10_class.pkl', 'rb') as f:
        label_hash_codes = torch.load(f)
    label_hash_codes.to(device)
    
    noise_types = ['aggre_label','random_label1','worse_label','random_label2', 'random_label3','clean_label']
    noise_types = ['worse_label','aggre_label','random_label1','random_label2', 'random_label3','clean_label']
    #noise_rates = [0.2,0.4,0.6,0.8,0.0]
    if noise_type is not None:
        noise_types = [noise_type]

    for noise_type in noise_types:
        # 加载模型
        trainloader, testloader = load_dataset(noise_type=noise_type, batch_size=batch_size)
        noise_rate = trainloader.dataset.noise_rate
        model = CSQModel(hash_bits).to(device)
        logging.info(f'Start Training with: {noise_type}-{noise_rate}')
        train_model(model, trainloader, testloader, label_hash_codes,epochs=epochs)
        logging.info(f'Finished Training with: {noise_type}-{noise_rate}')

def test_hashbits():
    epochs=100
    logging.basicConfig(filename=f'./logs/{model_name}_test_hashbits.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info(f'Training Configuration: batch_size={batch_size}, epochs={epochs}, lr={lr}, weight_decay={weight_decay}, lambda1={lambda1}, hash_bits={hash_bits}, model_name={model_name}, device={device}')
    
    hashbits = [128,32,16,64]
    

    for hashbit in hashbits:
        with open(f'./labels/{hashbit}_cifar10_10_class.pkl', 'rb') as f:
            label_hash_codes = torch.load(f)
        label_hash_codes.to(device)
    
        # 加载模型
        
        trainloader, testloader = load_dataset(noise_type='sym', noise_rate=0.4,batch_size=batch_size)
        model = image_hash_model.HASH_Net(model_name, hashbit,pretrained=False).to(device)
        logging.info(f'Start Training with hash_bits: {hashbit}')
        train_model(model, trainloader, testloader, label_hash_codes,epochs=epochs,hash_bits=hashbit)
        logging.info(f'Finished Training with hash_bits: {hashbit}')

 

if __name__ == '__main__':
    #test_cifarn("random_label1",epoch = 50)
    #test_nr("sym",0.4,30)
    #test_cifarn(epoch=100)
    test_nr("sym",0.4,100)

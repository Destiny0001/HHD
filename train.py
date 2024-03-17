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
from utils.denoise import *
from utils.lr_scheduler import WarmupLR
import os
from utils.PreResnet import *
from torch.profiler import profile, record_function, ProfilerActivity


os.environ['cpu_LAUNCH_BLOCKING'] = '1'

# 参数定义
batch_size = 128
epochs = 100
lr = 0.01
weight_decay = 10 ** -5
lambda1 = 0.01
hash_bits = 16
model_name = "resnet34"
device = torch.device("cuda")

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    print("Checkpoint saved to {}".format(filename))

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    # 注意: 输入模型 & 优化器需要事先初始化
    start_epoch = 0
    best_accuracy = 0.0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint.get('best_accuracy', best_accuracy)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
    return model, optimizer, start_epoch, best_accuracy



def adjust_optimizer(optimizer, epoch, init_momentum=0.9, target_epoch=4,end_epoch=8):

    for param_group in optimizer.param_groups:
        # Set momentum to 0 at the target epoch, otherwise use init_momentum
        if epoch >= target_epoch and epoch <=end_epoch:
            param_group['momentum'] = 0
        elif epoch >end_epoch:
            param_group['momentum'] = 0.9
        else:
            param_group['momentum'] = init_momentum

def load_dataset(noise_type, noise_rate=0.0, batch_size= 256, num_workers = 30):

    train_dataset = CIFAR10Custom(root='./data', train=True, noise_type=noise_type, noise_rate=noise_rate)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers,pin_memory=True, prefetch_factor=2)

    test_dataset = CIFAR10Custom(root='./data', train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers = num_workers,pin_memory=True, prefetch_factor=2)

    return train_loader, test_loader




def train_model(model, trainloader, testloader,label_hash_codes, epochs=epochs, hash_bits = 64):
    device = torch.device("cuda")
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.8, last_epoch=-1)

    best_accuracy = 0.0
    num_classes = label_hash_codes.size(0) 
    stats_matrix = torch.zeros(epochs, num_classes, hash_bits, device=device)  # 初始化统计矩阵
    distance_matrix = torch.zeros((epochs, hash_bits, 2), dtype=torch.long)
    metric_matrix = torch.zeros((epochs, hash_bits+64, 8), dtype=torch.long)
    class_distribution_matrix = torch.zeros((epochs, num_classes, 2), dtype=torch.long).to('cuda')
    sample_matrix = torch.zeros((epochs, 4), dtype=torch.long)


    label_hash_codes=label_hash_codes.to(device)

    start_epoch =0
    model, optimizer, start_epoch, best_accuracy = load_checkpoint(model, optimizer, filename=f'./model/{trainloader.dataset.noise_type}_epoch0_checkpoint.pth.tar')
    for epoch in range(start_epoch, epochs):
        adjust_optimizer(optimizer, epoch, init_momentum=0.9, target_epoch=4,end_epoch=100)
        stats_accum = {
            'clean_for_train': 0,
            'noise_for_train': 0,
            'delete_clean': 0,
            'delete_noise': 0,
        }
        model.train()
        scheduler.step()


        accuracy = test_accuracy(model, testloader, label_hash_codes, device)
        logging.info(f'Epoch {epoch}: Test Accuracy: {accuracy}%')
        print(f'Epoch {epoch}/{epochs}, Test Accuracy: {accuracy}%')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'./model/{trainloader.dataset.noise_type}_{model_name}.pth')
            print(f"Model saved with accuracy: {best_accuracy:.2f}%")
            logging.info(f"Model saved with accuracy: {best_accuracy:.2f}%")
            
        for iter, (inputs, labels,is_noise) in enumerate(trainloader):
            
            #labels = torch.from_numpy(np.array(labels))
            inputs= inputs.to(device)
            is_noise = is_noise.to(device)
            labels = labels.to(device)
            outputs = model(inputs).to(device)
            criterion = nn.BCELoss().to(device)
            
            outputs,labels,is_noise,stats  = sample_selection(epoch, labels,outputs,is_noise,label_hash_codes,hash_bits,device, iter, True)
            
            for key in stats_accum.keys():
                stats_accum[key] += stats[key]
            outputs = torch.clamp(outputs,-1.0 + 1e-4,1.0 - 1e-4)

            update_stats_matrix(epoch, outputs, labels, label_hash_codes, stats_matrix)
            update_distance_matrix(outputs, labels, is_noise, label_hash_codes, distance_matrix, epoch)
            update_metric_matrix(outputs, labels, is_noise, label_hash_codes, metric_matrix, epoch)
            update_class_distribution_matrix(outputs, labels, is_noise, label_hash_codes, class_distribution_matrix, epoch)

                    
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
            """ 
            if(iter%100==0):
                print("center_loss:",center_loss)
                print("Q_loss:",lambda1*Q_loss)
                print("pred_mean:",pred_mean)
                print("penalty",penalty)"""
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch} statistics:")
        for key, value in stats_accum.items():
                print(f"  {key}: {value}")
        if epoch % 10 == 0:
            # 仅当epoch是10的倍数时保存检查点
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'best_accuracy': best_accuracy,
                'optimizer': optimizer.state_dict(),
            }, filename=f'./model/{trainloader.dataset.noise_type}_epoch{epoch}_checkpoint.pth.tar')
            print(f"Checkpoint saved at epoch {epoch}")
        sample_matrix[epoch] = torch.tensor([stats_accum['clean_for_train'], 
                                            stats_accum['noise_for_train'], 
                                            stats_accum['delete_clean'], 
                                            stats_accum['delete_noise']])
    


    
    stats_matrix_path = './sta/stats_matrix_true.pt'  # 选择你想要保存的路径和文件名
    distance_matrix_path = f'./sta/distance_matrix_{trainloader.dataset.noise_type}_{trainloader.dataset.noise_rate}.pt'
    metric_matrix_path = f'./sta/metric_matrix_{trainloader.dataset.noise_type}_{trainloader.dataset.noise_rate}.pt'
    class_distribution_matrix_path = f'./sta/class_distribution_matrix_{trainloader.dataset.noise_type}_{trainloader.dataset.noise_rate}.pt'
    sample_matrix_path = f'./sta/sample_matrix_{trainloader.dataset.noise_type}_{trainloader.dataset.noise_rate}.pt'
    
    torch.save(stats_matrix, stats_matrix_path)
    torch.save(distance_matrix, distance_matrix_path)
    torch.save(metric_matrix, metric_matrix_path)
    torch.save(class_distribution_matrix,class_distribution_matrix_path)
    torch.save(sample_matrix,sample_matrix_path)


      

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
    #noise_rates = [0.0,0.4,0.6,0.8,0.2]
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
    #test_nr("sym",0.6,100)
    #test_cifarn(epoch=100)
    test_nr("sym",epoch=100)

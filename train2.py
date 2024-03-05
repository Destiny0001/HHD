import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import logging
import numpy as np
import utils.hash_model as image_hash_model  # 确保这个路径正确
from utils.cac import test_accuracy  # 确保这个路径正确
from utils.denoise import label_refurb  # 确保这个路径正确
from utils.lr_scheduler import WarmupLR  # 确保这个路径正确
from dataloader import cifarn_dataloader  # 确保这个路径正确

# 参数定义
batch_size = 128
epochs = 100
lr = 0.001
weight_decay = 10 ** -5
lambda1 = 0.01
hash_bits = 32
model_name = "resnet34"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(model, train_loader, test_loader, label_hash_codes, epochs=100, hash_bits=64):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = WarmupLR(optimizer, warmup_epochs=20, initial_lr=0.002)
    criterion = nn.BCELoss().to(device)
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        scheduler.step(epoch)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs, labels = label_refurb(labels, outputs, label_hash_codes, hash_bits, device, epoch < 6)
            loss = criterion(outputs, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accuracy = test_accuracy(model, test_loader, device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'./model/{model_name}_{hash_bits}_best.pth')
            logger.info(f'Epoch {epoch+1}: Higher accuracy {best_accuracy:.2f}% achieved, model saved.')

        logger.info(f'Epoch {epoch+1}: Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.2f}%')

def test_nr(noisetype=None):
    device = torch.device("cuda")
    logging.basicConfig(filename=f'./logs/{model_name}_{noisetype}_test_nr.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    logger.info(f'Training Configuration: batch_size={batch_size}, epochs={epochs}, lr={lr}, weight_decay={weight_decay}, lambda1={lambda1}, hash_bits={hash_bits}, model_name={model_name}, device={device}')
    
    with open(f'./labels/{hash_bits}_cifar10_10_class.pkl', 'rb') as f:
        label_hash_codes = torch.load(f)
    label_hash_codes = label_hash_codes.to(device)
    
    noise_rates = [0.8, 0.6, 0.4, 0.2, 0.0]
    
    for noise_rate in noise_rates:
        cifarn_loader = cifarn_dataloader(dataset='cifar10', noise_type=noisetype, batch_size=batch_size, num_workers=4, root_dir='./data', log=logger, noise_file=f'./noise/{noisetype}_{noise_rate}.json', r=noise_rate, noise_mode=noisetype)
        train_loader = cifarn_loader.run(mode='train')
        test_loader = cifarn_loader.run(mode='test')
        
        model = image_hash_model.HASH_Net(model_name, hash_bits).to(device)
        logger.info(f'Start Training with: {noisetype}-{noise_rate}')
        train_model(model, train_loader, test_loader, label_hash_codes, epochs=epochs)
        logger.info(f'Finished Training with: {noisetype}-{noise_rate}')

def test_cifarn():
    device = torch.device("cuda")
    logging.basicConfig(filename=f'./logs/{model_name}_test_cifarn.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    logger.info(f'Training Configuration: batch_size={batch_size}, epochs={epochs}, lr={lr}, weight_decay={weight_decay}, lambda1={lambda1}, hash_bits={hash_bits}, model_name={model_name}, device={device}')
    
    with open(f'./labels/{hash_bits}_cifar10_10_class.pkl', 'rb') as f:
        label_hash_codes = torch.load(f)
    label_hash_codes = label_hash_codes.to(device)
    
    noise_types = ['aggre_label', 'random_label1', 'worse_label', 'random_label2', 'random_label3', 'clean_label']
    
    for noise_type in noise_types:
        cifarn_loader = cifarn_dataloader(dataset='cifar10', noise_type=noise_type, batch_size=batch_size, num_workers=4, root_dir='./data', log=logger, noise_file=f'./noise/{noise_type}.json', r=0.2, noise_mode=noise_type)
        train_loader = cifarn_loader.run(mode='train')
        test_loader = cifarn_loader.run(mode='test')
        
        model = image_hash_model.HASH_Net(model_name, hash_bits).to(device)
        logger.info(f'Start Training with: {noise_type}')
        train_model(model, train_loader, test_loader, label_hash_codes, epochs=epochs)
        logger.info(f'Finished Training with: {noise_type}')

def test_hashbits():
    device = torch.device("cuda")
    logging.basicConfig(filename=f'./logs/{model_name}_test_hashbits.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    logger.info(f'Training Configuration: batch_size={batch_size}, epochs={epochs}, lr={lr}, weight_decay={weight_decay}, lambda1={lambda1}, hash_bits={hash_bits}, model_name={model_name}, device={device}')
    
    hashbits = [16, 32, 64, 128]
    with open(f'./labels/{hash_bits}_cifar10_10_class.pkl', 'rb') as f:
        label_hash_codes = torch.load(f)
    label_hash_codes = label_hash_codes.to(device)
    for hashbit in hashbits:
        cifarn_loader = cifarn_dataloader(dataset='cifar10', noise_type='sym', batch_size=batch_size, num_workers=4, root_dir='./data', log=logger, noise_file=f'./noise/sym_{hashbit}.json', r=0.4, noise_mode='sym', hash_bits=hashbit)
        train_loader = cifarn_loader.run(mode='train')
        test_loader = cifarn_loader.run(mode='test')
        
        model = image_hash_model.HASH_Net(model_name, hashbit).to(device)
        logger.info(f'Start Training with hash_bits: {hashbit}')
        train_model(model, train_loader, test_loader, label_hash_codes, epochs=epochs)
        logger.info(f'Finished Training with hash_bits: {hashbit}')

if __name__ == '__main__':
    test_nr("sym")
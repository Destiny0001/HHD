import torch
import torchvision
import torchvision.transforms as transforms
import utils.hash_model as image_hash_model
import numpy as np
import time
from utils.cac import test_accuracy
from utils.condition import add_label_noise
import torch.optim as optim
import torch.nn as nn
from utils.condition import cifar10ntrainloader
# 参数定义
top_k = 1000
batch_size = 64
epochs = 100
lr = 0.01
weight_decay = 10 ** -5
lambda1 = 0.01
hash_bits = 64
model_name = "resnet34"
device = torch.device("cuda")
import logging

# 设置日志记录

logging.basicConfig(filename='{model_name}_cifar10n.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')
logging.info(f'Training Configuration: batch_size={batch_size}, epochs={epochs}, lr={lr}, weight_decay={weight_decay}, lambda1={lambda1}, hash_bits={hash_bits}, model_name={model_name}, device={device}')
# 数据加载
def load_dataset(noise_type, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    trainloader = cifar10ntrainloader(noise_type,transform,batch_size)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

# 模型训练
def train_model(model, trainloader, testloader, label_hash_codes, noise_type):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.3, last_epoch=-1)

    total_loss = []
    accuracy_list = []
    best_accuracy = 0.0

    for epoch in range(epochs):
        scheduler.step()
        if epoch % 1 == 0:
            accuracy = test_accuracy(model, testloader, label_hash_codes, device)
            print(f'Epoch {epoch}, Test Accuracy: {accuracy}%')
            logging.info(f'Epoch {epoch}, Test Accuracy: {accuracy}%')

            accuracy_list.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), f'./model/nt_{noise_type}_{model_name}.pth')
                print(f"Model saved with accuracy: {best_accuracy:.2f}%")
                logging.info(f"Model saved with accuracy: {best_accuracy:.2f}%")

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
            #total_loss.append(loss.item())
            total_loss.append(loss.data.to(device).numpy)

            if iter % 500 == 0:
                print(f'Epoch: {epoch}, Iter: {iter}, Loss: {loss}')

    return total_loss, accuracy_list

def main():
    

    # 设定不同的噪声率
    noise_types = ['clean_label','worse_label' ,'aggre_label','random_label1','random_label2','random_label3']

 # 加载标签哈希码
    with open('./labels/64_cifar10_10_class.pkl', 'rb') as f:
        label_hash_codes = torch.load(f)
    label_hash_codes.to(device)

    for noise_type in noise_types:
        trainloader, testloader = load_dataset(noise_type=noise_type)
        # 初始化模型
        model = image_hash_model.HASH_Net(model_name, hash_bits).to(device)

        # 进行训练
        total_loss, accuracy_list = train_model(model, trainloader, testloader, label_hash_codes, noise_type)
        logging.info(f'Finished Training with noise_rate: {noise_type}')
        logging.info(f'Epoch accuracies with noise_rate {noise_type}: {accuracy_list}')
        
        # 可以在这里添加代码来保存每个噪声率的训练结果，例如损失和准确率

if __name__ == '__main__':
   main()

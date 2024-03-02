import numpy as np
import torch


dataset = 'cifar10'
noise_path = None

# please change it to your own datapath for CIFAR-N



import torch
import torchvision
import torchvision.transforms as transforms

# 加载CIFAR-10N数据集
def cifar10ntrainloader(noise_type, transforms, batchsize,  dataset='cifar10' ):
    if dataset == 'cifar10':
        noise_path = './data/cifarn/CIFAR-10_human.pt'
    elif dataset == 'cifar100':
        noise_path = './data/cifarn/CIFAR-100_human.pt'
    else:
        pass

    # 加载CIFAR-10的训练集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)

    # 加载噪声标签
    noise_file = torch.load(noise_path)
    noisy_labels = noise_file[noise_type]

    # 替换原始训练集标签为噪声标签
    trainset.targets = noisy_labels

    # 创建DataLoader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=False, num_workers=30)

    return trainloader





def add_label_noise(labels, noise_rate=0.1, num_classes=10):
    """
    添加标签噪声。
    labels: 标签数组
    noise_rate: 噪声率，表示要翻转的标签比例
    num_classes: 类别总数
    """
    # 确定需要翻转的标签数量
    num_noisy_labels = int(noise_rate * len(labels))
    noisy_indices = np.random.choice(len(labels), num_noisy_labels, replace=False)

    for idx in noisy_indices:
        # 生成一个不同于原始标签的随机新标签
        original_label = labels[idx]
        new_label = np.random.choice([l for l in range(num_classes) if l != original_label])
        labels[idx] = new_label
    
    return labels

'''

noise_file = torch.load(noise_path)
clean_label = noise_file['clean_label']
worst_label = noise_file['worse_label']
aggre_label = noise_file['aggre_label']
random_label1 = noise_file['random_label1']
random_label2 = noise_file['random_label2']
random_label3 = noise_file['random_label3']
'''
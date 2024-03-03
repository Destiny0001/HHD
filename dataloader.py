import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def dataloaders(root='./data', transform_train=None, transform_test=None, batch_size_train=64, batch_size_test=64, noise_type='sym', noise_rate=0.0, cifar10n_path='./data/cifarn/CIFAR-10_human.pt'):
    # 加载训练集
    trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    if noise_type in ['sym', 'asym'] and noise_rate > 0:
        if noise_type == 'sym':
            labels = np.array(trainset.targets)
            n_noisy = int(noise_rate * len(labels))
            noisy_indices = np.random.choice(len(labels), n_noisy, replace=False)
            for idx in noisy_indices:
                labels[idx] = np.random.choice([n for n in range(10) if n != labels[idx]])
            trainset.targets = labels.tolist()
        elif noise_type == 'asym':
            transition = {0:0, 2:0, 4:7, 7:7, 1:1, 9:1, 3:5, 5:3, 6:6, 8:8}
            labels = np.array(trainset.targets)
            for i, label in enumerate(labels):
                if np.random.rand() < noise_rate: 
                    labels[i] = transition[label]
            trainset.targets = labels.tolist()
    elif cifar10n_path:
        cifarn_labels = torch.load(cifar10n_path)[noise_type]
        trainset.targets = cifarn_labels
    trainloader = DataLoader(trainset, batch_size=batch_size_train, shuffle=False, num_workers=40)

    # 加载测试集
    testset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=40)

    return trainloader, testloader

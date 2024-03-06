import os
import json
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from utils.autoaugment import *
class CIFAR10Custom(Dataset):
    def __init__(self, root, train=True, noise_type='sym', noise_rate=0.0, cifar10n_path='./data/cifarn/CIFAR-10_human.pt'):
        self.noise_type = noise_type
        self.noise_rate= noise_rate
        if train:
            self.transform = transform_train
            self.transform = transform 
        else:
            self.transform = transform_test
            self.transform = transform
        self.dataset = datasets.CIFAR10(root=root, train=train, download=False, transform=self.transform)
        if train:
            if noise_type == 'sym':
                self.dataset.targets = self.add_label_noise(np.array(self.dataset.targets), noise_rate=noise_rate, num_classes=10)
            elif noise_type == 'asym':
                self.dataset.targets = self.add_asymmetric_noise(np.array(self.dataset.targets))
            elif cifar10n_path:
                self.dataset.targets = self.load_cifarn_labels(cifar10n_path)

    def add_label_noise(self, labels, noise_rate, num_classes=10):
        noisy_labels = np.array(labels, copy=True)
        n_noisy = int(noise_rate * len(labels))
        noisy_indices = np.random.choice(len(labels), n_noisy, replace=False)
        for idx in noisy_indices:
            noisy_labels[idx] = np.random.choice([n for n in range(num_classes) if n != labels[idx]])
        return noisy_labels

    def add_asymmetric_noise(self, labels):
        transition = {0:0, 2:0, 4:7, 7:7, 1:1, 9:1, 3:5, 5:3, 6:6, 8:8}
        noisy_labels = np.array(labels, copy=True)
        for i, label in enumerate(labels):
            if np.random.rand() < self.noise_rate: 
                noisy_labels[i] = transition[label]
        return noisy_labels


    def load_cifarn_labels(self, cifar10n_path):
        cifarn_labels = torch.load(cifar10n_path)[self.noise_type]
        return cifarn_labels

    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
            # 获取数据和标签
        image, label = self.dataset[idx]

        # 返回转换后的图像和标签
        return image, label

transform_train = transforms.Compose([
                                        #transforms.RandomHorizontalFlip(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, padding=4),
                                        #CIFAR10Policy(),
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

transform_test = transforms.Compose([
                
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ])

transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(56),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
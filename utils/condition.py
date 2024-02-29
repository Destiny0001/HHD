import numpy as np



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

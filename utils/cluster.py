from sklearn.cluster import KMeans
import torch
import numpy as np

def cluster_and_evaluate_noise(outputs, labels, is_noise, label_hash_codes, epoch, n_clusters=10, noise_distance_threshold=32):
    """
    结合语义编码和标签编码进行聚类，并基于与聚类中心的距离识别噪声点。

    参数:
    - outputs: 模型的输出（语义编码）。
    - labels: 样本的实际标签。
    - is_noise: 布尔数组，表示每个样本是否是噪声。
    - label_hash_codes: 每个类别的标签编码哈希码。
    - epoch: 当前的epoch编号。
    - n_clusters: 聚类的数量，应该与类别数相匹配。
    - noise_distance_threshold: 识别噪声的距离阈值。
    """
    # 转换标签到标签编码
    label_codes = label_hash_codes[labels].cpu().numpy()
    
    # 结合语义编码和标签编码
    combined_features = torch.cat((outputs, label_codes), dim=1).cpu().numpy()

    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(combined_features)
    cluster_centers = kmeans.cluster_centers_
    labels_pred = kmeans.labels_

    # 计算每个样本到其聚类中心的距离
    distances = np.linalg.norm(combined_features - cluster_centers[labels_pred], axis=1)

    # 识别噪声样本
    predicted_noise = distances > noise_distance_threshold
    true_noise = is_noise.cpu().numpy().astype(bool)

    # 计算正确和错误识别的噪声点数量
    correct_noise = np.sum(predicted_noise & true_noise)
    incorrect_noise = np.sum(predicted_noise & ~true_noise)
    
    print(f'Epoch {epoch}: Correctly identified noise points: {correct_noise}')
    print(f'Epoch {epoch}: Incorrectly identified noise points: {incorrect_noise}')

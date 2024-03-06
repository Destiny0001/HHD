import torch
import numpy as np
from fvcore.nn import FlopCountAnalysis
from torchsummary import summary

def test_accuracy(model, test_loader, label_hash_codes, device):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算
        for data, labels in test_loader:
            data = data.to(device)
            outputs = torch.sign(model(data)).to(device) 
            
            # 通过计算输出和每个类别哈希码之间的相似度来简化汉明距离的计算
            # 计算相似度
            similarities = torch.mm(outputs, label_hash_codes.t().to(device))
            
            # 相似度最高的类别即为预测类别
            _, predicted = similarities.max(dim=1)
            
            total += labels.size(0)
            correct += (predicted.to(device) == labels.to(device)).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy



def get_predicted_cate(outputs,label_hash_codes, device = torch.device("cuda")):
    outputs = torch.sign(outputs).to(device)  # 模型输出的哈希值 后续可以继续修改
    
    # 通过计算输出和每个类别哈希码之间的相似度来简化汉明距离的计算
    # 计算相似度
    similarities = torch.mm(outputs, label_hash_codes.t().to(device))
    
    # 相似度最高的类别即为预测类别
    _, predicted = similarities.max(dim=1)
    #predicted =  torch.from_numpy(np.array(predicted))
    #predicted_codes = label_hash_codes[predicted.cpu()].to(device) 
    return predicted


    


def calculate_macs(model, input_size=(1, 3, 224, 224)):
    inputs = torch.randn(input_size)
    flops = FlopCountAnalysis(model, inputs)
    total_flops = flops.total()
    print(f"Total FLOPs: {total_flops}")

def model_size(model, input_size=(1, 3, 224, 224), bits=32):
    summary(model, input_size=input_size)
    params = sum(p.numel() for p in model.parameters())
    model_size = params * bits / 8 / 1024 / 1024  # 转换为MB
    print(f"模型总参数数: {params}, 大小大约为 {model_size} MB")

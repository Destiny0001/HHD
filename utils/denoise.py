import torch
from utils.cac import get_predicted_cate
import numpy as np
from torchnet.meter import AUCMeter




def compute_probability(dis, scale=1.0):
    """
    根据给定的距离(dis)计算样本被判定为正例的概率。
    使用 Sigmoid 函数作为映射，可以通过调整 scale 参数来控制函数的形状。
    """
    prob = torch.sigmoid(-scale * dis)
    return prob



def sample_selection(epoch, labels, outputs,is_noise,  label_hash_codes, hashbits, device,iter,turn = True):
    stats={}
    if epoch<6:
        turn = False
    if turn == False:
        return outputs,labels,is_noise,stats
    K =1/2*hashbits
    predicted_label = get_predicted_cate(outputs,label_hash_codes,device)
    precodes = label_hash_codes[predicted_label.cpu()].to(device)
    simcodes = torch.sign(outputs).to(device)
    labelcodes = label_hash_codes[labels].to(device)
    predis = torch.sum(simcodes!=precodes,dim=1).to(device)
    labeldis = torch.sum(simcodes!=labelcodes,dim=1).to(device)
        
    noise_samples_mask = (labeldis+predis>=K)

    noise_indices = torch.where(noise_samples_mask)[0]  # 获取噪声样本的索引
    random_perm = torch.randperm(noise_indices.size(0), device=device)  # 生成随机排列
    if epoch <50:
        select_num = int(noise_indices.size(0) * 0.8) 
    else:
        select_num = int(noise_indices.size(0) * 0.9) 

    selected_indices = noise_indices[random_perm[:select_num]] 
    new_noise_samples_mask = torch.zeros_like(noise_samples_mask)
    new_noise_samples_mask[selected_indices] = True  # 更新噪声样本掩码
    noise_samples_mask = new_noise_samples_mask

    
    #noise_samples_mask = is_noise&noise_samples_mask
    clean_samples_mask = ~noise_samples_mask.to(device)
    true_noise_samples_mask = (is_noise==1).to(device)

    clean_for_train = torch.sum(~noise_samples_mask & ~true_noise_samples_mask).item()
    noise_for_train = torch.sum(~noise_samples_mask & true_noise_samples_mask).item()
    delete_clean = torch.sum(noise_samples_mask & ~true_noise_samples_mask).item()
    delete_noise = torch.sum(noise_samples_mask & true_noise_samples_mask).item()

    stats = {
        'clean_for_train': clean_for_train,
        'noise_for_train': noise_for_train,
        'delete_clean': delete_clean,
        'delete_noise': delete_noise,
    }

    dis = labeldis + predis  # 计算总距离
    probability = compute_probability(dis)  # 计算概率

    
   
    if iter%200==0:
        auc_meter = AUCMeter()
        auc_meter.reset()
        auc_meter.add(probability,~true_noise_samples_mask)        
        auc,_,_ = auc_meter.value() 
        print("auc:",auc)


    valid_outputs = outputs[clean_samples_mask]
    valid_labels = labels[clean_samples_mask]
    valid_isnoise = is_noise[clean_samples_mask]

    return valid_outputs, valid_labels,valid_isnoise,stats









def label_refurb(epoch, labels, outputs, is_noise, label_hash_codes, hashbits, device, iter, turn=True):
    
    K = 1/2 * hashbits
    predicted_label = get_predicted_cate(outputs, label_hash_codes, device)
    precodes = label_hash_codes[predicted_label.cpu()].to(device)
    simcodes = torch.sign(outputs).to(device)
    labelcodes = label_hash_codes[labels].to(device)
    predis = torch.sum(simcodes != precodes, dim=1).to(device)
    labeldis = torch.sum(simcodes != labelcodes, dim=1).to(device)
    
    if epoch < 6:
        turn = False
        
    noise_samples_mask = (labeldis + predis >= K)
    noise_indices = torch.where(noise_samples_mask)[0]  # 获取噪声样本的索引
    random_perm = torch.randperm(noise_indices.size(0), device=device)  # 生成随机排列
    
    if epoch < 50:
        select_num = int(noise_indices.size(0) * 0.2)
    else:
        select_num = int(noise_indices.size(0) * 0.9)
    
    selected_indices = noise_indices[random_perm[:select_num]]
    refurb_indices = noise_indices[random_perm[select_num:]]  # 选择剩余的样本进行标签更换
    
    labels[refurb_indices] = predicted_label[refurb_indices]  # 更换标签为预测的precodes
    is_noise[refurb_indices] = 0  # 将这部分样本视为非噪声样本

    # 更新噪声样本掩码
    noise_samples_mask = torch.zeros_like(noise_samples_mask)
    noise_samples_mask[selected_indices] = True
    
    clean_samples_mask = ~noise_samples_mask.to(device)
    true_noise_samples_mask = (is_noise == 1).to(device)
    
    clean_for_train = torch.sum(~noise_samples_mask & ~true_noise_samples_mask).item()
    noise_for_train = torch.sum(~noise_samples_mask & true_noise_samples_mask).item()
    delete_clean = torch.sum(noise_samples_mask & ~true_noise_samples_mask).item()
    delete_noise = torch.sum(noise_samples_mask & true_noise_samples_mask).item()
    
    stats = {
        'clean_for_train': clean_for_train,
        'noise_for_train': noise_for_train,
        'delete_clean': delete_clean,
        'delete_noise': delete_noise,
    }
    
    valid_outputs = outputs[clean_samples_mask]
    valid_labels = labels[clean_samples_mask]
    valid_isnoise = is_noise[clean_samples_mask]
    
    if turn == False:
        return outputs, labels, is_noise, stats
    return valid_outputs, valid_labels, valid_isnoise, stats
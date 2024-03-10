import torch
from utils.cac import get_predicted_cate
import numpy as np

def label_refurb(epoch, labels, outputs,is_noise,  label_hash_codes, hashbits, device,iter,turn = True):
    
    K =1/2*hashbits
    predicted_label = get_predicted_cate(outputs,label_hash_codes,device)
    precodes = label_hash_codes[predicted_label.cpu()].to(device) 
    simcodes = torch.sign(outputs).to(device)
    labelcodes = label_hash_codes[labels].to(device)
    device = torch.device("cuda")
    predis = torch.sum(simcodes!=precodes,dim=1).cuda()
    labeldis = torch.sum(simcodes!=labelcodes,dim=1).cuda()
    is_noise = is_noise.to(device)

    if epoch<6:
        noise_samples_mask = (labeldis!=labeldis)
    elif epoch<30:
        noise_samples_mask = (labeldis-predis>0.5*K)&(labeldis>=K)
        #noise_samples_mask = (labeldis-predis>=0.5*K)
        
    elif epoch<100:
        noise_samples_mask = (labeldis-predis>0.6*K)& (labeldis>K) 

    

     # 新增代码：随机选择一半噪声样本
    noise_indices = torch.where(noise_samples_mask)[0]  # 获取噪声样本的索引
    random_perm = torch.randperm(noise_indices.size(0), device=device)  # 生成随机排列
    select_half = noise_indices[random_perm[:noise_indices.size(0) // 2]]  # 选择一半
    new_noise_samples_mask = torch.zeros_like(noise_samples_mask)
    new_noise_samples_mask[select_half] = True  # 更新噪声样本掩码
    noise_samples_mask = new_noise_samples_mask


    
    #noise_samples_mask = is_noise&noise_samples_mask
    clean_samples_mask = ~noise_samples_mask.cuda()
    true_noise_samples_mask = (is_noise==1).cuda()
    true_noise_count = torch.sum(true_noise_samples_mask & noise_samples_mask.cuda()).item()
    false_noise_count = torch.sum(~true_noise_samples_mask & noise_samples_mask.cuda()).item()
    clean_count = clean_samples_mask.sum().item() 


    if(iter%100 ==0 ):
        print(f"The {epoch} epoch clean:{clean_count}")
        print(f"True noise samples count: {true_noise_count}")
        print(f"False noise samples count: {false_noise_count}")
        print(predis)
        print(labeldis)

    num_predis_less_than_K4 = torch.sum(predis < K/4)
    num_predis_less_than_K2 = torch.sum(predis < K/2)
    num_labeldis_less_than_K4 = torch.sum(labeldis < K/4)
    num_labeldis_less_than_K2 = torch.sum(labeldis < K/2)
  

    valid_outputs = outputs.cuda()[clean_samples_mask]
    valid_labels = labels.cuda()[clean_samples_mask]
    """
    if iter%100==0:
        print('mask:',clean_samples_mask)
        print('ave_a:',torch.mean(predis))
        print('ave_b:',torch.mean(labeldis))
        print('a4:',num_predis_less_than_K4)
        print('a2:',num_predis_less_than_K2)
        print('b4:',num_labeldis_less_than_K4)
        print('b2:',num_labeldis_less_than_K2)
    """
    if turn == False:
        return outputs,labels 
    return valid_outputs, valid_labels

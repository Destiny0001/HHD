import torch
from utils.cac import get_predicted_cate
import numpy as np

def label_refurb(epoch, labels, outputs,is_noise,  label_hash_codes, hashbits, device,iter,turn = True):
    
    K =1/2*hashbits
    predicted_label = get_predicted_cate(outputs,label_hash_codes,device)
    precodes = label_hash_codes[predicted_label.cpu()].to(device) 
    simcodes = torch.sign(outputs).to(device)
    labelcodes = label_hash_codes[labels].to(device)
    predis = torch.sum(simcodes!=precodes,dim=1).to(device)
    labeldis = torch.sum(simcodes!=labelcodes,dim=1).to(device)
    is_noise = is_noise.to(device)

    if epoch<5:
        turn = False
        
    if epoch<100:
        noise_samples_mask = (labeldis-predis>=0.5*K)&(labeldis>=K)
        #noise_samples_mask = (labeldis-predis>=0.5*K)
        noise_samples_mask = (labeldis+predis>=K)
        
    else:
        noise_samples_mask = (labeldis-predis>0.6*K)& (labeldis>K) 

    

    noise_indices = torch.where(noise_samples_mask)[0]  # 获取噪声样本的索引
    random_perm = torch.randperm(noise_indices.size(0), device=device)  # 生成随机排列
    select_num = int(noise_indices.size(0) * 0.6)  
    selected_indices = noise_indices[random_perm[:select_num]] 
    new_noise_samples_mask = torch.zeros_like(noise_samples_mask)
    new_noise_samples_mask[selected_indices] = True  # 更新噪声样本掩码
    noise_samples_mask = new_noise_samples_mask

    
    #noise_samples_mask = is_noise&noise_samples_mask
    clean_samples_mask = ~noise_samples_mask.to(device)
    true_noise_samples_mask = (is_noise==1).to(device)
    true_noise_count = torch.sum(true_noise_samples_mask & noise_samples_mask.to(device)).item()
    false_noise_count = torch.sum(~true_noise_samples_mask & noise_samples_mask.to(device)).item()
    clean_count = clean_samples_mask.sum().item() 


    if(iter%100 ==0 ):
        print(f"The {epoch} epoch clean:{clean_count}")
        print(f"True noise samples count: {true_noise_count}")
        print(f"False noise samples count: {false_noise_count}")
        print("predis:",predis)
        print("labels:",labeldis)



    valid_outputs = outputs.to(device)[clean_samples_mask]
    valid_labels = labels.to(device)[clean_samples_mask]
    valid_isnoise = labels.to(device)[clean_samples_mask]

    if turn == False:
        return outputs,labels,is_noise 
    return valid_outputs, valid_labels,valid_isnoise

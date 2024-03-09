import torch
from utils.cac import get_predicted_cate
import numpy as np

def hdot(a,b): #返回相同位的个数
    # 先进行元素级乘法
    elementwise_product = torch.mul(a, b)
    # 然后对每一行求和，以得到每一行的点积结果
    return torch.sum(elementwise_product, dim=1)

def label_refurb(epoch, labels, outputs,is_noise,  label_hash_codes, hashbits, device,iter,turn = True):
    
    K =1/2*hashbits
    predicted_label = get_predicted_cate(outputs,label_hash_codes,device)
    precodes = label_hash_codes[predicted_label.cpu()].to(device) 
    simcodes = torch.sign(outputs).to(device)
    labelcodes = label_hash_codes[labels].to(device)
    predis = torch.sum(simcodes!=precodes,dim=1).to(device)
    labeldis = torch.sum(simcodes!=labelcodes,dim=1).to(device)
    is_noise = is_noise.to(device)
    #assert (is_noise == (labeldis >= predis)).all()
    #assert ((not is_noise) == (labeldis <= predis)).all()

    if epoch<6:
        noise_samples_mask = (labeldis!=labeldis)
    elif epoch<30:
        #noise_samples_mask = (labeldis-predis>0.5*K)&(labeldis>=K)
        noise_samples_mask = (labeldis-predis>=0.5*K)
        
    elif epoch<100:
        noise_samples_mask = (labeldis-predis>0.6*K)& (labeldis>K) 
    
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
        print(predis)
        print(labeldis)

    num_predis_less_than_K4 = torch.sum(predis < K/4)
    num_predis_less_than_K2 = torch.sum(predis < K/2)
    num_labeldis_less_than_K4 = torch.sum(labeldis < K/4)
    num_labeldis_less_than_K2 = torch.sum(labeldis < K/2)
  

    valid_outputs = outputs.to(device)[clean_samples_mask]
    valid_labels = labels.to(device)[clean_samples_mask]
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

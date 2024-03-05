import torch
from utils.cac import get_predicted_cate

def hdot(a,b):
    # 先进行元素级乘法
    elementwise_product = torch.mul(a, b)
    # 然后对每一行求和，以得到每一行的点积结果
    return torch.sum(elementwise_product, dim=1)

def label_refurb(labels, outputs, label_hash_codes, hashbits, device,iter,turn = True):
    k = 4
    if turn == False:
        return outputs,labels 
    K =1/2*hashbits
    predicted = get_predicted_cate(outputs,label_hash_codes,device)
    predicted = label_hash_codes[predicted.cpu()].to(device) 
    simcodes = torch.sign(outputs).to(device)
    labels = label_hash_codes[labels].to(device)

    alpha = hdot(simcodes, predicted)
    beta = hdot(simcodes, labels)
    
    # 定义有效样本的掩码
    valid_samples_mask = (alpha >= 0) & (beta >= 0)
       # 计算(alpha < K/4) 和 (beta <K/4)的样本个数
    num_alpha_less_than_K4 = torch.sum(alpha < K/4)
    num_alpha_less_than_K2 = torch.sum(alpha < K/2)
    num_beta_less_than_K4 = torch.sum(beta < K/4)
    num_beta_less_than_K2 = torch.sum(beta < K/2)
  
    # 删除明显错误和模糊样本
    valid_outputs = outputs[valid_samples_mask]
    valid_labels = labels[valid_samples_mask]
    valid_alpha = alpha[valid_samples_mask].view(-1,1)
    valid_beta = beta[valid_samples_mask].view(-1,1)

    # 计算翻新标签
    k1 = 2*k/(1+k)
    k2 = 2/(1+k)
    refurb_labels = k1*(valid_beta / (valid_alpha + valid_beta)) * valid_labels + \
                    k2*(valid_alpha / (valid_alpha + valid_beta)) * predicted[valid_samples_mask]
    #refurb_labels = torch.sign(refurb_labels)  # 应用sgn函数
    refurb_labels = get_predicted_cate(refurb_labels,label_hash_codes,device)
    valid_labels = get_predicted_cate(valid_labels,label_hash_codes,device)
    predicted_labels = get_predicted_cate(predicted,label_hash_codes,device)
    if iter%100==0:
        print('mask:',valid_samples_mask)
        print('ave_a:',torch.mean(alpha))
        print('ave_b:',torch.mean(beta))
        print('a4:',num_alpha_less_than_K4)
        print('a2:',num_alpha_less_than_K2)
        print('b4:',num_beta_less_than_K4)
        print('b2:',num_beta_less_than_K2)
        print('refurb:',refurb_labels)
        print('valid:',valid_labels)
        print('predicted:',predicted_labels)
    return valid_outputs, refurb_labels

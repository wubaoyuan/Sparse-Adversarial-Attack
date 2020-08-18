import torch
import torch.nn as nn
import numpy as np
import os
import json
from PIL import Image

def project_shifted_lp_ball(x, shift_vec):
    shift_x = x - shift_vec

    # compute L2 norm: sum(abs(v)^2)^(1/2)
    norm2_shift = torch.norm(shift_x, 2)

    n = float(x.numel())
    xp = (n**(1/2))/2 * (shift_x / norm2_shift) + shift_vec
    return xp

def Normalization(x, mean_std):
    mean, std = mean_std
    result = (x - mean) / std

    return result

def compute_loss(model, images, target, epsilon, G, args, imgnet_normalized_ops, B, noise_Weight):

    # compute L2 loss of |e*G|_2^2 (L2 square for easy grad calculation)
    l2_loss = (torch.norm(G*epsilon*noise_Weight, 2).item())**2  
    
    # compute cnn-loss (cross-entropy loss)
    cnn_loss= compute_cnn_loss(model, images, target, epsilon, G, args, imgnet_normalized_ops)

    # compute group loss (group lasso)
    group_loss= compute_group_loss(G, B)
    
    # overall loss
    overall_loss = l2_loss + args.lambda1*cnn_loss+ args.lambda2*group_loss

    loss = {'loss': float(overall_loss),
            'l2_loss': float(l2_loss),
            'cnn_loss': float(cnn_loss),
            'group_loss':float(group_loss)
            }
    return loss


def compute_group_loss(G, B):  
    BG = B*G  #300*3*224*224
    index,dim,w,h = BG.shape
    Norm = torch.norm(BG.reshape(index,dim*w*h), p=2, dim=1) #300
    group_norm = torch.sum(Norm).item()  #1
    
    return  group_norm    

def compute_cnn_loss(model, images, target, epsilon, G, args, imgnet_normalized_ops):

    #
    image_s = images + torch.mul(G, epsilon)
    image_s = torch.clamp(image_s, args.min_pix_value, args.max_pix_value)
    image_s = Normalization(image_s, imgnet_normalized_ops)

    prediction = model(image_s)
    
    if args.loss == 'ce':
        ce = nn.CrossEntropyLoss()
        loss = ce(prediction, target)   #here loss 1*1 is a tensor, we can use loss.item() to obtain the scalar value

    elif args.loss == 'cw':
        label_to_one_hot = torch.tensor([[target.item()]]) #one-hot
        label_one_hot = torch.zeros(1, args.categories).scatter_(1, label_to_one_hot, 1).cuda()
        
        real = torch.sum(prediction*label_one_hot)
        other_max = torch.max((torch.ones_like(label_one_hot).cuda()-label_one_hot)*prediction - (label_one_hot*10000))
        loss = torch.clamp(other_max - real + args.confidence, min=0)

    return loss.item()   # .item() return a scalar , .detach() return a tensor

def compute_statistics(images, epsilon, G, args, B, Weight):
    
    
    epsilon_G = torch.mul(epsilon, G)                                     #1*3*224*224
    noise_images = torch.clamp(images+epsilon_G, args.min_pix_value, args.max_pix_value)   
    noise = noise_images - images
    w_noise = noise*Weight
    
    results = {
        'G_sum': float(torch.sum(G).item()),
        'L0': int(torch.sum((G > 0.5).float()).item()),
        'L1': float(torch.norm(noise, 1).item()),
        'L2': float((torch.norm(noise, 2).item())),
        'Li': float(torch.max(torch.abs(noise)).item()),
        'WL1': float(torch.norm(w_noise, 1).item()),
        'WL2': float((torch.norm(w_noise, 2).item())),
        'WLi': float(torch.max(torch.abs(w_noise)).item()),
    }

    return results

def compute_loss_statistic(model, images, target, epsilon, G, args, imgnet_normalized_ops, B, noise_Weight):

    loss = compute_loss(model, images, target, epsilon, G, args, imgnet_normalized_ops, B, noise_Weight)
    statistics = compute_statistics(images, epsilon, G, args, B, noise_Weight)

    results = {'loss': loss, 'statistics': statistics}
    return results

def compute_predictions_labels(model, images, epsilon, G, args, imgnet_normalized_ops):
    #whether to add epsilon
    image_s = images + torch.mul(G, epsilon)
    image_s = torch.clamp(image_s, args.min_pix_value, args.max_pix_value)
    adv_image = image_s   
    
    image_s = Normalization(image_s, imgnet_normalized_ops)

    predictions = model(image_s) # 1*c variable, c is the class
    predictions_labels = torch.argmax(predictions, dim=1)  #1*1 tensor with only one item, we can get it by predictions_labels[0]
    return predictions_labels.detach() , adv_image.detach()# .detach() is similar to .data in pytorch 0.3.1, that is to return a tensor

def parse_dict(input_dict):
    '''
    :param input_dict:
    :return: return the infos stored in input_dict
    '''

    result_info = ''
    for key in input_dict:
        temp_str=str(input_dict[key])
        result_info = result_info + key + ': ' + temp_str[0:7] + ', '

    return result_info



def save_results(results, args):
    if not os.path.exists(args.res_root):
        os.mkdir(args.res_root)
    res_path = os.path.join(args.res_root, results['img_name'].split('.')[0])    
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    np.save(os.path.join(res_path, str(results['label_target'])+'_noise.npy'), np.array(results['epsilon'], dtype ='float32'))
    np_img = (255 * np.array(results['adv_image'])).astype('uint8')
    im = Image.fromarray(np_img).convert('RGB')
    im.save(os.path.join(res_path, str(results['label_target'])+'_adv.png'))

def compute_sensitive(image, weight_type='none'):
    weight = torch.ones_like(image) 
    n, c, h, w = image.shape   #1,3,299,299
    
    if weight_type == 'none':
        return weight
        
    else:
        if weight_type == 'gradient':
            from scipy.ndimage import filters
            im = image.cpu().numpy().squeeze(axis=0).transpose((1,2,0))                            #229,229,3
            im_Prewitt_x = np.zeros(im.shape ,dtype='float32')
            im_Prewitt_y = np.zeros(im.shape ,dtype='float32')
            im_Prewitt_xy = np.zeros(im.shape ,dtype='float32')
    
            filters.prewitt(im, 1, im_Prewitt_x)
            filters.prewitt(im, 0, im_Prewitt_y)
            im_Prewitt_xy = np.sqrt(im_Prewitt_x ** 2 + im_Prewitt_y ** 2) 
            
            im_Prewitt_xy = im_Prewitt_xy.transpose((2,0,1))[np.newaxis,...]                       #1,3,299,299
            weight = torch.from_numpy(im_Prewitt_xy).cuda().float()
    
        else:
            for i in range(h):
                for j in range(w):
                    left = max(j - 1, 0)
                    right = min(j + 2, w)
                    up = max(i - 1, 0)
                    down = min(i + 2, h)
                    
                    for k in range(c):
                        if weight_type == 'variance':
                            weight[0, k, i, j] = torch.std(image[0, k, up:down, left:right])
                        elif weight_type == 'variance_mean':
                            weight[0, k, i, j] = torch.std(image[0, k, up:down, left:right]) * torch.mean(image[0, k, up:down, left:right])
                        elif weight_type == 'contrast':
                            weight[0, k, i, j] = (torch.max(image[0, k, up:down, left:right]) - torch.min(image[0, k, up:down, left:right])) / (torch.max(image[0, k, up:down, left:right]) + torch.min(image[0, k, up:down, left:right]))
                        elif weight_type == 'contrast_mean':
                            contrast = (torch.max(image[0, k, up:down, left:right]) - torch.min(image[0, k, up:down, left:right])) / (torch.max(image[0, k, up:down, left:right]) + torch.min(image[0, k, up:down, left:right]))
                            weight[0, k, i, j] = contrast * torch.mean(image[0, k, up:down, left:right])
                            
                        if torch.isnan(weight[0, k, i, j]):
                            weight[0, k, i, j] = 1e-4
                            
        weight = 1.0 / (weight + 1e-4) 
        for k in range(c):
            weight[0, k, :, :] = (weight[0, k, :, :] - torch.min(weight[0, k, :, :])) / (torch.max(weight[0, k, :, :]) - torch.min(weight[0, k, :, :]))
        
        return weight



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn

class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

        return loss  
 

class InfoNCE_class(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, label_pred, label_gt,logit_scale):
        label_gt = label_gt.squeeze(1)
        loss_class = self.loss_function(label_pred, label_gt)
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2
        #print('loss_class:{}, loss:{}'.format(loss_class, loss))
        return loss+0.01*loss_class
       # return loss+0.01*loss_class

 
class InfoNCE_noise(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device
        self.L1_smooth = torch.nn.SmoothL1Loss(reduction = 'mean')
    

    def forward(self, image_features1, image_features2, noise_pred=None, noise_gt=None, image_features1_teacher=None, image_features2_teacher=None,logit_scale=None):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
            
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2
        # print('loss_class:{}, loss:{}'.format(loss_class, loss))
        noise_loss = 0.
        if noise_pred is not None:
            #pixel_shift = (noise_pred * 768).int() # similar to crop operation
      
            #print(noise_pred)
            #print(noise_gt)
            noise_loss = self.L1_smooth(noise_pred, noise_gt.float())
             #loss = loss + 0.01*noise_loss
        loss_distill_1 = 0.
        if image_features1_teacher is not None:
            image_features1_teacher = F.normalize(image_features1_teacher, dim=-1)
            logits_per_image1_distill = logit_scale * image_features1_teacher @ image_features1.T
            logits_per_image1_distill_T = logits_per_image1_distill.T
            labels_distill_1 = torch.arange(len(logits_per_image1_distill), dtype=torch.long, device=self.device)
            loss_distill_1 = (self.loss_function(logits_per_image1_distill, labels_distill_1) + self.loss_function(logits_per_image1_distill_T, labels_distill_1))/2
    
            #image_features2_teacher = F.normalize(image_features2_teacher, dim=-1)
            #logits_per_image2_distill = logit_scale * image_features2_teacher @ image_features2.T
            #logits_per_image2_distill_T = logits_per_image2_distill.T
            #labels_distill_2 = torch.arange(len(logits_per_image2_distill), dtype=torch.long, device=self.device)
            #loss_distill_2 = (self.loss_function(logits_per_image2_distill, labels) + self.loss_function(logits_per_image2_distill_T, labels_distill_2))/2
  
        loss = loss + 0.01*loss_distill_1 + 0.01*noise_loss

        return loss


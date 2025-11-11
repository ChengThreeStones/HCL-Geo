import torch
import timm
import numpy as np
import torch.nn as nn
from einops import rearrange, repeat

class TimmModel(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=384,):
                 
        super(TimmModel, self).__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0,pretrained_cfg_overlay = dict(file="convnext_base_22k_1k_384.pth"))
            
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img1, img2=None,mode=None):
        
        if img2 is not None:
            
            image_features1 = self.model(img1)   
    
            image_features2 = self.model(img2)
            
            return image_features1, image_features2   
              
        else:
            if mode:
                image_features = self.model(img1)   
            
            else:
                image_features = self.model(img1)
                
            return image_features


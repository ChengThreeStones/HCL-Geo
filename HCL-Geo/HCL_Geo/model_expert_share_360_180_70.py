import torch
import timm
import numpy as np
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F


class LastExpert(nn.Module):
    def __init__(self,model_name,continue_learn=None):
        super(LastExpert,self).__init__()

        #-------- convnext_block --------
        expert = timm.create_model(model_name, pretrained=True, num_classes=0,pretrained_cfg_overlay = dict(file="/home/threestone/cl/A_Conv_Sample4Geo/Sample4Geo_main/convnext_base_22k_1k_384.pth"))
        if continue_learn is not None:
            expert = init_weight_share(expert,continue_learn)
        # for shallow expert structure
        self.expert = nn.Sequential(
            expert.stages[3].blocks[-1],
            expert.head
        )
        
        # for deep expert structure
        #self.expert = nn.Sequential(
        #                            expert.stages[3],
        #                            expert.head
        #)
    def forward(self,x):

        return self.expert(x)

                

def init_weight_share(model,dict):
    tmp_dict = torch.load(dict)
    del tmp_dict['logit_scale']
    model_dict = model.state_dict()
    for k in list(tmp_dict.keys()):
        if 'pred' in k:
            del tmp_dict[k]    

    tmp_dict_list = list(tmp_dict.keys())
    model_dict_list = list(model_dict.keys())
    if len(tmp_dict_list) == len(model_dict_list):
        print('successful for load pretrained params')
    else:
        print('prepared error, model_params_nums:{}, but dict_nums:{}'.format(len(model_dict_list), len(tmp_dict_list)))
    for i in range(len(tmp_dict_list)):
        model_dict[model_dict_list[i]] = tmp_dict[tmp_dict_list[i]] 
        
    model.load_state_dict(model_dict, strict=True)

    return model


class TimmModel_Share(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=384,
                 teacher_path=None
                 ):
                 
        super(TimmModel_Share, self).__init__()
        
        self.img_size = img_size
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0,pretrained_cfg_overlay = dict(file="convnext_base_22k_1k_384.pth"))
            if teacher_path is not None:
                self.model = init_weight_share(self.model,teacher_path)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
 
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img1, img2=None,mode=None,noise_pred=None,limit_FoV=None):
        # noise_pred and limit_Fov are for continue learning
        if img2 is not None:
            feature_map = self.model.stem(img1)
            feature_map = self.model.stages[:4](feature_map)
            if noise_pred is not None:
                
                B,C,H,W = feature_map.shape
                a = torch.arange(feature_map.shape[3]).cuda()
                final_width = feature_map.shape[3] * limit_FoV / 360
                final_width = torch.tensor(final_width, dtype=torch.int64).to("cuda")
                feature_map_crop = torch.empty(B,C,H,final_width).cuda()
                
                for i in range(feature_map.shape[0]):
                    noise_pred_pixel = noise_pred[i,:] * feature_map.shape[3]
                    noise_pred_pixel = torch.tensor(noise_pred_pixel, dtype=torch.int64).to("cuda")
                
                    feature_map_crop[i,:,:,:] = feature_map[i,:,:,((a-noise_pred_pixel)%feature_map.shape[3])[:final_width]]
                
                image_features1 = self.model.head(feature_map_crop)

            else:
                image_features1 = self.model.head(feature_map)

        
            
            image_features2 = self.model(img2)
            
            return image_features1, image_features2
              
        else:
            if mode:
                image_features = self.model(img1)   
            
            else:
                image_features = self.model(img1)
                
            return image_features

class TimmModel_Review_Share(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=384,
                 continue_learn=None):
                 
        super(TimmModel_Review_Share, self).__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0,pretrained_cfg_overlay = dict(file="convnext_base_22k_1k_384.pth"))
            
            if continue_learn:
                self.model = init_weight_share(self.model,continue_learn)
                
        # for extend expert structure
        #self.grd_model = nn.Sequential(
        #                grd_model.stem,
        #                grd_model.stages,                
        #    )        
     
        # for shallow expert structure
        self.bk_model = nn.Sequential(
                        self.model.stem,
                        self.model.stages[:3],
                        self.model.stages[3].downsample,
                        self.model.stages[3].blocks[:2]
            )
        
        # for deep expert structure
        #self.grd_model = nn.Sequential(
        #                          grd_model.stem,
        #                          grd_model.stages[:3],
        #                            )

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.last_expert_70  = LastExpert(model_name = model_name,continue_learn=continue_learn)
        #self.last_expert_90  = LastExpert(model_name = model_name,continue_learn=continue_learn)
        self.last_expert_180 = LastExpert(model_name = model_name,continue_learn=continue_learn)
        #self.last_expert_270 = LastExpert(model_name = model_name,continue_learn=continue_learn)
        self.last_expert_360 = LastExpert(model_name = model_name,continue_learn=continue_learn)
        self.last_expert_360_know = LastExpert(model_name = model_name,continue_learn=continue_learn)
        classifier = LastExpert(model_name = model_name,continue_learn=continue_learn)


        self.classifier = nn.Sequential(
                            classifier,
                            nn.Linear(1024, 1024//4),   
                            nn.GELU(),
                            nn.Dropout(0.1),
                            nn.Linear(1024//4, 1024//16),
                            nn.GELU(),
                            nn.Linear(1024//16, 4),
                            nn.Softmax(dim=-1),
                            )
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img1, img2=None,mode=None, limit_Fov=360, mix_input = 0):

        if img2 is not None:
            image_features1 = self.bk_model(img1)           
            image_features_class = self.classifier(image_features1)
            image_features1_360_know = self.last_expert_360_know(image_features1)
            image_features1_360 = self.last_expert_360(image_features1)
            #image_features1_270 = self.last_expert_270(image_features1)
            image_features1_180 = self.last_expert_180(image_features1)
            #image_features1_90 = self.last_expert_90(image_features1)    
            image_features1_70 = self.last_expert_70(image_features1)
            #image_features1 =image_features1_360_know*image_features_class[:,0].unsqueeze(1) +image_features1_360*image_features_class[:,1].unsqueeze(1) + image_features1_270*image_features_class[:,2].unsqueeze(1) + image_features1_180*image_features_class[:,3].unsqueeze(1)  + image_features1_90*image_features_class[:,4].unsqueeze(1)  + image_features1_70*image_features_class[:,5].unsqueeze(1) 
            image_features1 =image_features1_360_know*image_features_class[:,0].unsqueeze(1) +image_features1_360*image_features_class[:,1].unsqueeze(1) +  image_features1_180*image_features_class[:,2].unsqueeze(1)  + image_features1_70*image_features_class[:,3].unsqueeze(1) 

            image_features2 = self.bk_model(img2)
            image_features2_360_know = self.last_expert_360_know(image_features2)
            image_features2_360 = self.last_expert_360(image_features2)
            #image_features2_270 = self.last_expert_270(image_features2)
            image_features2_180 = self.last_expert_180(image_features2)
            #image_features2_90 = self.last_expert_90(image_features2)    
            image_features2_70 = self.last_expert_70(image_features2)
            #image_features2 = (image_features2_360_know + image_features2_360 + image_features2_270 + image_features2_180 + image_features2_90 + image_features2_70) / 6.
            image_features2 = (image_features2_360_know + image_features2_360 + image_features2_180 +  image_features2_70) /4.
            
            return image_features1, image_features2, image_features_class 
              
        else:
            if mode:
                image_features1 = self.bk_model(img1)           
                image_features_class = self.classifier(image_features1)
                image_features1_360_know = self.last_expert_360_know(image_features1)
                image_features1_360 = self.last_expert_360(image_features1)
                #image_features1_270 = self.last_expert_270(image_features1)
                image_features1_180 = self.last_expert_180(image_features1)
                #image_features1_90 = self.last_expert_90(image_features1)    
                image_features1_70 = self.last_expert_70(image_features1)
                #image_features1 =image_features1_360_know*image_features_class[:,0].unsqueeze(1) +image_features1_360*image_features_class[:,1].unsqueeze(1) + image_features1_270*image_features_class[:,2].unsqueeze(1) + image_features1_180*image_features_class[:,3].unsqueeze(1)  + image_features1_90*image_features_class[:,4].unsqueeze(1)  + image_features1_70*image_features_class[:,5].unsqueeze(1) 
                image_features1 =image_features1_360_know*image_features_class[:,0].unsqueeze(1) +image_features1_360*image_features_class[:,1].unsqueeze(1) +  image_features1_180*image_features_class[:,2].unsqueeze(1)  + image_features1_70*image_features_class[:,3].unsqueeze(1) 

                return image_features1
            else:

                image_features2 = self.bk_model(img1)
                image_features2_360_know = self.last_expert_360_know(image_features2)
                image_features2_360 = self.last_expert_360(image_features2)
                #image_features2_270 = self.last_expert_270(image_features2)
                image_features2_180 = self.last_expert_180(image_features2)
                #image_features2_90 = self.last_expert_90(image_features2)    
                image_features2_70 = self.last_expert_70(image_features2)
                #image_features2 = (image_features2_360_know + image_features2_360 + image_features2_270 + image_features2_180 + image_features2_90 + image_features2_70) / 6.
                image_features2 = (image_features2_360_know + image_features2_360 + image_features2_180 +  image_features2_70) / 4.
                
                return image_features2

class TimmModel_Contin_Share(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=384,
                 continue_learn=None):
                 
        super(TimmModel_Contin_Share, self).__init__()
        
        self.img_size = img_size
        self.continue_learn = continue_learn
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0,pretrained_cfg_overlay = dict(file="/home/threestone/cl/A_Conv_Sample4Geo/Sample4Geo_main/convnext_base_22k_1k_384.pth"))
            
        if continue_learn:
            #predict_model = LastExpert(model_name = model_name,continue_learn=continue_learn)
            self.predict_model = nn.Sequential(
                nn.Linear(1024, 1024//4),   
                nn.GELU(),
                nn.Linear(1024//4, 1024//16),
                nn.GELU(),
                nn.Linear(1024//16, 1),
                #nn.RELU(),
                nn.Sigmoid()
            )
            for layer in self.predict_model:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight.data, gain=1)
                    nn.init.constant_(layer.bias.data, 0)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

            
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img1, img2=None,teacher=None, mode=None):
        
        if img2 is not None:
            
            image_features1 = self.model(img1)
            
            if self.continue_learn and teacher is not None:
                q_feature = F.normalize(image_features1, dim=-1)
    
                r_feature = F.normalize(teacher, dim=-1)
            
                score = q_feature @ r_feature.T 

                mix_feature = image_features1 + score @ teacher
                noise_pred = self.predict_model(mix_feature)
            else:
                noise_pred = None
        
            
            image_features2 = self.model(img2)
            
            return image_features1, image_features2, noise_pred   
              
        else:
            if mode:
                image_features = self.model(img1)   
            
            else:
                image_features = self.model(img1)
                
            return image_features


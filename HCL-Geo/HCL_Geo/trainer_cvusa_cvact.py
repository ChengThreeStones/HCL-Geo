import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F

def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None,model_teacher=None):

    # set model train mode
    model.train()
   
    
    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query, query_fov, reference, ids, noise_label in bar:
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                query_fov = query_fov.to(train_config.device)
                noise_label = noise_label.to(train_config.device)
            
                # Forward pass
                features1, features2, noise_pred = model(img1=query_fov, img2=reference)
                
                if model_teacher:
                    with torch.no_grad():
                        features1_fov, features2_fov = model_teacher(img1=query, img2=reference, noise_pred = noise_pred, limit_FoV = train_config.limit_Fov)
                if train_config.noise == 360 and model_teacher is None:
                  
                    if torch.cuda.device_count() > 1 and train_config.gpu_ids> 1: 
                        loss = loss_function(features1, features2,noise_pred,noise_label,model.module.logit_scale.exp())
                    else:
                        loss = loss_function(features1, features2,noise_pred=noise_pred,noise_gt=noise_label,logit_scale = model.logit_scale.exp()) 
                elif train_config.noise == 360 and model_teacher is not None:
                
                    if torch.cuda.device_count() > 1 and train_config.gpu_ids> 1: 
                        loss = loss_function(features1, features2, noise_pred, noise_label, features1_fov, features2_fov, model.module.logit_scale.exp())
                    else:
                        loss = loss_function(features1, features2,noise_pred, noise_label, features1_fov, features2_fov, model.logit_scale.exp()) 
                
                elif train_config.noise == 0:
                    if torch.cuda.device_count() > 1 and train_config.gpu_ids> 1: 
                        loss = loss_function(features1, features2, model.module.logit_scale.exp())
                    else:
                        loss = loss_function(features1, features2, logit_scale = model.logit_scale.exp())
                
                else:
                    print("ERROR")

                losses.update(loss.item())
                
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
        
            if model_teacher:
                with torch.no_grad():
                    features1_fov, features2_fov = model_teacher(img1=query, img2=reference, noise_pred = noise_pred, limit_FoV = train_config.limit_Fov)
            if train_config.noise == 360 and model_teacher is None:
                
                if torch.cuda.device_count() > 1 and train_config.gpu_ids> 1: 
                    loss = loss_function(features1, features2,noise_pred,noise_label,model.module.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2,noise_pred=noise_pred,noise_gt=noise_label,logit_scale = model.logit_scale.exp()) 
            elif train_config.noise == 360 and model_teacher is not None:
            
                if torch.cuda.device_count() > 1 and train_config.gpu_ids> 1: 
                    loss = loss_function(features1, features2, noise_pred, noise_label, features1_fov, features2_fov, model.module.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2,noise_pred, noise_label, features1_fov, features2_fov, model.logit_scale.exp()) 
            
            elif train_config.noise == 0:
                if torch.cuda.device_count() > 1 and train_config.gpu_ids> 1: 
                    loss = loss_function(features1, features2, model.module.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2, logit_scale = model.logit_scale.exp())
            
            else:
                print("ERROR")
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader,mode = False):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = torch.empty([len(dataloader.dataset), 1024])
    #if vigor:
    #if mode:
    #    ids_list = torch.empty([len(dataloader.dataset),4])
    #else:
    #    ids_list = torch.empty([len(dataloader.dataset)])
    #else:
    ids_list = torch.empty([len(dataloader.dataset)])

    # -------------------- count the pred task labels and weights for testing-----------------------
    #class_north_aligned_label = 0
    #class_north_aligned_weights = 0.
    #class_360_label = 0
    #class_360_weights = 0.
    #class_180_label = 0
    #class_180_weights = 0.
    #class_70_label = 0
    #class_70_weights = 0. 
    
    with torch.no_grad():
        i = 0
        for img, ids in bar:

            if mode:
                ids_list[i*train_config.batch_size:min((i+1)*train_config.batch_size , len(dataloader.dataset))] = ids
            else:
                ids_list[i*train_config.batch_size:min((i+1)*train_config.batch_size , len(dataloader.dataset))] = ids
            with autocast():
                
                img = img.to(train_config.device)
                if mode:
                    img_feature = model(img,mode=mode)
                    #img_feature, class_pred = model(img, mode=mode)
                    # -------------------- count the pred task labels and weights for testing-----------------------
                    #value,index = torch.max(class_pred,1)
                    #for j in range(class_pred.shape[0]):
                   
                        #class_north_aligned_weights += class_pred[j,0]
                        #class_360_weights += class_pred[j,1]
                        #class_180_weights += class_pred[j,2]
                        #class_70_weights += class_pred[j,3]
                        #if index[j] == 0:
                        #    class_north_aligned_label += 1
                        #elif index[j] == 1:
                        #    class_360_label += 1
                        #elif index[j] == 2:
                        #    class_180_label += 1
                        #elif index[j] == 3:
                        #    class_70_label += 1
                        #else:
                        #    print("Some ERROR, MayBe Using False Model")
                else:
                    img_feature = model(img,mode=mode)
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list[i*train_config.batch_size:min((i+1)*train_config.batch_size , len(dataloader.dataset)), :] = img_feature.to(torch.float32)
            i += 1
      

    if train_config.verbose:
        bar.close()
    if mode:
        #print("class_north_aligned_label num is{}".format(class_north_aligned_label))    
        #print("class_360_label num is{}".format(class_360_label))
        #print("class_180_label num is{}".format(class_180_label))
        #print("class_70_label num is{}".format(class_70_label))
        #print("class_north_aligned avg is{}".format(class_north_aligned_weights / 8884))
        #print("class_360_weight avg is{}".format(class_360_weights / 8884))
        #print("class_180_weight avg is{}".format(class_180_weights / 8884))
        #print("class_70_weight avg is{}".format(class_70_weights / 8884))
        return img_features_list, ids_list
    else:
        return img_features_list, ids_list
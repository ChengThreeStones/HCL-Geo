import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F

def train(train_config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

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
    for query, reference, ids, fov_label in bar:

       
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                fov_label = fov_label.to(train_config.device)
            
                # Forward pass
                features1, features2, fov_pred = model(img1=query, img2=reference)
               
                if torch.cuda.device_count() > 1 and train_config.gpu_ids> 1: 
                    loss = loss_function(features1, features2, fov_pred, fov_label, model.module.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2, fov_pred, fov_label,model.logit_scale.exp()) 
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
        
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)
            fov_label = fov_label.to(train_config.device)

            # Forward pass
            features1, features2, fov_pred = model(img1=query, img2=reference)
                
            if torch.cuda.device_count() > 1 and train_config.gpu_ids> 1: 
                loss = loss_function(features1, features2, fov_pred, fov_label, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2,fov_pred, fov_label, model.logit_scale.exp()) 
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


def predict(train_config, model, dataloader,limit_fov,mode = False):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = torch.empty([len(dataloader.dataset), 1024])
    
    ids_list = torch.empty([len(dataloader.dataset)])
    with torch.no_grad():
        i = 0
        for img, ids in bar:

            ids_list[i*train_config.batch_size:min((i+1)*train_config.batch_size , len(dataloader.dataset))] = ids
            
            with autocast():
                
                img = img.to(train_config.device)
                img_feature = model(img, limit_Fov = train_config.limit_Fov,mode=mode)
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list[i*train_config.batch_size:min((i+1)*train_config.batch_size , len(dataloader.dataset)), :] = img_feature.to(torch.float32)
            i += 1

    if train_config.verbose:
        bar.close()
        
    return img_features_list, ids_list
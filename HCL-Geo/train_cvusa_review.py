import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import math
import shutil
import sys
import torch
import pickle
import argparse
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from HCL_Geo.dataset.cvusa_fov_360_180_70 import CVUSADatasetEval, CVUSADatasetTrain
from HCL_Geo.transforms import get_transforms_train, get_transforms_val
from HCL_Geo.utils import setup_system, Logger
from HCL_Geo.trainer_class import train
from HCL_Geo.evaluate.cvusa_and_cvact_class import evaluate, calc_sim
from HCL_Geo.loss import InfoNCE_class
from HCL_Geo.model_expert_share_360_180_70 import TimmModel_Review_Share

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument('--model',default='convnext_base.fb_in22k_ft_in1k_384',type=str,
                    help="the bockbone for net")
parser.add_argument('--img_size',default=384, type=int, 
                    help='the image_size for sat')
parser.add_argument('--mixed_precision',action='store_true',
                    help='use mixed_precision to trianing')
parser.add_argument('--seed', default=42, type=int,
                    help='choose seed for random')
parser.add_argument('--epochs', default=80, type=int,
                    help='the epoch for trainingf')
parser.add_argument('--batch_size',default=16, type=int,
                    help='batchsize for trian')
parser.add_argument('--verbose', action='store_true', 
                    help='the process for train and test')
parser.add_argument('--gpu_ids', default=1, type=int,
                    help='the gpu_id for train')
# using data mining
parser.add_argument('--custom_sampling',action='store_true',
                    help='using data mining instead for random')
parser.add_argument('--gps_sample',action='store_true',
                    help='using gps for data mining in early training, same as sample4_geo')
parser.add_argument('--sim_sample',action='store_true',
                    help='using simility for data mining in early training, same as sample4_geo')
parser.add_argument('--neighbour_select',default=16,type=int,
                    help='the number of hard sample in one batch')
parser.add_argument('--neighbour_range',default=128,type=int,
                    help='the size of pool for selecting hard samples')
parser.add_argument('--gps_dict_path',default='',type=str,
                    help='the path for gps file using for gps_sample')
parser.add_argument('--eval_epoch',default=4,type=int,
                    help='eval every n Epoch')
parser.add_argument('--normalize_features',action='store_true',
                    help='normalize final features')
parser.add_argument('--clip_grad',default=100.,type=float,
                    help='clip gradient')
parser.add_argument('--label_smoothing',default=0.1,type=float)
parser.add_argument('--lr',default=0.0001,type=float,
                    help='learning rate')
parser.add_argument('--scheduler',default='cosine',type=str,
                    help='polynomial, cosine, constant, None')
parser.add_argument('--warmup_epochs',default=1 ,type=int,
                    help='the epoch for warmup')
parser.add_argument('--lr_end',default=0.00001 ,type=float,
                    help='the lr for learning end')
parser.add_argument('--data_folder',default='' ,type=str,
                    help='the dataset path')
parser.add_argument('--prob_rotate',default=0.75 ,type=float,
                    help='the probability of rotate for image')
parser.add_argument('--prob_flip',default=0.5 ,type=float,
                    help='the probability of flip for image')
parser.add_argument('--model_path',default='' ,type=str,
                    help='the path for saving model')
parser.add_argument('--zero_shot',action='store_true',
                    help='Eval before training')  
parser.add_argument('--checkpoint_start',default='' ,type=str,
                    help='Checkpoint to start from')  
parser.add_argument('--num_workers',default=4,type=int,
                    help='set num_workers to 4')  
parser.add_argument('--device',default='cuda',type=str,
                    help='train on GPU if available, cuda or cpu')  
parser.add_argument('--cudnn_benchmark',action='store_true',
                    help='for better performance')       
parser.add_argument('--limit_Fov', default=360, type=int,
                    help='the fov of grd image, 360, 180, 90, 70')
parser.add_argument('--Fov_Mix', default=0, type=int,
                    help='0,1,2')
parser.add_argument('--continue_learn', default=None, type=str, help='continue learning for eary to hard data flow')


    


if __name__ == '__main__':

    args = parser.parse_args() 
    model_path = "{}/review".format(args.model_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(__file__, "{}/train.py".format(model_path))


    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(seed=args.seed,
                 cudnn_benchmark=args.cudnn_benchmark,
                 cudnn_deterministic=False)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(args.model))


    model = TimmModel_Review_Share(args.model,
                      pretrained=True,
                      img_size=args.img_size,
                      continue_learn = args.continue_learn)
    
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = args.img_size
    image_size_sat = (img_size, img_size)
    new_width = 768   
    new_hight = round((224 / 1232) * new_width)
    img_size_ground = (new_hight, new_width)

    # Load pretrained Checkpoint    
    if args.checkpoint_start:  
        print("Start from:", args.checkpoint_start)
        model_state_dict = torch.load(args.checkpoint_start)  
        model.load_state_dict(model_state_dict, strict=False)     
      
    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and args.gpu_ids > 1:
        model = torch.nn.DataParallel(model, device_ids=(0,1))
            
    # Model to device   
    model = model.to(args.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    # choose expert for train

    for name, params in model.named_parameters():
        if params.requires_grad:
            print(name)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{trainable_params:,} training parameters.')

    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    sat_transforms_train, ground_transforms_train = get_transforms_train(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   )
                                                                   
                                                                   
    # Train
    train_dataset = CVUSADatasetTrain(data_folder=args.data_folder ,
                                      transforms_query=ground_transforms_train,
                                      transforms_reference=sat_transforms_train,
                                      prob_flip=args.prob_flip,
                                      prob_rotate=args.prob_rotate,
                                      shuffle_batch_size=args.batch_size,
                                      )
    
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=not args.custom_sampling,
                                  pin_memory=True)
    
    
    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                               img_size_ground,
                                                               mean=mean,
                                                               std=std,
                                                               )


    # Reference Satellite Images
    reference_dataset_test = CVUSADatasetEval(data_folder=args.data_folder ,
                                              split="test",
                                              img_type="reference",
                                              transforms=sat_transforms_val,
                                              )
    
    reference_dataloader_test = DataLoader(reference_dataset_test,
                                           batch_size=args.batch_size,
                                           num_workers=args.num_workers,
                                           shuffle=False,
                                           pin_memory=True)
    
    
    
    # Query Ground Images Test
    query_dataset_test_360_know = CVUSADatasetEval(data_folder=args.data_folder ,
                                          split="test",
                                          img_type="query",    
                                          transforms=ground_transforms_val,
                                          limit_fov=360
                                          )
    
    query_dataloader_test_360_know = DataLoader(query_dataset_test_360_know,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    query_dataset_test_360 = CVUSADatasetEval(data_folder=args.data_folder ,
                                          split="test",
                                          img_type="query",    
                                          transforms=ground_transforms_val,
                                          limit_fov=360,
                                          noise = 360
                                          )
        
    query_dataloader_test_360 = DataLoader(query_dataset_test_360,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    query_dataset_test_270 = CVUSADatasetEval(data_folder=args.data_folder ,
                                          split="test",
                                          img_type="query",    
                                          transforms=ground_transforms_val,
                                          limit_fov=270,
                                          noise = 360
                                          )
        
    query_dataloader_test_270 = DataLoader(query_dataset_test_270,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    query_dataset_test_180 = CVUSADatasetEval(data_folder=args.data_folder ,
                                          split="test",
                                          img_type="query",    
                                          transforms=ground_transforms_val,
                                          limit_fov=180
                                          )
    
    query_dataloader_test_180 = DataLoader(query_dataset_test_180,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       shuffle=False,
                                       pin_memory=True)    
    query_dataset_test_90 = CVUSADatasetEval(data_folder=args.data_folder ,
                                          split="test",
                                          img_type="query",    
                                          transforms=ground_transforms_val,
                                          limit_fov=90
                                          )
    
    query_dataloader_test_90 = DataLoader(query_dataset_test_90,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    query_dataset_test_70 = CVUSADatasetEval(data_folder=args.data_folder ,
                                          split="test",
                                          img_type="query",    
                                          transforms=ground_transforms_val,
                                          limit_fov=70
                                          )
    
    query_dataloader_test_70 = DataLoader(query_dataset_test_70,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       shuffle=False,
                                       pin_memory=True)        
    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test_360))
    
    
    #-----------------------------------------------------------------------------#
    # GPS Sample                                                                  #
    #-----------------------------------------------------------------------------#
    if args.gps_sample:
        with open(args.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None

    #-----------------------------------------------------------------------------#
    # Sim Sample                                                                  #
    #-----------------------------------------------------------------------------#
    
    if args.sim_sample:
    
        # Query Ground Images Train for simsampling
        query_dataset_train = CVUSADatasetEval(data_folder=args.data_folder ,
                                               split="train",
                                               img_type="query",   
                                               transforms=ground_transforms_val,                                           
                                               )
            
        query_dataloader_train = DataLoader(query_dataset_train,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            shuffle=False,
                                            pin_memory=True)
        
        
        reference_dataset_train = CVUSADatasetEval(data_folder=args.data_folder ,
                                                   split="train",
                                                   img_type="reference", 
                                                   transforms=sat_transforms_val,
                                                   )
        
        reference_dataloader_train = DataLoader(reference_dataset_train,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                shuffle=False,
                                                pin_memory=True)


        print("\nReference Images Train:", len(reference_dataset_train))
        print("Query Images Train:", len(query_dataset_train))        

    
    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    loss_function = InfoNCE_class(loss_function=loss_fn,
                            device=args.device,
                            )

    if args.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None
        
    #-----------------------------------------------------------------------------#
    # optimizer                                                                   #
    #-----------------------------------------------------------------------------#


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#

    train_steps = len(train_dataloader) * args.epochs
    warmup_steps = len(train_dataloader) * args.warmup_epochs
       
    if args.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(args.lr, args.lr_end))  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = args.lr_end,
                                                              power=1.5,
                                                              num_warmup_steps=warmup_steps)
        
    elif args.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(args.lr))   
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
        
    elif args.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(args.lr))   
        scheduler =  get_constant_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=warmup_steps)
           
    else:
        scheduler = None
        
    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(args.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(args.epochs, train_steps))
        
        
    #-----------------------------------------------------------------------------#
    # Zero Shot                                                                   #
    #-----------------------------------------------------------------------------#
    if args.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))  

      
        r1_test_fov360_know = evaluate(config=args,
                            model=model,
                            reference_dataloader=reference_dataloader_test,
                            query_dataloader=query_dataloader_test_360_know, 
                            ranks=[1, 5, 10],
                            step_size=1000,
                            cleanup=True,
                            limit_fov=360)
        
        r1_test_fov360 = evaluate(config=args,
                            model=model,
                            reference_dataloader=reference_dataloader_test,
                            query_dataloader=query_dataloader_test_360, 
                            ranks=[1, 5, 10],
                            step_size=1000,
                            cleanup=True,
                            limit_fov=360)
        r1_test_fov270 = evaluate(config=args,
                            model=model,
                            reference_dataloader=reference_dataloader_test,
                            query_dataloader=query_dataloader_test_270, 
                            ranks=[1, 5, 10],
                            step_size=1000,
                            cleanup=True,
                            limit_fov=270)
        r1_test_fov180 = evaluate(config=args,
                            model=model,
                            reference_dataloader=reference_dataloader_test,
                            query_dataloader=query_dataloader_test_180, 
                            ranks=[1, 5, 10],
                            step_size=1000,
                            cleanup=True,
                            limit_fov=180)
        r1_test_fov90 = evaluate(config=args,
                            model=model,
                            reference_dataloader=reference_dataloader_test,
                            query_dataloader=query_dataloader_test_90, 
                            ranks=[1, 5, 10],
                            step_size=1000,
                            cleanup=True,
                            limit_fov=90)
        r1_test_fov70 = evaluate(config=args,
                            model=model,
                            reference_dataloader=reference_dataloader_test,
                            query_dataloader=query_dataloader_test_70, 
                            ranks=[1, 5, 10],
                            step_size=1000,
                            cleanup=True,
                            limit_fov=70)
        
        if args.sim_sample:
            r1_train, sim_dict = calc_sim(config=args,
                                          model=model,
                                          reference_dataloader=reference_dataloader_train,
                                          query_dataloader=query_dataloader_train, 
                                          ranks=[1, 5, 10],
                                          step_size=1000,
                                          cleanup=True)
                
    #-----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    #-----------------------------------------------------------------------------#            
    if args.custom_sampling:
        train_dataloader.dataset.shuffle(sim_dict,
                                         neighbour_select=args.neighbour_select,
                                         neighbour_range=args.neighbour_range)
            
    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    start_epoch = 0   
    best_score = 0
    

    for epoch in range(1, args.epochs+1):

        train_loss = train(args,
                           model,
                           dataloader=train_dataloader,
                           loss_function=loss_function,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           scaler=scaler)
        
        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr']))
        
        # evaluate
        if (epoch % args.eval_epoch == 0 and epoch != 0) or epoch == args.epochs:
        
            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))

            r1_test_fov360_know = evaluate(config=args,
                               model=model,
                               reference_dataloader=reference_dataloader_test,
                               query_dataloader=query_dataloader_test_360_know, 
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True,
                               limit_fov=360)
            
            r1_test_fov360 = evaluate(config=args,
                               model=model,
                               reference_dataloader=reference_dataloader_test,
                               query_dataloader=query_dataloader_test_360, 
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True,
                               limit_fov=360)
            r1_test_fov270 = evaluate(config=args,
                               model=model,
                               reference_dataloader=reference_dataloader_test,
                               query_dataloader=query_dataloader_test_270, 
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True,
                               limit_fov=270)
            r1_test_fov180 = evaluate(config=args,
                               model=model,
                               reference_dataloader=reference_dataloader_test,
                               query_dataloader=query_dataloader_test_180, 
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True,
                               limit_fov=180)
            r1_test_fov90 = evaluate(config=args,
                               model=model,
                               reference_dataloader=reference_dataloader_test,
                               query_dataloader=query_dataloader_test_90, 
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True,
                               limit_fov=90)
            r1_test_fov70 = evaluate(config=args,
                               model=model,
                               reference_dataloader=reference_dataloader_test,
                               query_dataloader=query_dataloader_test_70, 
                               ranks=[1, 5, 10],
                               step_size=1000,
                               cleanup=True,
                               limit_fov=70)
            if args.sim_sample:
                r1_train, sim_dict = calc_sim(config=args,
                                              model=model,
                                              reference_dataloader=reference_dataloader_train,
                                              query_dataloader=query_dataloader_train, 
                                              ranks=[1, 5, 10],
                                              step_size=1000,
                                              cleanup=True)
                
            

            if torch.cuda.device_count() > 1 and len(args.gpu_ids) > 1:
                torch.save(model.module.state_dict(), '{}/weights_e{}_360know:{:.4f}_360:{:.4f}_360:{:.4f}_180:{:.4f}_90:{:.4f}_70:{:.4f}.pth'.format(model_path, epoch, r1_test_fov360_know, r1_test_fov360,r1_test_fov270,r1_test_fov180,r1_test_fov90,r1_test_fov70))

            else:
                torch.save(model.state_dict(), '{}/weights_e{}_360know:{:.4f}_360:{:.4f}_360:{:.4f}_180:{:.4f}_90:{:.4f}_70:{:.4f}.pth'.format(model_path, epoch,  r1_test_fov360_know, r1_test_fov360,r1_test_fov270,r1_test_fov180,r1_test_fov90,r1_test_fov70))
        
        if args.custom_sampling:
            train_dataloader.dataset.shuffle(sim_dict,
                                             neighbour_select=args.neighbour_select,
                                             neighbour_range=args.neighbour_range)
    

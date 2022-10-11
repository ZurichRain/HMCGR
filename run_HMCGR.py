import sys
import os
import config as config
import logging
import pickle
import numpy as np
import json

from dataset.HMCGR_dataset1 import HMCGRDataset
from model.HMCGR1 import HMCGR
from eval_test.eval_HMCGR import eval_link

from train_script import train, evaluate

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup 
import torch.optim as optim
from util_script.optimizer_wf import RAdam
import sys
import warnings
from torch.utils.tensorboard import SummaryWriter
import shutil
import random
import torch
from util_script.metrics import f1_score_3

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.INFO)


MODEL_CLASSES={
    'HMCGR_joint_three_link' :  HMCGR,
}
MODEL_DATASET={
    'HMCGR_joint_three_link' :  HMCGRDataset,    
}
MODEL_EVAL={
    'HMCGR_joint_three_link' :  eval_link,
}
def seed_everything(seed=1226):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True 

def read_data(dirname):
    all_data = []
    for file in os.listdir(dirname):
        if(file.split('.')[-1] != 'jsonl'):
            continue
        doc_seq_data = []
        with open(os.path.join(dirname,file),'r') as f:
            for lin in f.readlines():
                doc_seq_data.append(json.loads(lin.strip()))
        all_data+=doc_seq_data
    return all_data

def run():
    """train the model"""
    seed_everything()
    logging.info("device: {}".format(config.device))
    CurDataset = MODEL_DATASET[config.model_type]
    model:torch.nn.Module = MODEL_CLASSES[config.model_type]()
    
    CurEvallink = MODEL_EVAL[config.model_type]
    
    train_data_lis = read_data(config.train_data_dir)
    vail_data_lis = read_data(config.vail_data_dir)
    test_data_lis = read_data(config.test_data_dir)

    logging.info("--------Process Done!--------")
    

    train_dataset = CurDataset(train_data_lis)
    dev_dataset = CurDataset(vail_data_lis)
    test_dataset = CurDataset(test_data_lis)
    logging.info("--------Dataset Build!--------")
    # get dataset size
    train_size = len(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
 
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=dev_dataset.collate_fn)

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=test_dataset.collate_fn)
    logging.info("--------Get Dataloader!--------")
    # Prepare model
    device = config.device
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    train_steps_per_epoch = train_size // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=train_steps_per_epoch,
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)

    # Train the model
    logging.info("--------Start Training!--------")
    if(not os.path.exists(config.train_log_dir)):
        os.makedirs(config.train_log_dir)
    else :
        shutil.rmtree(config.train_log_dir)
    if(not os.path.exists(config.test_log_dir)):
        os.makedirs(config.test_log_dir)
    else:
        shutil.rmtree(config.test_log_dir)
    train_writer = SummaryWriter(log_dir=config.train_log_dir)
    test_writer = SummaryWriter(log_dir=config.test_log_dir)
   
    if config.do_train:
        
        train(model, optimizer, train_loader,eval_fun=f1_score_3,dev_loader=dev_loader,\
            scheduler=scheduler,train_writer=train_writer,test_writer=test_writer)
        

    if config.do_test:
        
        CurEvallink(test_loader,eval_fun=f1_score_3)

if __name__ == '__main__':
    run()
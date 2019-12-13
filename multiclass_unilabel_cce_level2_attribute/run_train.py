#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:00:47 2019

@author: amrita
"""
import os
from options import get_options
from datasets import get_dataloader, get_dataset, get_catalog
from trainer import get_trainer
from model import get_model
opt = get_options('train')
worker_number = opt.worker_number
train_catalog = get_catalog(opt, 'train')
print (type(train_catalog))
start = 10*(worker_number-1)
end = 10*worker_number
dump_checkpoint_path = opt.dump_checkpoint_path
load_checkpoint_path = opt.load_checkpoint_path
for cluster_id in range(start,end):
    opt.dump_checkpoint_path = os.path.join(dump_checkpoint_path, str(cluster_id))
    opt.load_checkpoint_path = os.path.join(load_checkpoint_path, str(cluster_id))
    if cluster_id not in train_catalog.clusters_inv:
       continue
    print ('Starting cluster (', cluster_id, '):  ', train_catalog.clusters_inv[cluster_id])
    train_dataset = get_dataset(train_catalog, cluster_id)
    train_loader = get_dataloader(train_dataset, opt)
    vocab_size = train_dataset.vocab_size
    print ('number of classes ', vocab_size)
    if vocab_size==1:
       print ('Number of classes ==  1, hence skipping this cluster')
       continue
    model = get_model(opt, vocab_size)
    trainer = get_trainer(opt, model, train_loader)
    trainer.train()













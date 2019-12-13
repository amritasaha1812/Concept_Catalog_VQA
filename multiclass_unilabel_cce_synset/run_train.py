#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:00:47 2019

@author: amrita
"""

from options import get_options
from datasets import get_dataloader, get_dataset
from trainer import get_trainer
from model import get_model

opt = get_options('train')
train_dataset = get_dataset(opt, 'train')
train_loader = get_dataloader(train_dataset, opt)
#val_dataset = get_dataset(opt, 'val')
print ('got train and valid dataset')
vocab_size = train_dataset.vocab_size
cluster_size = train_dataset.cluster_size
clusters = train_dataset.clusters_np
print ('number of classes ', train_dataset.get_num_classes())
print ('vocab size', train_dataset.vocab_size)
print ('cluster size', train_dataset.cluster_size)
model = get_model(opt, cluster_size, vocab_size, clusters)
print ('built model')
trainer = get_trainer(opt, model, train_loader)
trainer.train()







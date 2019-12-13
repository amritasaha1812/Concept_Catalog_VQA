#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:17:39 2019

@author: amrita
"""
from torch.utils.data import DataLoader
from .visual_genome_attribute import VisualGenomeAttributeDataset

def get_dataset(opt, split):
    if opt.dataset == 'visualgenome':
        ds = VisualGenomeAttributeDataset(split, opt.image_dir, opt.preprocessed_data_dir, opt.preprocessed_pos_data, opt.dump_data_path,
                   opt.vocab_file, opt.cluster_file, opt.glove_embedding_file, opt.image_concepts_glove_emb_file, opt.gpu_ids, opt.cluster_classify, opt.shuffle_data, opt.sort_by)
    return ds

def get_dataloader(ds, opt):
    loader = DataLoader(dataset=ds, batch_size=opt.batch_size, shuffle=opt.shuffle_data)
    return loader        
            







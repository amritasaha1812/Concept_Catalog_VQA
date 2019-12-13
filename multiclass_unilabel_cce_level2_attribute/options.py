#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:22:42 2019

@author: amrita
"""

import argparse
import os
import utils
import torch

class BaseOptions():
    
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        
    def initialize(self):
        self.parser.add_argument('--run_dir', default='/dccstor/cssblr/amrita/Concept_Catalog_VQA/multiclass_unilabel_cce_level2_attribute', type=str, help='experiment directory')
        self.parser.add_argument('--image_dir', default='/dccstor/cssblr/amrita/VisualGenome/data/VG_100K', type=str, help='directory containing VisualGenome images')
        self.parser.add_argument('--dataset', default='visualgenome', type=str, help='dataset')
        self.parser.add_argument('--load_checkpoint_path', default=None, type=str, help='load checkpoint path')
        self.parser.add_argument('--dump_checkpoint_path', default=None, type=str, help='dump checkpoint path')
        self.parser.add_argument('--dump_data_path', default=None, type=str, help='dump data path')
        self.parser.add_argument('--gpu_ids', default='0', type=str, help='ids of gpu to be used')
        self.parser.add_argument('--image_region_resnet_features', default='region_resnet_features_filtered_datasize100.h5', type=str, help='file name consisting of the precomputed resnet features of the image regions')
        self.parser.add_argument('--image_region_dictionary', default='region_keys_filtered_datasize100.pkl', type=str, help='file containing image regions for which precomputed resnet features are available')
        self.parser.add_argument('--glove_embedding_file', default='attribute_glove_emb_filtered_datasize100.pkl', type=str, help='file containing glove embedding of class labels')
        self.parser.add_argument('--image_concepts_glove_emb_file', default='image_concepts_glove_embedding_filtered_datasize100.pkl', type=str, help='file containing glove embedding of image concepts')
        self.parser.add_argument('--vocab_file', default='manual_gvqa_glove_wordnet_concepts_final.json', type=str, help='file containing clusters of attributes')
        self.parser.add_argument('--cluster_file', default='manual_gvqa_glove_wordnet_concept_clusters_final.json', type=str, help='file containing clusters of attributes')
        self.parser.add_argument('--seed', default=1, type=int, help='random seed')
        self.parser.add_argument('--batch_size', default=200, type=int, help='batch size')
        self.parser.add_argument('--learning_rate', default=5e-3, type=float, help='learning rate')
        self.parser.add_argument('--input_channels', default=3, type=int, help='number of input channels for the image')
        self.parser.add_argument('--image_feature_size', default=512, type=int, help='feature size of the image')
        self.parser.add_argument('--hidden_size', default=512, type=int, help='hidden dimension')
        self.parser.add_argument('--common_embedding_size', default=512, type=int, help='common dimension of the image and concept')
        self.parser.add_argument('--glove_embedding_size', default=100, type=int, help='glove embedding dimension of the concept')
        self.parser.add_argument('--fc1_size', default=1000, type=int, help='dimension of the output fc1 layer')
        self.parser.add_argument('--fc2_size', default=1000, type=int, help='dimension of the output fc2 layer')
        self.parser.add_argument('--num_att_layers', default=2, type=int, help='number of stacked attention layers')
        self.parser.add_argument('--worker_number', default=None, type=int, help='worker number indicates which split of the data to work on')
        self.initialized = True
        
    def parse(self):
        if not self.initialized:
            self.initialize()
            
        self.opt = self.parser.parse_args()
        
        str_gpu_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_gpu_ids:
            if str_id.isdigit() and int(str_id) >= 0:
                self.opt.gpu_ids.append(int(str_id))
        if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(self.opt.gpu_ids[0])
        else:
            print('| using cpu')
            self.opt.gpu_ids = []
        # print and save options
        args = vars(self.opt)
        print('| options')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        if not os.path.exists(self.opt.run_dir):
            os.mkdir(self.opt.run_dir)

        if self.is_train:
            filename = 'train_opt.txt'
        else:
            filename = 'test_opt.txt'
        file_path = os.path.join(self.opt.run_dir, filename)
        with open(file_path, 'wt') as fout:
            fout.write('| options\n')
            for k, v in sorted(args.items()):
                fout.write('%s: %s\n' % (str(k), str(v)))

        return self.opt


class TrainOptions(BaseOptions):
    
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--preprocessed_data_dir', default='/dccstor/cssblr/amrita/VisualGenome/data/preprocessed', type=str, help='directory consisting of the preprocessed data for training')
        self.parser.add_argument('--preprocessed_pos_data', default='attribute_regions_filtered_datasize100.pkl', type=str, help='file name consisting of the preprocessed positive data for training')
        self.parser.add_argument('--preprocessed_neg_data', default='attribute_negative_regions_filtered_datasize100.pkl', type=str, help='file name consisting of the preprocessed negative data for training')
        self.parser.add_argument('--max_iters', default=100000, type=int, help='maximum number of iterations/batches')
        self.parser.add_argument('--max_epochs', default=10, type=int, help='maximum number of epochs')
        self.parser.add_argument('--display_every', default=1, type=int, help='display training information every N batchs')
        self.parser.add_argument('--checkpoint_every', default=2000, type=int, help='save every N batches')
        self.parser.add_argument('--weighted_random_sampler', default=1, type=int, help='class balanced weighted random sampler')
        self.parser.add_argument('--shuffle_data', default=1, type=int, help='shuffle dataset')
        self.parser.add_argument('--sort_by', default='none', type=str, help='sort by synset/attribute/cluster/none')
        self.parser.add_argument('--cluster_classify', default=0, type=int, help='classify the label cluster instead of the label directly')
        self.is_train = True 
            

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--preprocessed_data_dir', default='/dccstor/cssblr/amrita/VisualGenome/data/preprocessed', type=str, help='directory name where preprocessed validation')
        self.parser.add_argument('--preprocessed_pos_data_file', default='attribute_regions_filtered_datasize100.pkl', type=str, help='file name consisting of the preprocessed positive data for validation')
        self.parser.add_argument('--preprocessed_neg_data_file', default='attribute_negative_regions_filtered_datasize100.pkl', type=str, help='file name consisting of the preprocessed negative data for validation')
        self.parser.add_argument('--split', default='val')
        self.parser.add_argument('--output_path', default='results.json', type=str, help='filename for dumping results')
        self.is_train = False
        
        
def get_options(mode):
    if mode == 'train':
        opt = TrainOptions().parse()
    elif mode == 'test':
        opt = TestOptions().parse()
    else:
        raise ValueError('Invalid mode for option parsing: %s ', mode)
    return opt
    
            




#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:06:16 2019

@author: amrita
"""
import torch
import numpy as np
import time
import random
import os

class Trainer:
    
    def __init__(self, opt, model, train_loader, val_loader=None):
        self.cluster_classify = opt.cluster_classify
        self.max_iters = opt.max_iters
        self.run_dir = opt.run_dir
        self.save_dir = os.path.join(opt.run_dir, os.path.join('checkpoints',opt.dump_checkpoint_path))
        print ('save dir ', self.save_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.display_every = opt.display_every
        self.checkpoint_every = opt.checkpoint_every
        self.iter = 0
        self.max_iters = opt.max_iters
        self.max_epochs = opt.max_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = None
        self.model = model
        if len(opt.gpu_ids)==0:
            self.device = 'cpu'
        else:
            self.device = 'cuda'
            torch.backends.cudnn.benchmark = True
        self.stats = {
                'train_losses': [],
                'train_losses_ts': [],
                'val_losses': [],
                'val_losses_ts': [],
                'best_val_loss': 9999,
                'model_t': 0
        }
        self.data_distribution_specific_sampling = True
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
    

    def train_epoch(self, epoch):
        i = 0
        data_time = 0.
        epoch_loss = 0.
        epoch_score = 0.
        for train_data_batch, labels_batch, context_word_glove_emb, context_attention_vector in self.train_loader:#.get_batch_posneg_data(self.data_distribution_specific_sampling)            
            if train_data_batch is None or labels_batch is None:
                break
            start = time.time()
            #print (np.max(labels_batch.numpy()))
            self.model.set_input(context_word_glove_emb, context_attention_vector, train_data_batch, labels_batch) 
            self.model.step()
            loss = self.model.get_loss()
            score_mean = self.model.compute_score()
            loss_mean = np.mean(loss.cpu().data.numpy())
            if self.iter % self.display_every == 0:
                #self.stats['train_losses'].append(loss_mean)
                #print ('iteration: %d, epoch: %d, loss_mean: %f score_mean: %f' % (self.iter,  epoch, loss_mean, score_mean))
                self.stats['train_losses_ts'].append(self.iter)
            if self.iter % self.checkpoint_every == 0 or self.iter >= self.max_iters:    
                if self.val_loader is not None:
                    print ('checking validation loss')
                    val_loss = self.check_val_loss()
                    print ('validation loss %f' % val_loss)
                    if val_loss <= self.stats['best_val_loss']:
                        print ('best model')
                        self.stats['best_val_loss'] = val_loss
                        self.stats['model_t'] = self.iter
                        self.model.save_checkpoint('%s/checkpoint_best.pt' % self.save_dir)
                    self.stats['val_losses'].append(val_loss)
                    self.stats['val_losses_ts'].append(self.iter)
                else:
                    self.stats['model_t'] = self.iter
                    self.model.save_checkpoint('%s/checkpoint_best.pt' % self.save_dir)
            #with open('%s/stats.json' % self.run_dir, 'w') as fout:
            #    print ('stats',self.stats)
            #    json.dump(self.stats, fout, indent=1)
            end = time.time()
            data_time += (end-start)
            epoch_loss += loss_mean
            epoch_score += score_mean
            i+=1
            #if self.iter % self.display_every==0 and self.iter>0:
            print ('Epoch ', epoch, 'Batch No ', i, ' Avg Loss (till now) ', epoch_loss/float(i), ' Avg Score (till now) ', epoch_score/float(i),' ( Time taken ', float(data_time)/float(i), 'secs per batch )')
            self.iter+=1
        return epoch_loss/float(i), epoch_score/float(i)
        
    
    def train(self):
        print ('Starting ...')    
        last_epoch = 0
        print ('Starting Training ...')
        for epoch in range(last_epoch, self.max_epochs):
            if self.iter >= self.max_iters:
                break
            epoch_loss, epoch_score = self.train_epoch(epoch)
            print ('Epoch  ', epoch, ' Loss ', epoch_loss, ' Score ', epoch_score)
            
def get_trainer(opt, model, train_loader, val_loader=None):
    return Trainer(opt, model, train_loader, val_loader)
            

    





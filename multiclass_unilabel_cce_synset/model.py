#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:50:14 2019

@author: amrita
"""
import numpy 
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
import os

class ObjectCatalog():
    
    def __init__(self, opt, cluster_size, vocab_size, clusters):
        self.net = _Net(opt, cluster_size, vocab_size, clusters)
        self.clusters = clusters
        self.cluster_size = cluster_size
        if opt.load_checkpoint_path:
            try:
                print('loading checkpoint from %s', os.path.join('checkpoints',opt.load_checkpoint_path))
                checkpoint = torch.load(os.path.join('checkpoints',os.path.join(opt.load_checkpoint_path, 'checkpoint_best.pt')))
                self.net.load_state_dict(checkpoint['model_state'])
                print ('Model loaded')
            except:
                print ('Model initialized... Not loaded')
        self.num_classes = vocab_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.learning_rate)
        self.use_cuda = len(opt.gpu_ids) > 0 and torch.cuda.is_available()
        self.gpu_ids = opt.gpu_ids
        if self.use_cuda:
            self.net.cuda(opt.gpu_ids[0])
            print ('transferred model to gpu')
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.input, self.label = None, None
        
    def _to_var(self, x):    
        if self.use_cuda:
            x = x.cuda()
        return Variable(x)
    
    def set_input(self, context_word_glove_emb, context_attention_vector, x, y=None):
        self.input = self._to_var(x)
        self.context_word_glove_emb = self._to_var(context_word_glove_emb)
        self.context_attention_vector = self._to_var(context_attention_vector)
        if y is not None:
            self.label = self._to_var(y)
            #self.label_onehot = one_hot(self.label, self.label.size(0), self.num_classes, self.device).long()
            
   
    def forward(self):
        self.pred = self.net(self.input, self.context_word_glove_emb, self.context_attention_vector)
        if self.label:
            self.loss = self.criterion(self.pred, self.label)

    def get_loss(self):
        return self.loss.data
    
    def compute_score(self):
        logits = torch.max(self.pred, 1)[1].cpu().numpy()
        label = self.label.cpu().numpy()
        score = (logits==label).astype(np.int).sum() / label.shape[0]
        return score
   
    def compute_recall(self, k):
        topk_scores, topk_ind = torch.topk(self.pred, k)
        #topk_scores and topk_ind is of dimension batch_size x k
        topk_ind = topk_ind.cpu().numpy()
        topk_ind_onehot = np.sum((np.arange(self.num_classes) == topk_ind[...,None]-1).astype(float), axis=1)
        label_onehot = np.sum((np.arange(self.num_classes) == np.expand_dims(self.label.cpu().numpy(), axis=1)[...,None]-1).astype(float), axis=1)
        score = np.sum(numpy.logical_and(numpy.equal(topk_ind_onehot, label_onehot), numpy.equal(label_onehot, np.ones_like(label_onehot))).astype(np.float), axis=1)
        rec = np.sum(score) / label_onehot.shape[0]
        return rec
 
    def compute_clustering_score(self):
        logits = torch.max(self.pred, 1)[1].cpu().numpy()
        
        label = self.label.cpu().numpy()
        cluster_label = self.clusters[label]
        logits_cluster_label = self.clusters[logits]
        score = (cluster_label == logits_cluster_label).astype(np.float).sum() / label.shape[0]
        return score

    def compute_clustering_recall(self, k):
        topk_scores, topk_ind = torch.topk(self.pred, k)
        topk_ind = topk_ind.cpu().numpy()
        topk_cluster_ind = self.clusters[topk_ind]
        topk_cluster_ind_onehot = np.max((np.arange(self.cluster_size) == topk_cluster_ind[...,None]-1).astype(float), axis=1)
        cluster_label = self.clusters[self.label.cpu().numpy()]
        cluster_label_onehot = np.sum((np.arange(self.cluster_size) == np.expand_dims(cluster_label, axis=1)[...,None]-1).astype(float), axis=1)
        score = np.sum(numpy.logical_and(numpy.equal(topk_cluster_ind_onehot, cluster_label_onehot), numpy.equal(cluster_label_onehot, np.ones_like(cluster_label_onehot))).astype(np.float), axis=1)
        rec = np.sum(score) / cluster_label.shape[0]
        return rec
    
    def step(self):
        self.optimizer.zero_grad()
        self.forward()
        self.loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.25)
        self.optimizer.step()
        return
        
    def get_pred(self):
        return self.pred.data.cpu().numpy()
    
    def eval_mode(self):
        self.net.eval()
        
    def train_mode(self):
        self.net.train()
        
    def save_checkpoint(self, save_path):
        checkpoint = {
            'model_state': self.net.cpu().state_dict()
        }
        torch.save(checkpoint, save_path)
        if self.use_cuda:
            self.net.cuda(self.gpu_ids[0])
            

class _Net(nn.Module):
    
    def __init__(self, opt, cluster_size, vocab_size, attention_weight_mat):
        super(_Net, self).__init__()
        self.use_cuda = len(opt.gpu_ids) > 0 and torch.cuda.is_available()
        self.gpu_ids = opt.gpu_ids
        if self.use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.input_channels = opt.input_channels
        self.cluster_size = cluster_size
        self.img_feat_size = opt.image_feature_size
        self.hidden_size = opt.hidden_size
        self.cluster_classify = opt.cluster_classify
        self.common_emb_size = opt.common_embedding_size
        self.glove_emb_size = opt.glove_embedding_size
        self.fc1_size = opt.fc1_size
        self.fc2_size = opt.fc2_size
        self.num_att_layers = opt.num_att_layers
        self.tanh = nn.Tanh
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)
        self.vocab_size = vocab_size
        resnet = models.resnet34(pretrained=True)
        resnet_layers = list(resnet.children())
        #for layer in resnet_layers:
        #    for param in layer.parameters():
        #        param.requires_grad = False
        # remove the last two layer
        resnet_layers.pop()
        resnet_layers.pop()
        # remove the first layer as we take a 6-channel input
        resnet_layers.pop(0)
        resnet_layers.insert(0, nn.Conv2d(self.input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
        self.resnet_layers_seq = nn.Sequential(*resnet_layers)
        self.image_pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.concept_image_att_mat = nn.Linear(self.glove_emb_size, self.img_feat_size)
        layers1 = []
        layers1.append(nn.Linear(self.img_feat_size, self.hidden_size))
        layers1.append(nn.LeakyReLU())
        layers1.append(nn.Dropout(0.2))
        layers2 = []
        layers2.append(nn.Linear(self.glove_emb_size, self.hidden_size))
        layers2.append(nn.LeakyReLU())
        layers2.append(nn.Dropout(0.2))
        self.layers1_seq = nn.Sequential(*layers1)
        self.layers2_seq = nn.Sequential(*layers2)
        self.img_linear_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.common_emb_size) for i in range(self.num_att_layers)])
        self.concept_linear_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.common_emb_size) for i in range(self.num_att_layers)])
        self.img_concept_linear_layers = nn.ModuleList([nn.Linear(self.common_emb_size, 1) for i in range(self.num_att_layers)])
        final_layers = []
        final_layers.append(nn.Dropout(0.1))
        final_layers.append(nn.Linear(self.hidden_size, self.fc1_size))
        final_layers.append(nn.LeakyReLU())
        final_layers.append(nn.Dropout(0.1))
        final_layers.append(nn.Linear(self.fc1_size, self.fc2_size))
        final_layers.append(nn.LeakyReLU())
        final_layers.append(nn.Dropout(0.1))
        final_layers.append(nn.Linear(self.fc2_size, self.vocab_size))
        final_layers.append(nn.LeakyReLU())
        final_layers.append(nn.Dropout(0.1))
        #final_layers.append(nn.Sigmoid())
        self.final_layers_seq = nn.Sequential(*final_layers)
        
    def forward(self, img, context_glove_embed, context_attention_vector):
        img_emb = self.resnet_layers_seq(img)
        img_emb_pooled = torch.unsqueeze(torch.squeeze(self.image_pooling_layer(img_emb)), dim=1)
        #img_emb_pooled is of dimension batch_size x 1 x img_feat_size
        concept_emb = self.concept_image_att_mat(context_glove_embed)
        #concept_emb is of dimension batch_size x vocab_size x img_feat_size
        concept_emb = torch.transpose(concept_emb, 2, 1)
        #concept_emb is of dimension batch_size x img_feat_size x vocab_size)
        img_concept_emb = torch.squeeze(torch.bmm(img_emb_pooled, concept_emb))
        #img_concept_emb is of dimension batch_size x vocab_size
        img_concept_att = self.softmax(img_concept_emb)
        img_concept_att = torch.unsqueeze(self.mask_attention(img_concept_att, context_attention_vector), dim=1)
        #img_concept_att is of dimension batch_size x 1 x vocab_size
        img_concept_emb = torch.squeeze(torch.bmm(img_concept_att, context_glove_embed))
        #img_concept_emb is of dimension batch_size x glove_emb_size
        img_emb = torch.transpose(torch.transpose(img_emb, 2, 1), 3, 2).view(-1, 64, self.img_feat_size)
        img_hid = self.layers1_seq(img_emb).view(-1, 64, self.hidden_size)
        #imd_hid is of dimension batch_size x 64 x img_hid_size
        img_concept_hid = self.layers2_seq(img_concept_emb)
        #img_concept_hid is of dimension batch_size x concept_hid_size
        i=0
        for i in range(self.num_att_layers):
            img_common = self.img_linear_layers[i](img_hid)
            #img_common is of dimension batch_size x 64 x common_emb_size
            concept_common = self.concept_linear_layers[i](img_concept_hid)
            #concept_common is of dimension batch_size x common_emb_size
            concept_common = torch.unsqueeze(concept_common, dim=1).repeat(1, 64, 1)
            #concept_common is of dimension batch_size x 64 x common_emb_size
            img_concept_common = self.dropout((img_common+concept_common).tanh())
            #img_concept_common is of dimension batch_size x 64 x common_emb_size
            h = torch.squeeze(self.img_concept_linear_layers[i](img_concept_common), dim=2)
            #h is of dimension batch_size x 64
            p_att = torch.unsqueeze(self.softmax(h), dim=1)
            #p_att is of dimension batch_size x 1 x 64
            img_att = torch.squeeze(torch.bmm(p_att, img_hid), dim=1)
            #img_att is of dimension batch_size x img_hid_size
            if i==(self.num_att_layers-1):
                img_concept_hid = img_att
                #concept_hid is of dimension batch_size x img_hid_size
            else:
                img_concept_hid = img_att+img_concept_hid
                #concept_hid is of dimension batch_size x img_hid_size
        clusters = self.final_layers_seq(img_concept_hid)
        #clusters is of dimension batch_size x cluster_size
        return clusters
    
    def mask_attention(self, attention, mask):
        masked_atten = torch.mul(attention, mask)
        num = len(masked_atten.shape)
        l1norm = torch.sum(masked_atten, dim=1)
        stacked_norm = torch.mul(torch.ones_like(masked_atten), torch.unsqueeze(l1norm,num-1))
        masked_atten = torch.where(stacked_norm==0, torch.ones_like(masked_atten), masked_atten)
        new_l1_norm = torch.sum(masked_atten, dim=1)
        masked_atten = masked_atten/new_l1_norm.view([-1,1])
        return masked_atten                         

def get_model(opt, cluster_size, vocab_size, clusters):
    model = ObjectCatalog(opt, cluster_size, vocab_size, clusters)
    return model

    






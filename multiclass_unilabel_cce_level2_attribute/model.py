#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:50:14 2019

@author: amrita
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import one_hot
import numpy as np
import torchvision.models as models
import os

class AttributeCatalog():
    
    def __init__(self, opt, vocab_size):
        self.net = _Net(opt, vocab_size)
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
        self.input, self.label, self.label_onehot = None, None, None
        
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
            self.label_onehot = one_hot(self.label, self.label.size(0), self.num_classes, self.device).long()
            
   
    def forward(self):
        self.pred = self.net(self.input, self.context_word_glove_emb, self.context_attention_vector)
        if self.label_onehot is not None:
            self.loss = self.criterion(self.pred, self.label)

    def get_loss(self):
        return self.loss.data
    
    def compute_score(self):
        logits = torch.max(self.pred, 1)[1].cpu().numpy()
        label = self.label.cpu().numpy()
        score = (logits==label).astype(np.int).sum() / label.shape[0]
        return score
    
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
            
'''
class _Net(nn.Module):

    def __init__(self, hidden_dim):
        super(_Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = 2
        
        self.leaky_relu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        
        layers = []
        layers.append(nn.Linear(2048, self.hidden_dim))
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.main = nn.Sequential(*layers)
        self.final_layer = nn.Linear(self.hidden_dim, self.num_classes)
        
    def forward(self, x):
        x = self.main(x)
        output = self.final_layer(x)
        out_dist = F.softmax(output, dim=1)
        return out_dist
'''


class _Net(nn.Module):
    
    def __init__(self, opt, vocab_size):
        super(_Net, self).__init__()
        self.use_cuda = len(opt.gpu_ids) > 0 and torch.cuda.is_available()
        self.gpu_ids = opt.gpu_ids
        if self.use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.input_channels = opt.input_channels
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
        layers1.append(nn.Dropout(0.1))
        layers2 = []
        layers2.append(nn.Linear(self.glove_emb_size, self.hidden_size))
        layers2.append(nn.LeakyReLU())
        layers2.append(nn.Dropout(0.1))
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
        '''
        if not self.cluster_classify:
            self.cluster_embedding = nn.Parameter(torch.tensor(cluster_embedding_mat), requires_grad=True)
            layers1_2 = []
            layers1_2.append(nn.Linear(self.img_feat_size, self.hidden_size))
            layers1_2.append(nn.LeakyReLU())
            layers1_2.append(nn.Dropout(0.2))
            self.layers1_2_seq = nn.Sequential(*layers1_2)
            layers2_2 = []
            layers2_2.append(nn.Linear(self.glove_emb_size, self.hidden_size))
            layers2_2.append(nn.LeakyReLU())
            layers2_2.append(nn.Dropout(0.2))
            self.layers2_2_seq = nn.Sequential(*layers2_2)
            self.img_linear_layers2 = nn.ModuleList([nn.Linear(self.hidden_size, self.common_emb_size) for i in range(self.num_att_layers)])
            self.concept_linear_layers2 = nn.ModuleList([nn.Linear(self.hidden_size, self.common_emb_size) for i in range(self.num_att_layers)])
            self.img_concept_linear_layers2 = nn.ModuleList([nn.Linear(self.common_emb_size, 1) for i in range(self.num_att_layers)])
            final_layers2 = []
            final_layers2.append(nn.Dropout(0.1))
            final_layers2.append(nn.Linear(self.hidden_size, self.fc1_size))
            final_layers2.append(nn.LeakyReLU())
            final_layers2.append(nn.Dropout(0.1))
            final_layers2.append(nn.Linear(self.fc1_size, self.fc2_size))
            final_layers2.append(nn.LeakyReLU())
            final_layers2.append(nn.Dropout(0.1))
            final_layers2.append(nn.Linear(self.fc2_size, self.vocab_size))
            final_layers2.append(nn.LeakyReLU())
            final_layers2.append(nn.Dropout(0.1))
            self.final_layers2_seq = nn.Sequential(*final_layers2)
        print ('initialized _Net model')
        self.attention_weight_mat = torch.tensor(attention_weight_mat, requires_grad=False, device=self.device, dtype=torch.long)
        '''
        print ('initialized _Net model')
        
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
# =============================================================================
#         if self.cluster_classify:
#             return clusters     
#         else:
#             clusters_att = self.softmax(clusters)
#             #clusters_att is of dimension batch_size x cluster_size
#             clusters_emb = torch.mm(clusters_att, self.cluster_embedding)
#             #clusters_hid is of dimension batch_size x glove_emb_size
#             img_hid2 = self.layers1_2_seq(img_emb).view(-1, 64, self.hidden_size)
#             concept_hid2 = self.layers2_2_seq(clusters_emb)
#             i=0
#             for i in range(self.num_att_layers):
#                 img_common2 = self.img_linear_layers2[i](img_hid2)
#                 concept_common2 = self.concept_linear_layers2[i](concept_hid2)
#                 concept_common2 = torch.unsqueeze(concept_common2, dim=1).repeat(1, 64, 1)
#                 img_concept_common2 = self.dropout((img_common2+concept_common2).tanh())
#                 h2 = torch.squeeze(self.img_concept_linear_layers2[i](img_concept_common2), dim=2)
#                 p_att2 = torch.unsqueeze(self.softmax(h2), dim=1)
#                 img_att2 = torch.squeeze(torch.bmm(p_att2, img_hid2), dim=1)
#                 if i==(self.num_att_layers-1):
#                     concept_hid2 = img_att2
#                 else:
#                     concept_hid2 = img_att2 + concept_hid2
#             concepts = self.final_layers2_seq(concept_hid2)   
#             concepts_att = self.softmax(concepts)
#             #concepts_att is of dimension batch_size x vocab_size
#             
#             att_weight = torch.unsqueeze(self.attention_weight_mat, dim=0).repeat(concepts_att.size(0), 1)
#             #att_weight is of dimension batch_size x vocab_size
#             att_weight = torch.gather(clusters, dim=1, index=att_weight)
#             #att_weight is of dimension batch_size x vocab_size
#             concepts_att = torch.mul(concepts_att, att_weight)
#             return concepts_att
# =============================================================================
   
    def mask_attention(self, attention, mask):
        masked_atten = torch.mul(attention, mask)
        num = len(masked_atten.shape)
        l1norm = torch.sum(masked_atten, dim=1)
        stacked_norm = torch.mul(torch.ones_like(masked_atten), torch.unsqueeze(l1norm,num-1))
        masked_atten = torch.where(stacked_norm==0, torch.ones_like(masked_atten), masked_atten)
        new_l1_norm = torch.sum(masked_atten, dim=1)
        masked_atten = masked_atten/new_l1_norm.view([-1,1])
        return masked_atten                         

def get_model(opt, vocab_size):
    model = AttributeCatalog(opt, vocab_size)
    return model

    






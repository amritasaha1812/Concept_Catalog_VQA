#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:04:01 2019

@author: amrita
"""
import time
import numpy as np
import pickle as pkl
import os
import random
import sys
import math
import PIL.Image as Image
import cv2
from pattern.en import pluralize, singularize, lemma
sys.path.append('../..')
from nltk.corpus import stopwords
import json

class VisualGenomeObjectDataset():
    def __init__(self, split, image_dir, preprocessed_data_dir, preprocessed_data_file, dump_data_path, vocab_file, cluster_file,
                 attributes_glove_emb_file, image_concepts_glove_emb_file, gpu_ids, cluster_classify, shuffle_data, sort_by):
        self.split = split
        self.sort_by = sort_by
        self.data_dir = preprocessed_data_dir
        self.dump_data_path = dump_data_path
        if not os.path.exists(os.path.join('preprocessed_data',self.dump_data_path)):
            os.mkdir(os.path.join('preprocessed_data',self.dump_data_path))
        self.cluster_classify = cluster_classify
        self.shuffle_data = shuffle_data
        if self.sort_by!="none" and self.sort_by!="cluster" and self.shuffle_data==1:
            raise Exception('sort_by option has to be None if shuffle data is on')
        self.image_dir = image_dir
        self.desired_size = 256
        self.stopwords = set(stopwords.words('english'))
        data_raw = None
        if preprocessed_data_file:
           for k,v in pkl.load(open(os.path.join(self.data_dir, preprocessed_data_file), 'rb'), encoding='latin1').items():
              k = '.'.join(k.split('.')[:-2])
              if k not in self.stopwords:
                  if k in data_raw:
                      for vi in v:
                         if vi in data_raw[k]:
                               data_raw[k][vi].extend(v[vi])
                         else:
                               data_raw[k][vi] = v[vi]
                  else:
                      data_raw[k] = v
        self.vocab = self.get_vocab(vocab_file, data_raw)
        self.clusters, self.clusters_inv = self.get_clusters(cluster_file, data_raw)
        if image_concepts_glove_emb_file:
            self.image_concepts_glove_emb = pkl.load(open(os.path.join(self.data_dir, image_concepts_glove_emb_file), 'rb'), encoding='latin1')
        else:
            self.image_concepts_glove_emb = None
        self.attributes_glove_emb = np.zeros((len(self.vocab), 100), dtype=np.float32)
        if attributes_glove_emb_file:
            for k,v in pkl.load(open(os.path.join(self.data_dir, attributes_glove_emb_file), 'rb'), encoding='latin1').items():
                k = '.'.join(k.split('.')[:-2])
                if k not in self.vocab:
                     continue
                k_id = self.vocab[k]
                self.attributes_glove_emb[k_id] = self.attributes_glove_emb[k_id] + v
        else:
           self.attributes_glove_emb = None
        #self.attributes_glove_emb = np.load(os.path.join(self.data_dir, attributes_glove_emb_file))
        self.vocab_size = len(set((self.vocab.values())))
        self.clusters_np = np.zeros((self.vocab_size), dtype=np.int32)
        self.cluster_size = len(self.clusters_inv)
        for k,v in self.clusters.items():
            self.clusters_np[self.vocab[k]] = int(v)
        start = time.time()
        self.maxlen = 0
        if data_raw:
            self.data, self.labels = self.preprocess_data(data_raw)
        if len(gpu_ids)==0:
            self.device = 'cpu'
        else:
            self.device = 'cuda'
        end = time.time()
        print ('Preprocessed data in ', (end-start), 'secs')
    
    
    def get_vocab(self, vocab_file, data_raw):
        if os.path.exists(os.path.join(os.path.join('preprocessed_data', self.dump_data_path), 'vocab.pkl')):
            vocab = pkl.load(open(os.path.join(os.path.join('preprocessed_data', self.dump_data_path), 'vocab.pkl'), 'rb'), encoding='latin1')
            print ('read vocab file from ', os.path.abspath(os.path.join(os.path.join('preprocessed_data', self.dump_data_path), 'vocab.pkl')))
            return vocab
        vocab = {}
        id = 0
        for k,words in json.load(open(os.path.join(self.data_dir, vocab_file))).items():
            for word in words:
                word = '.'.join(word.split('.')[:-2])
                if word in data_raw:
                    vocab[word] = id
            if len(words)>0:
                id += 1
        for k in set(data_raw)-set(vocab):
            del data_raw[k]
        pkl.dump(vocab, open(os.path.join(os.path.join('preprocessed_data', self.dump_data_path), 'vocab.pkl'), 'wb'))    
        return vocab    
    
    def get_clusters(self, cluster_file, data_raw):
        if os.path.exists(os.path.join(os.path.join('preprocessed_data',self.dump_data_path), 'clusters.pkl')):
            clusters = pkl.load(open(os.path.join(os.path.join('preprocessed_data',self.dump_data_path), 'clusters.pkl'), 'rb'), encoding='latin1')
            clusters_inv = pkl.load(open(os.path.join(os.path.join('preprocessed_data',self.dump_data_path), 'clusters_inv.pkl'), 'rb'), encoding='latin1')
            print ('read cluster file from ', os.path.abspath(os.path.join(os.path.join('preprocessed_data',self.dump_data_path), 'clusters.pkl')))
            return clusters, clusters_inv       
        clusters_inv = {}
        clusters = {}
        cluster_id = -1
        for k,v in json.load(open(os.path.join(self.data_dir, cluster_file))).items():
            v = ['.'.join(vi.split('.')[:-2]) for vi in v]
            if not any([vi in data_raw for vi in v]):
                continue
            cluster_id +=1 
            for vi in v:
                if vi in data_raw:
                    clusters[vi] = cluster_id
        for k in set(data_raw.keys())-set(clusters.keys()):
            if k in self.stopwords:
                continue
            for ki in k.replace(' ','_').replace('-','_').split('_'):
                ki = ki.split("'")[0].replace('.','')
                if ki in clusters:
                    clusters[k] = clusters[ki]
                    continue
                ki_sing = singularize(ki)
                ki_lemma = lemma(ki)  
                if ki_lemma.endswith('ish') or ki_lemma.endswith('ing') or ki_lemma.endswith('ive'):
                    ki_lemma = ki_lemma[:-3]
                if ki_lemma.endswith('ed') or ki_lemma.endswith('ly'):
                    ki_lemma = ki_lemma[:-2]
                if ki_sing in clusters:
                    clusters[k] = clusters[ki_sing]
                    continue
                if ki_lemma in clusters:
                    clusters[k] = clusters[ki_lemma]
                    continue
            if k not in clusters:
                print ('Added cluster for ', k)
                clusters[k] = cluster_id
                cluster_id += 1        
        clusters_inv = {}        
        for k,v in clusters.items():
            if v not in clusters_inv:
                clusters_inv[v] = []       
            clusters_inv[v].append(k)  
        pkl.dump(clusters, open(os.path.join(os.path.join('preprocessed_data',self.dump_data_path), 'clusters.pkl'), 'wb'))
        pkl.dump(clusters_inv, open(os.path.join(os.path.join('preprocessed_data',self.dump_data_path), 'clusters_inv.pkl'), 'wb'))
        return clusters, clusters_inv       
    
    def preprocess_data(self, data):
        if self.sort_by=="synset":
            data_processed_by_synset = {}
            labels_by_synset = {}
            for attr in self.vocab:
                for synset, data_processed_for_attr in self.preprocess_data_for_attribute(data[attr], attr).items():
                    if self.cluster_classify:
                        labels_for_attr = [self.clusters[attr]]*(len(data_processed_for_attr))
                    else:
                        labels_for_attr = [self.vocab[attr]]*(len(data_processed_for_attr))
                    if synset not in data_processed_by_synset:
                        data_processed_by_synset[synset] = []
                    if synset not in labels_by_synset:
                        labels_by_synset[synset] = []
                    data_processed_by_synset[synset].extend(data_processed_for_attr)
                    labels_by_synset[synset].extend(labels_for_attr)
            data_processed = []
            labels = []
            for synset in data_processed_by_synset:
                data_processed.extend(data_processed_by_synset[synset])
                labels.extend(labels_by_synset[synset])
        elif self.sort_by=="cluster":
            data_processed_by_cluster = {}
            labels_by_cluster = {}
            for attr in self.vocab:
                cluster_id = self.clusters[attr]
                for data_processed_for_attr in self.preprocess_data_for_attribute(data[attr], attr).values():
                    if self.cluster_classify:
                        labels_for_attr = [self.clusters[attr]]*(len(data_processed_for_attr))
                        if self.clusters[attr]!=self.clusters_np[self.vocab[attr]]:
                            raise Exception('Something wrong with the cluster vocab')
                    else:    
                        labels_for_attr = [self.vocab[attr]]*(len(data_processed_for_attr))
                    if cluster_id not in data_processed_by_cluster:
                        data_processed_by_cluster[cluster_id] = []
                    if cluster_id not in labels_by_cluster:
                        labels_by_cluster[cluster_id] = []
                    data_processed_by_cluster[cluster_id].extend(data_processed_for_attr)   
                    labels_by_cluster[cluster_id].extend(labels_for_attr)
            data_processed = []
            labels = []
            for cluster in data_processed_by_cluster:
                data_processed.extend(data_processed_by_cluster[cluster])
                labels.extend(labels_by_cluster[cluster])
        else:
            data_processed = []
            labels = []
            for attr in self.vocab:
                for data_processed_for_attr in self.preprocess_data_for_attribute(data[attr], attr).values():
                    if self.cluster_classify:   
                        labels_for_attr = [self.clusters[attr]]*(len(data_processed_for_attr))
                    else:
                        labels_for_attr = [self.vocab[attr]]*(len(data_processed_for_attr))
                    data_processed.extend(data_processed_for_attr)
                    labels.extend(labels_for_attr) 
        if self.shuffle_data==1:
            data_labels = list(zip(data_processed, labels))
            random.shuffle(data_labels)
            data_processed, labels = zip(*data_labels)            
        return data_processed, labels
    
    def get_num_classes(self):
        if self.cluster_classify:
            return self.cluster_size
        else:
            return self.vocab_size
        
    def chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]
            
    def get_nonattribute_words(self, words, attribute_words):
        nonattribute_words = (words - attribute_words) - self.stopwords
        self.maxlen = max(self.maxlen, len(nonattribute_words))
        return nonattribute_words
    
    def preprocess_data_for_attribute(self, data_for_attribute, attribute):
        attribute_sing = singularize(attribute)
        attribute_plur = pluralize(attribute)
        attribute_lemma = lemma(attribute)
        attribute_words = set([attribute, attribute_sing, attribute_plur, attribute_lemma])
        data = {}
        for synset in data_for_attribute:
            if synset not in data:
                data[synset] = []
            for region in data_for_attribute[synset]:
                image_id = region['image_id']
                if image_id in self.image_concepts_glove_emb:
                    glove_emb = self.image_concepts_glove_emb[image_id]
                    nonattr_words = self.get_nonattribute_words(glove_emb.keys(), attribute_words)
                    nonattr_words_glove_emb = [glove_emb[x] for x in nonattr_words]
                    region['nonattr_words_glove_emb'] = nonattr_words_glove_emb
                    data[synset].append(region)
        return data
        
    def get_image_region(self, region):
        image_id = region['image_id']
        bbox = region['bbox']
        image_file = self.image_dir+'/'+str(image_id)+'.jpg'
        image = cv2.imread(image_file)
        x, y, h, w = bbox
        cropped_image = image[y:y+h,x:x+w]
        try:
            cropped_image = Image.fromarray(cropped_image)
        except:
            cropped_image = Image.fromarray(image)
        old_size = cropped_image.size
        ratio = float(self.desired_size)/max(old_size)
        new_size = tuple([int(math.ceil(x*ratio)) for x in old_size])
        cropped_image = cropped_image.resize(new_size, Image.ANTIALIAS)
        new_im = Image.new("RGB", (self.desired_size, self.desired_size))
        new_im.paste(cropped_image, ((self.desired_size-new_size[0])//2,
                            (self.desired_size-new_size[1])//2))
        new_im = np.transpose(np.asarray(new_im, dtype=np.float32), (2,0,1))
        return new_im    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_region = self.get_image_region(self.data[idx])
        context_word_glove_emb = np.asarray(self.data[idx]['nonattr_words_glove_emb'], dtype=np.float32)
        pad_size = self.maxlen - context_word_glove_emb.shape[0]
        zeros = np.zeros((pad_size, context_word_glove_emb.shape[1]), dtype=np.float32)
        context_attention_vector = np.asarray([1.]*context_word_glove_emb.shape[0]+[1e-5]*pad_size, dtype=np.float32)
        context_word_glove_emb = np.concatenate([context_word_glove_emb, zeros], axis=0)
        label = self.labels[idx]
        return image_region, label, context_word_glove_emb, context_attention_vector







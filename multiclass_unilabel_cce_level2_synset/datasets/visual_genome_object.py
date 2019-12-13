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
    def __init__(self, image_dir, data, label, vocab, maxlen, glove_embs):
        self.image_dir = image_dir
        self.data = data
        self.label = label
        self.vocab = vocab
        self.vocab_size = len(set(vocab.values()))
        self.maxlen = maxlen
        self.glove_embs = glove_embs
        self.desired_size = 256
        
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
        if context_word_glove_emb.shape[0]==0:
            print ('context_word_glove_emb.shape[0]==0')
        pad_size = self.maxlen - context_word_glove_emb.shape[0]
        zeros = np.zeros((pad_size, context_word_glove_emb.shape[1]), dtype=np.float32)
        context_attention_vector = np.asarray([1.]*context_word_glove_emb.shape[0]+[1e-5]*pad_size, dtype=np.float32)
        context_word_glove_emb = np.concatenate([context_word_glove_emb, zeros], axis=0)
        label = self.label[idx]
        return image_region, label, context_word_glove_emb, context_attention_vector
        #return image_region, label, self.glove_embs, context_attention_vector
    
        

class VisualGenomeObjectCatalog():
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
        data_raw = {}
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
        self.image_concepts_glove_emb = pkl.load(open(os.path.join(self.data_dir, image_concepts_glove_emb_file), 'rb'), encoding='latin1')
        attributes_glove_emb = pkl.load(open(os.path.join(self.data_dir, attributes_glove_emb_file), 'rb'), encoding='latin1')
        glove_embedding_dim = list(attributes_glove_emb.values())[0].shape[0]
        self.attributes_glove_emb = np.zeros((len(self.vocab), glove_embedding_dim), dtype=np.float32)
        for k,v in attributes_glove_emb.items():
            k = '.'.join(k.split('.')[:-2])
            if k not in self.vocab:
                continue
            k_id = self.vocab[k]
            self.attributes_glove_emb[k_id] = self.attributes_glove_emb[k_id] + v
        self.vocab_size = len(set(self.vocab.values()))
        self.cluster_size = len(self.clusters_inv)
        print ('vocab size ', self.vocab_size)
        print ('cluster size ', self.cluster_size)
        self.clusters_np = np.zeros((self.vocab_size), dtype=np.int32)
        for k,v in self.clusters.items():
            self.clusters_np[self.vocab[k]] = int(v)
        self.cluster_embedding_mat = np.zeros((self.cluster_size, glove_embedding_dim),dtype=np.float32)
        for k,v in self.clusters_inv.items():
            glove_embs = np.asarray([self.attributes_glove_emb[self.vocab[vi]] for vi in v])
            cluster_glove_emb = np.mean(glove_embs, axis=0)
            self.cluster_embedding_mat[k] = cluster_glove_emb
            
        start = time.time()
        self.maxlen = 0
        self.data, self.labels, self.vocabs, self.maxlens = self.preprocess_data(data_raw)
        self.datasets = self.get_datasets()
        if len(gpu_ids)==0:
            self.device = 'cpu'
        else:
            self.device = 'cuda'
        end = time.time()
        print ('Preprocessed positive and negative data in ', (end-start), 'secs')
    
    
    def get_datasets(self):
        datasets = {}
        for cluster_id in self.data:
            data = self.data[cluster_id]
            label = self.labels[cluster_id]
            vocab = self.vocabs[cluster_id]
            maxlen = self.maxlens[cluster_id]
            glove_embs = np.asarray([self.attributes_glove_emb[self.vocab[vi]] for vi in self.clusters_inv[cluster_id]], dtype=np.float32)
            dataset = VisualGenomeObjectDataset(self.image_dir, data, label, vocab, maxlen, glove_embs)
            datasets[cluster_id] = dataset
        return datasets
    
    def get_dataset(self, cluster_id):
        return self.datasets[cluster_id]
    
    def get_vocab(self, vocab_file, data_raw):
        if os.path.exists(os.path.join(os.path.join('preprocessed_data', self.dump_data_path), 'vocab.pkl')):
            vocab = pkl.load(open(os.path.join(os.path.join('preprocessed_data', self.dump_data_path), 'vocab.pkl'), 'rb'), encoding='latin1')
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
            print ('deleting ', k)
            del data_raw[k]    
        print ('vocab size ', len(vocab))
        pkl.dump(vocab, open(os.path.join(os.path.join('preprocessed_data', self.dump_data_path), 'vocab.pkl'), 'wb'))    
        return vocab    
    
    def get_clusters(self, cluster_file, data_raw):
        if os.path.exists(os.path.join(os.path.join('preprocessed_data',self.dump_data_path), 'clusters.pkl')):
            clusters = pkl.load(open(os.path.join(os.path.join('preprocessed_data',self.dump_data_path), 'clusters.pkl'), 'rb'), encoding='latin1')
            clusters_inv = pkl.load(open(os.path.join(os.path.join('preprocessed_data',self.dump_data_path), 'clusters_inv.pkl'), 'rb'), encoding='latin1')
            return clusters, clusters_inv       
        clusters_inv = {}
        clusters = {}
        cluster_id = -1
        for k,v in json.load(open(os.path.join(self.data_dir, cluster_file))).items():
            v = ['.'.join(vi.split('.')[:-2]) for vi in v]
            if not any([vi in self.vocab for vi in v]):
                continue
            cluster_id +=1 
            for vi in v:
                if vi in data_raw:
                    clusters[vi] = cluster_id
        for k in set(self.vocab.keys())-set(clusters.keys()):
            if k in self.stopwords:
                continue
            searched_for = set([])
            for ki in k.replace(' ','_').replace('-','_').split('_'):
                ki = ki.split("'")[0].replace('.','')
                searched_for.add(ki)
                if ki in clusters:
                    clusters[k] = clusters[ki]
                    continue
                ki_sing = singularize(ki)
                ki_lemma = lemma(ki)  
                if ki_lemma.endswith('ish') or ki_lemma.endswith('ing') or ki_lemma.endswith('ive'):
                    ki_lemma = ki_lemma[:-3]
                if ki_lemma.endswith('ed') or ki_lemma.endswith('ly'):
                    ki_lemma = ki_lemma[:-2]
                searched_for.add(ki_sing)
                searched_for.add(ki_lemma)
                if ki_sing in clusters:
                    clusters[k] = clusters[ki_sing]
                    continue
                if ki_lemma in clusters:
                    clusters[k] = clusters[ki_lemma]
                    continue
            if k not in clusters:
                print ('Added cluster for ', k)
                cluster_id += 1
                clusters[k] = cluster_id
        clusters_inv = {}        
        for k,v in clusters.items():
            if v not in clusters_inv:
                clusters_inv[v] = []       
            clusters_inv[v].append(k) 
        pkl.dump(clusters, open(os.path.join(os.path.join('preprocessed_data',self.dump_data_path), 'clusters.pkl'), 'wb'))
        pkl.dump(clusters_inv, open(os.path.join(os.path.join('preprocessed_data',self.dump_data_path), 'clusters_inv.pkl'), 'wb'))
        return clusters, clusters_inv       
    
    def preprocess_data(self, data):
        data_processed_by_cluster = {}
        labels_by_cluster = {}
        vocab_by_cluster = {}
        maxlen_by_cluster = {}
        for cluster_id in self.clusters_inv:
            attrs = self.clusters_inv[cluster_id]
            vocab = {}
            self.maxlen = 0
            for attr in attrs:
                vocab[attr] = self.vocab[attr]
            vocab_id_map = {}
            id = 0
            for v in set(vocab.values()):
                vocab_id_map[v] = id
                id += 1
            vocab_by_cluster[cluster_id] = {k:vocab_id_map[v] for k,v in vocab.items()}
            data_processed_by_cluster[cluster_id] = []
            labels_by_cluster[cluster_id] = []
            for attr in attrs:
                for data_processed_for_attr in self.preprocess_data_for_attribute(data[attr], attr).values():
                    labels_for_attr = [vocab_id_map[vocab[attr]]]*(len(data_processed_for_attr))
                    data_processed_by_cluster[cluster_id].extend(data_processed_for_attr) 
                    labels_by_cluster[cluster_id].extend(labels_for_attr)
            maxlen_by_cluster[cluster_id] = self.maxlen       
        return data_processed_by_cluster, labels_by_cluster, vocab_by_cluster, maxlen_by_cluster  
        
    
    def get_num_classes(self):
        if self.cluster_classify:
            return (self.clusters_size)
        else:
            return (self.vocab_size)
        
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
        
    





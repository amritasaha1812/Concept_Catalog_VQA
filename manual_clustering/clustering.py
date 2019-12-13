#import pickle as pkl
import json
from collections import Counter
clusters = {}
cluster_id = 0
list = []
for line in open('synset_clusters.txt').readlines():
    line = line.strip()
    if len(line)>0:
        list.append(line)
    else:
        if len(list)==0:
            continue
        clusters[cluster_id] = list    
        list = []
        cluster_id += 1
print ('num clusters ', len(clusters))        
print ('Cluster size histogram ', Counter([len(x) for x in clusters.values()]))
json.dump(clusters, open('synset_concept_clusters_final.json','w'), indent=1)         

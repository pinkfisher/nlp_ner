'''
Created on Sep 23, 2017

@author: Prateek Kolhar
'''

import os
os.chdir("D:\\Programming\\eclipse-workspace-java\\nlp1_ner_tag\\src")
import crf_pieces as cp
train = cp.read_data("./data/eng.train")
dev = cp.read_data("./data/eng.testa")

file=open("./paths","r")
lines = file.readlines()
brown_clusters={}
for line in lines:
    sp = line.split("\t")
    brown_clusters[sp[1]]=sp[0]
    
file.close()

file = open("./all_data_clark_output.200","r")
lines = file.readlines()
clark_clusters={}
for line in lines:
    sp = line.split(" ")
    clark_clusters[sp[0]]=sp[1]

file.close()

[tag_indexer, feature_indexer, feature_cache] = cp.crf_train_feature_generate(train, brown_clusters, clark_clusters)
[dev_feature_cache] = cp.crf_test_feature_generate(dev, tag_indexer, feature_indexer, brown_clusters, clark_clusters)
model = cp.crf_train_pieces(train, tag_indexer, feature_indexer, feature_cache,190,dev,1,brown_clusters, clark_clusters, dev_feature_cache=dev_feature_cache)
print "german"

dev_decoded = [cp.crf_decode_piece(test_ex, model.tag_indexer, model.feature_indexer, model.feature_weights, brown_clusters, clark_clusters) for test_ex in dev]
cp.print_evaluation(dev, dev_decoded)

mod = cp.crf_train_pieces([train[9]], tag_indexer, feature_indexer, [feature_cache[9]],1)
mod = cp.crf_train_pieces(train[9:20], tag_indexer, feature_indexer, feature_cache[9:20],1)
model = cp.crf_train_pieces(train, tag_indexer, feature_indexer, feature_cache,10)
  
reload(cp)
dev_decoded = [cp.crf_decode_piece(test_ex, model.tag_indexer, model.feature_indexer, model.feature_weights) for test_ex in dev]
cp.crf_decode_piece(sentence, mod.tag_indexer, mod.feature_indexer, mod.feature_weights)
cp.crf_decode_piece(sentence, model.tag_indexer, model.feature_indexer, model.feature_weights)
cp.print_evaluation(dev, dev_decoded)

cv = train[100:3500]
cv_decoded = [cp.crf_decode_piece(test_ex, model.tag_indexer, model.feature_indexer, model.feature_weights) for test_ex in cv]
cp.print_evaluation(cv, cv_decoded)


import pickle
file = open ("model1000.pickle", "wb")
pickle.dump(model1000,file)

file= open ("model1000.pickle","r")
model1000_new = pickle.load(file) 
dev_decoded1000_new = [cp.crf_decode_piece(test_ex, model1000_new.tag_indexer, model1000_new.feature_indexer, model1000_new.feature_weights) for test_ex in dev]
cp.print_evaluation(dev, dev_decoded1000_new)






'''
Created on Sep 23, 2017

@author: Prateek Kolhar
'''

import os
os.chdir("D:\\Programming\\eclipse-workspace-java\\nlp1_ner_tag\\src")
import crf_pieces as cp
train = cp.read_data("./data/eng.train")
dev = cp.read_data("./data/eng.testa")

file=open("all_data_output.txt","r")
lines = file.readlines()
brown_clusters={}
for line in lines:
    sp = line.split("\t")
    brown_clusters[sp[0]]=sp[1]


[tag_indexer, feature_indexer, feature_cache] = cp.crf_train_feature_generate(train, brown_clusters)
model = cp.crf_train_pieces(train, tag_indexer, feature_indexer, feature_cache,10,dev,brown_clusters,1)
dev_decoded = [cp.crf_decode_piece(test_ex, model.tag_indexer, model.feature_indexer, model.feature_weights, brown_clusters) for test_ex in dev]
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






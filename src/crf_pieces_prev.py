'''
Created on Sep 24, 2017

@author: Prateek Kolhar
'''

from nerdata import *
from utils import *
from models import *
import timeit

import numpy as np


# import os
# os.chdir("D:\\Programming\\eclipse-workspace-java\\nlp1_ner_tag\\src")
# from crf_pieces import *
# train = read_data("./data/eng.train")
# dev = read_data("./data/eng.testa")
# [tag_indexer, feature_indexer, feature_cache] = crf_train_feature_generate(train)

def crf_train_feature_generate(sentences):
    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.get_index(tag)
    print "Extracting features"
    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in xrange(0, len(tag_indexer))] for j in xrange(0, len(sentences[i]))] for i in xrange(0, len(sentences))]
    for sentence_idx in xrange(0, len(sentences)):
        if sentence_idx % 100 == 0:
            print "Ex " + repr(sentence_idx) + "/" + repr(len(sentences))
        for word_idx in xrange(0, len(sentences[sentence_idx])):
            for tag_idx in xrange(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx], word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)
    
    return [tag_indexer, feature_indexer, feature_cache]

# mod = crf_train_pieces([train[9]], tag_indexer, feature_indexer, [feature_cache[9]],1)
# model = crf_train_pieces(train, tag_indexer, feature_indexer, feature_cache,1)
def crf_train_pieces(sentences, tag_indexer, feature_indexer, feature_cache, epoch_count):
    start = timeit.default_timer()
    wt = np.random.rand(len(feature_indexer))
    for epoch in range(0,epoch_count):
        
        
        for sentence_idx in xrange(0, len(sentences)):
#             start = timeit.default_timer()
            if sentence_idx%1000 ==0: print sentence_idx
            w_num=len(sentences[sentence_idx])
            t_num=len(tag_indexer)
#             feature_cache[sentence_idx]
            psi=np.zeros([t_num,w_num])
            for i in range(0,t_num):
                for j in range(0,w_num):
                    psi[i][j]=wt[feature_cache[sentence_idx][j][i]].sum() #check
#             stop = timeit.default_timer()
#             print "1:"+str(stop - start)
            
#             start = timeit.default_timer()
            print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            print np.isinf(psi).sum()
            print np.isnan(psi).sum()

            alpha=np.zeros([t_num,w_num])
            beta=np.zeros([t_num,w_num])
            gamma=np.zeros([t_num,w_num])
            
            
            
            
            alpha[:,0]=psi[:,0]
#             print "alpha"+str(0)
#             print np.isinf(alpha[:,i]).sum()
#             print np.isnan(alpha[:,i]).sum()
            
            for i in range(1,w_num):  
                alpha[:,i] = log_add_exp((np.ones([t_num,1])*(alpha[:,i-1]).reshape(1,t_num)),1)+psi[:,i]
                
#                 print "alpha"+str(i)
#                 print alpha[:,i].max()
#                 print np.isinf(alpha[:,i]).sum()
#                 print np.isnan(alpha[:,i]).sum()
            print alpha
            beta[:,-1]=np.zeros(t_num);
            for i in range(w_num-2,-1,-1):  
                beta[:,i] = log_add_exp((np.ones([t_num,1])*(beta[:,i+1]+psi[:,i+1]).reshape(1,t_num)),1)
            
            
            Z= log_add_exp(alpha[:,-1]+beta[:,-1])
#             print "alpha:"
#             print np.isinf(alpha).sum()
#             print np.isnan(alpha).sum()
#             print "beta:"
#             print np.isinf(beta).sum()
#             print np.isnan(beta).sum()
#             print "Z:"
#             print np.isinf(Z)
#             print np.isnan(Z)
#             print "gamma:"
#             print np.isinf(gamma).sum()
#             print np.isnan(gamma).sum()
            
            
            
            
            gamma=(alpha+beta)-Z
            print gamma
#             stop = timeit.default_timer()
#             print "2:"+str(stop - start)
            
            lr=1
#             start = timeit.default_timer()
            y_star = [tag_indexer.get_index(tag) for tag in sentences[sentence_idx].get_bio_tags()]
            for word_idx in range(w_num):
                f= feature_cache[sentence_idx][word_idx][y_star[word_idx]]
                
                wt[f]+=1*lr

                for t_idx in range(t_num):
                    f1 = feature_cache[sentence_idx][word_idx][t_idx]
                    
                    wt[f1] -=lr*np.exp(gamma[t_idx][word_idx])
#             print "wt:"
#             
#             print np.isinf(wt).sum()
#             print np.isnan(wt).sum()
#             stop = timeit.default_timer()
#             print "3:"+str(stop - start)
#             start = timeit.default_timer()
#             updatable_w = list(set([item for list1 in feature_cache[sentence_idx] for list2 in list1 for item in list2]))
#             stop = timeit.default_timer()
#             print "3:"+str(stop - start)
#             
#             times = np.zeros(3)
#             print "updatable_w"+str(len(updatable_w))
#             start1 = timeit.default_timer()
#             for wt_idx in updatable_w:
#                 
#                 start = timeit.default_timer()
#                 f_ind_mat = np.zeros([t_num,w_num])
#                 stop = timeit.default_timer()
#                 times[0]+=(stop - start)
#                 
#                 start = timeit.default_timer()
#                 for i in range(0,t_num):
#                     for j in range(0,w_num):
#                         if wt_idx in feature_cache[sentence_idx][j][i]: f_ind_mat[i][j] = 1
#                 stop = timeit.default_timer()
#                 times[1]+=(stop - start)
#                 
#                 start = timeit.default_timer()
#                 y_star = [tag_indexer.get_index(tag) for tag in sentences[sentence_idx].get_bio_tags()]                
#                 sum_f = sum([ f_ind_mat[(y_star[i],i)] for i in range(0,len(y_star))])
#                 exp_sum_f = np.multiply(f_ind_mat,np.exp(gamma)).sum()
#                 grad = sum_f - exp_sum_f
#                 wt[wt_idx]=wt[wt_idx]+lr*grad
#                 stop = timeit.default_timer()
#                 times[2]+=(stop - start)
#                 
#             stop1 = timeit.default_timer()
#             print "4.1:"+str(times[0]*1.0/len(updatable_w))
#             print "4.2:"+str(times[1]*1.0/len(updatable_w))
#             print "4.3:"+str(times[2]*1.0/len(updatable_w))
#             print "4:"+str(stop1 - start1)
    stop = timeit.default_timer()
    print "total time:"+str(stop - start)       
    return CrfNerModel(tag_indexer, feature_indexer, wt)

# dev_decoded = [crf_decode_piece(test_ex, mod.tag_indexer, mod.feature_indexer, mod.feature_weights) for test_ex in dev]
# crf_decode_piece(sentence, mod.tag_indexer, mod.feature_indexer, mod.feature_weights)
# crf_decode_piece(sentence, model.tag_indexer, model.feature_indexer, model.feature_weights)
# print_evaluation(dev, dev_decoded)
def crf_decode_piece(sentence, tag_indexer, feature_indexer, feature_weights):
    
    
    w_num=len(sentence)
    t_num=len(tag_indexer)
    psi=np.zeros([t_num,w_num])
    feature_cache = [[[] for k in xrange(0, len(tag_indexer))] for j in xrange(0, len(sentence))]
    for word_idx in xrange(0, len(sentence)):
        for tag_idx in xrange(0, len(tag_indexer)):
            feature_cache[word_idx][tag_idx] = extract_emission_features(sentence, word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=False)
            
    for i in range(0,t_num):
        for j in range(0,w_num):
#             print "i: "+str(i)+" j: "+str(j)
            psi[i][j]=feature_weights[feature_cache[j][i]].sum()
        
    pred_idx=psi.argmax(1)
    
    pred_tags = [tag_indexer.get_object(i) for i in pred_idx]
    return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))

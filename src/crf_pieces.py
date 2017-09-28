'''
Created on Sep 23, 2017

@author: Prateek Kolhar
'''
from nerdata import *
from utils import *
from models import *
import timeit
import re

import numpy as np
from scipy import misc



def crf_train_feature_generate(sentences, brown_clusters):
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
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_experiments_features(sentences[sentence_idx], word_idx, tag_indexer.get_object(tag_idx), feature_indexer, brown_clusters, add_to_indexer=True)
    
        
    return [tag_indexer, feature_indexer, feature_cache]

def crf_train_pieces(sentences, tag_indexer, feature_indexer, feature_cache, epoch_count, dev, step, brown_clusters):
    start = timeit.default_timer()
    wt = np.ones(len(feature_indexer))
    for epoch in range(0,epoch_count):
        print epoch
        for sentence_idx in xrange(0, len(sentences)):
            
            
            w_num=len(sentences[sentence_idx])
            t_num=len(tag_indexer)

            psi=np.zeros([t_num,w_num])
            for i in range(0,t_num):
                for j in range(0,w_num):
                    psi[i][j]=wt[feature_cache[sentence_idx][j][i]].sum() #check
                
            alpha=np.zeros([t_num,w_num])
            beta=np.zeros([t_num,w_num])
            gamma=np.zeros([t_num,w_num])
            
            alpha[:,0]=psi[:,0]

            for i in range(1,w_num):  
#                 for j in range(t_num):
#                     alpha[j,i] = 0;
#                     for k in range(t_num):
#                         alpha[j,i]= np.logaddexp(alpha[k][i-1]+psi[j][i],alpha[j][i])
                alpha[:,i] = log_add_exp((np.ones([t_num,1])*(alpha[:,i-1]).reshape(1,t_num)),1)+psi[:,i]
                
#             print alpha
            beta[:,-1]=np.zeros(t_num);
            for i in range(w_num-2,-1,-1):  
                beta[:,i] = log_add_exp((np.ones([t_num,1])*(beta[:,i+1]+psi[:,i+1]).reshape(1,t_num)),1)
            
            Z= log_add_exp(alpha[:,-1]+beta[:,-1])

            gamma=(alpha+beta)-Z
#             print alpha
            
            lr=.1

            y_star = [tag_indexer.get_index(tag) for tag in sentences[sentence_idx].get_bio_tags()]
            for word_idx in range(w_num):
                f= feature_cache[sentence_idx][word_idx][y_star[word_idx]]
                
                wt[f]+=1*lr

                for t_idx in range(t_num):
                    f1 = feature_cache[sentence_idx][word_idx][t_idx]
                    
                    wt[f1] -=lr*np.exp(gamma[t_idx][word_idx])
        if(epoch%step ==0):
            dev_decoded = [crf_decode_piece(test_ex, tag_indexer, feature_indexer, wt, brown_clusters) for test_ex in dev]
            print_evaluation(dev, dev_decoded)

    stop = timeit.default_timer()
    print "total time:"+str(stop - start)       
    return CrfNerModel(tag_indexer, feature_indexer, wt)

def crf_train_trans_pieces(sentences, tag_indexer, feature_indexer, feature_cache, epoch_count):
    start = timeit.default_timer()
    wt = np.ones(len(feature_indexer))
#     from tag1 to tag2 >> rows are tag1. columns are tag2
    wt_t = np.ones([len(tag_indexer),len(tag_indexer)])
    
    for epoch in range(0,epoch_count):
        print epoch
        for sentence_idx in xrange(0, len(sentences)):
            
            
            w_num=len(sentences[sentence_idx])
            t_num=len(tag_indexer)
            y_star = [tag_indexer.get_index(tag) for tag in sentences[sentence_idx].get_bio_tags()]
            
            psi=np.zeros([t_num,w_num])
            psi_t= np.zeros([t_num,t_num])
            for i in range(0,t_num):
                for j in range(0,w_num):
                    psi[i][j]=wt[feature_cache[sentence_idx][j][i]].sum() #check
            
            psi_t=wt_t
                
            alpha=np.zeros([t_num,w_num])
            beta=np.zeros([t_num,w_num])
            gamma=np.zeros([t_num,w_num])
            
            alpha[:,0]=psi[:,0]

            for i in range(1,w_num):  
                alpha[:,i] = log_add_exp((np.ones([t_num,1])*(alpha[:,i-1]).reshape(1,t_num))+psi_t.transpose(),1)+psi[:,i]
                
#             print alpha
            beta[:,-1]=np.zeros(t_num);
            for i in range(w_num-2,-1,-1):  
                beta[:,i] = log_add_exp((np.ones([t_num,1])*(beta[:,i+1]+psi[:,i+1]).reshape(1,t_num))+psi_t,1)
            
            Z= log_add_exp(alpha[:,-1]+beta[:,-1])
            
            check = (alpha+beta).sum(0)
            print (alpha+beta).sum(0)
            if (np.abs(check -check.max())).sum()>.5: print (alpha+beta).sum(0)
            gamma=(alpha+beta)-Z
#             print alpha
            
            lr=.1

            for w_idx in range(1,w_num):
                
                wt_t[y_star[w_idx-1],y_star[w_idx]] +=1*lr 
                
                temp = alpha[:,w_idx-1]*np.ones([1,t_num]) + np.ones([t_num,1])*(psi[:,w_idx]+beta[:,w_idx]).reshape([1,t_num]) + psi_t
                LAD_temp = log_add_exp(log_add_exp(temp, 1))
                temp = temp - LAD_temp
                wt_t -= lr*np.exp(temp)
#                 for t_idx in range(t_num):
#                     for t_p_idx in range(t_num):
#                         wt_t[t_p_idx,t_idx]-=np.exp(alpha[t_p_idx][w_idx-1] + psi[t_idx,w_idx]+psi_t[w_idx,t_p_idx,t_idx]+beta[t_idx,w_idx])
                        
                
            for word_idx in range(w_num):
                f= feature_cache[sentence_idx][word_idx][y_star[word_idx]]
                
                wt[f]+=t_num*lr
                
                w_idx = word_idx
#                 f_t is 1 only for one combination of y_i and y_i-1. 
                temp = alpha[:,w_idx-1]*np.ones([1,t_num]) + np.ones([t_num,1])*(psi[:,w_idx]+beta[:,w_idx]).reshape([1,t_num]) + psi_t
                LAD_temp = log_add_exp(log_add_exp(temp, 1))
                temp = temp - LAD_temp
                for t_idx in range(t_num):
                    f1 = feature_cache[sentence_idx][word_idx][t_idx]
                    
                    wt[f1] -=lr*np.exp(temp[:,t_idx]).sum()

    stop = timeit.default_timer()
    print "total time:"+str(stop - start)       
    return CrfNerModelTrans(tag_indexer, feature_indexer, wt, wt_t)

def crf_decode_piece(sentence, tag_indexer, feature_indexer, feature_weights, brown_clusters):
    
    
    w_num=len(sentence)
    t_num=len(tag_indexer)
    psi=np.zeros([t_num,w_num])
    feature_cache = [[[] for k in xrange(0, len(tag_indexer))] for j in xrange(0, len(sentence))]
    for word_idx in xrange(0, len(sentence)):
        for tag_idx in xrange(0, len(tag_indexer)):
            feature_cache[word_idx][tag_idx] = extract_emission_experiments_features(sentence, word_idx, tag_indexer.get_object(tag_idx), feature_indexer, brown_clusters, add_to_indexer=False)
            
    for i in range(0,t_num):
        for j in range(0,w_num):
            psi[i][j]=feature_weights[feature_cache[j][i]].sum()
    
    score = np.ones([t_num,w_num])*(-1*np.inf)
    back_tracking = np.ones([t_num,w_num])*(-1)
    

    for i in range(t_num):
        if isI(tag_indexer.get_object(i)) ==False : score[i,0] = psi[i,0]
    
    for i in range(1,w_num):
        for j in range(t_num):
            if not isI(tag_indexer.get_object(j)): 
                back_tracking[j,i] = score[:,i-1].argmax()
                score[j,i] = score[:,i-1].max()+psi[j,i]
            else:
                valid_idx = [tag_indexer.get_index("B-"+get_tag_label(tag_indexer.get_object(j))),j] 
                score[j,i] = score[valid_idx,i-1].max()+psi[j,i]
                arg_max = score[valid_idx,i-1].argmax()
                back_tracking[j,i] = valid_idx[arg_max]
    
    
    
    pred_idx=np.zeros(w_num,dtype='Int32')
    
    pred_idx[-1] = score[:,-1].argmax()
    for i in range(w_num-2,-1,-1):
        pred_idx[i] = back_tracking[pred_idx[i+1],i+1]
    
    pred_tags = [tag_indexer.get_object(i) for i in pred_idx]
    return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))


def crf_decode_piece_old(sentence, tag_indexer, feature_indexer, feature_weights):
    
    
    w_num=len(sentence)
    t_num=len(tag_indexer)
    psi=np.zeros([t_num,w_num])
    feature_cache = [[[] for k in xrange(0, len(tag_indexer))] for j in xrange(0, len(sentence))]
    for word_idx in xrange(0, len(sentence)):
        for tag_idx in xrange(0, len(tag_indexer)):
            feature_cache[word_idx][tag_idx] = extract_emission_features(sentence, word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=False)
    
    for i in range(0,t_num):
        for j in range(0,w_num):
            psi[i][j]=feature_weights[feature_cache[j][i]].sum()

  
        
    pred_idx=psi.argmax(0)
#     print pred_idx
    pred_tags = [tag_indexer.get_object(i) for i in pred_idx]
#     print pred_tags
    return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))


def crf_decode_trans_piece(sentence, tag_indexer, feature_indexer, feature_weights, transition_weights):
    
    
    w_num=len(sentence)
    t_num=len(tag_indexer)
    psi=np.zeros([t_num,w_num])
    psi_t=np.zeros([w_num,t_num,t_num])
    feature_cache = [[[] for k in xrange(0, len(tag_indexer))] for j in xrange(0, len(sentence))]
    for word_idx in xrange(0, len(sentence)):
        for tag_idx in xrange(0, len(tag_indexer)):
            feature_cache[word_idx][tag_idx] = extract_emission_features(sentence, word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=False)
    
    
    psi_t=transition_weights
        
    for i in range(0,t_num):
        for j in range(0,w_num):
            psi[i][j]=feature_weights[feature_cache[j][i]].sum()
    
    score = np.ones([t_num,w_num])*(-1*np.inf)
    back_tracking = np.ones([t_num,w_num])*(-1)
    

    for i in range(t_num):
        if isI(tag_indexer.get_object(i)) ==False : score[i,0] = psi[i,0]
    
    for i in range(1,w_num):
        for j in range(t_num):
            if not isI(tag_indexer.get_object(j)): 
                back_tracking[j,i] = (score[:,i-1]+psi_t[:,j]).argmax()
                score[j,i] = (score[:,i-1]+psi_t[:,j]).max()+psi[j,i]
            else:
                valid_idx = [tag_indexer.get_index("B-"+get_tag_label(tag_indexer.get_object(j))),j] 
                score[j,i] = (score[valid_idx,i-1]+psi_t[valid_idx,j]).max()+psi[j,i]
                arg_max = (score[valid_idx,i-1]+psi_t[valid_idx,j]).argmax()
                back_tracking[j,i] = valid_idx[arg_max]
    
    
    
    pred_idx=np.zeros(w_num,dtype='Int32')
    
    pred_idx[-1] = score[:,-1].argmax()
    for i in range(w_num-2,-1,-1):
        pred_idx[i] = back_tracking[pred_idx[i+1],i+1]
    
    pred_tags = [tag_indexer.get_object(i) for i in pred_idx]
    return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))



def create_text_file(sentences,file_name):
    file = open(file_name,"w")
    for sentence in sentences:
        for token in sentence.tokens:
            word = token.word
            word = re.sub(r'[^\w\s]','',word)
            if(len(word)>0):
                file.write(word)
                file.write(" ")
        file.write("\n")
    
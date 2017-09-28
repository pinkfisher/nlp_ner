# models.py

from nerdata import *
from utils import *

import numpy as np
from scipy import misc

# Scoring function for sequence models based on conditional probabilities.
# Scores are provided for three potentials in the model: initial scores (applied to the first tag),
# emissions, and transitions. Note that CRFs typically don't use potentials of the first type.
class ProbabilisticSequenceScorer(object):
    def __init__(self, tag_indexer, word_indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence, tag_idx):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence, prev_tag_idx, curr_tag_idx):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence, tag_idx, word_posn):
        word = sentence.tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.get_index("UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class HmmNerModel(object):
    def __init__(self, tag_indexer, word_indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs
        self.scorer = ProbabilisticSequenceScorer(tag_indexer, word_indexer, init_log_probs, transition_log_probs, emission_log_probs);
    # Takes a LabeledSentence object and returns a new copy of that sentence with a set of chunks predicted by
    # the HMM model. See BadNerModel for an example implementation
    
    def decode(self, sentence):
        w_num=len(sentence.tokens)
        t_num=len(self.tag_indexer)
        score=np.zeros([t_num,w_num])
        prev_token=np.ones([len(self.tag_indexer),len(sentence.tokens)])*-1
        word = sentence.tokens[0].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.get_index("UNK")
        score[:,0]=self.init_log_probs+self.emission_log_probs[:,word_idx]
        for i in range(1,w_num):  
            word = sentence.tokens[i].word
            word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.get_index("UNK")
            score[:,i] = ((np.ones([t_num,1])*score[:,i-1].reshape(1,t_num))+self.transition_log_probs.transpose()).max(1) + self.emission_log_probs[:,word_idx]
            prev_token[:,i] = ((np.ones([t_num,1])*score[:,i-1].reshape(1,t_num))+self.transition_log_probs.transpose()).argmax(1)
        pred_idx = np.zeros(w_num).astype(int)
        pred_idx[-1] = int(score[:,-1].argmax())
        for i in range(w_num-2,-1,-1):
            pred_idx[i] = int(prev_token[pred_idx[i+1],i+1])
        pred_tags = [self.tag_indexer.get_object(i) for i in pred_idx]
        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))
    
def log_add_exp(x,dim=0):
#     if dim ==0:
#         x_max = x.max()
#         return np.log(np.exp(x - x_max).sum(dim))
#     else:
#         x_max = x.max(1)
#         return np.log(np.exp(x - x_max.reshape([1,len(x_max)])).sum(dim))
    return misc.logsumexp(x,dim)

def forward_backward(sentence, model):
    w_num=len(sentence.tokens)
    t_num=len(model.tag_indexer)
    alpha=np.zeros([t_num,w_num])
    beta=np.zeros([t_num,w_num])
    gamma=np.zeros([t_num,w_num])
    
    word = sentence.tokens[0].word
    word_idx = model.word_indexer.index_of(word) if model.word_indexer.contains(word) else model.word_indexer.get_index("UNK")
    alpha[:,0]=model.init_log_probs+model.emission_log_probs[:,word_idx]
    
    for i in range(1,w_num):  
        word = sentence.tokens[i].word
        word_idx = model.word_indexer.index_of(word) if model.word_indexer.contains(word) else model.word_indexer.get_index("UNK")
        alpha[:,i] = log_add_exp((np.ones([t_num,1])*alpha[:,i-1].reshape(1,t_num))+model.transition_log_probs.transpose(),1) \
         + model.emission_log_probs[:,word_idx]
    
    beta[:,-1]=np.ones(9);
    for i in range(w_num-2,-1,-1):  
        word = sentence.tokens[i+1].word
        word_idx = model.word_indexer.index_of(word) if model.word_indexer.contains(word) else model.word_indexer.get_index("UNK")
        beta[:,i] = log_add_exp((np.ones([t_num,1])*(beta[:,i+1]+model.emission_log_probs[:,word_idx]).reshape(1,t_num))+model.transition_log_probs,1)
    gamma=(alpha+beta)-np.ones([t_num,1])*log_add_exp(alpha+beta, 0).reshape(1,w_num)
    pred_idx= gamma
   
    pred_tags = [model.tag_indexer.get_object(i) for i in pred_idx]
    return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))
# Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
# Any word that only appears once in the corpus is replaced with UNK. A small amount
# of additive smoothing is applied to
def train_hmm_model(sentences):
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter.increment_count(token.word, 1.0)
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer),len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer),len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in xrange(0, len(sentence)):
            tag_idx = tag_indexer.get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_indexer.get_index(bio_tags[i])] += 1.0
            else:
                transition_counts[tag_indexer.get_index(bio_tags[i-1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    print repr(init_counts)
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    print "Tag indexer: " + repr(tag_indexer)
    print "Initial state log probabilities: " + repr(init_counts)
    print "Transition log probabilities: " + repr(transition_counts)
    print "Emission log probs too big to print..."
    print "Emission log probs for India: " + repr(emission_counts[:,word_indexer.get_index("India")])
    print "Emission log probs for Phil: " + repr(emission_counts[:,word_indexer.get_index("Phil")])
    print "   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)"
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


# Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
# At test time, unknown words will be replaced by UNKs.
def get_word_index(word_indexer, word_counter, word):
    if word_counter.get_count(word) < 1.5:
        return word_indexer.get_index("UNK")
    else:
        return word_indexer.get_index(word)


class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights

    # Takes a LabeledSentence object and returns a new copy of that sentence with a set of chunks predicted by
    # the CRF model. See BadNerModel for an example implementation
    def decode(self, sentence):
        w_num=len(sentence)
        t_num=len(self.tag_indexer)
        psi=np.zeros([t_num,w_num])
        feature_cache = [[[] for k in xrange(0, len(self.tag_indexer))] for j in xrange(0, len(sentence))]
        for word_idx in xrange(0, len(sentence)):
            for tag_idx in xrange(0, len(self.tag_indexer)):
                feature_cache[word_idx][tag_idx] = extract_emission_features(sentence, word_idx, self.tag_indexer.get_object(tag_idx), self.feature_indexer, add_to_indexer=True)
                
        for i in range(0,t_num):
            for j in range(0,w_num):
                psi[i][j]=self.feature_weights[feature_cache[j][i]].sum()
            
        pred_idx=psi.argmax(1)
        
        pred_tags = [self.tag_indexer.get_object(i) for i in pred_idx]
        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tags))

class CrfNerModelTrans(CrfNerModel):
    def __init__(self, tag_indexer, feature_indexer, feature_weights, transition_weights):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        self.transition_weights = transition_weights
# Trains a CrfNerModel on the given corpus of sentences.
def train_crf_model(sentences):
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
#      code begins
    wt = np.random.rand(len(feature_indexer))
    for epoch in range(0,1):
        
        for sentence_idx in xrange(0, len(sentences)):
            print sentence_idx
            w_num=len(sentences[sentence_idx])
            t_num=len(tag_indexer)
#             feature_cache[sentence_idx]
            psi=np.zeros([t_num,w_num])
            for i in range(0,t_num):
                for j in range(0,w_num):
                    psi[i][j]=wt[feature_cache[sentence_idx][j][i]].sum() #check
            alpha=np.zeros([t_num,w_num])
            beta=np.zeros([t_num,w_num])
            gamma=np.zeros([t_num,w_num])
            
            alpha[:,0]=psi[:,0]
            for i in range(1,w_num):  
                alpha[:,i] = log_add_exp((np.ones([t_num,1])*(alpha[:,i-1]).reshape(1,t_num)),1)+psi[:,i]
                 
            beta[:,-1]=np.ones(t_num);
            for i in range(w_num-2,-1,-1):  
                beta[:,i] = log_add_exp((np.ones([t_num,1])*(beta[:,i+1]+psi[:,i+1]).reshape(1,t_num)))
            
            Z= log_add_exp(alpha[:,-1])
            gamma=(alpha+beta)-Z
            
            lr=1
            updatable_w = list(set([item for list1 in feature_cache[sentence_idx] for list2 in list1 for item in list2]))
            
            for wt_idx in updatable_w:
                f_ind_mat = np.zeros([t_num,w_num])
                for i in range(0,t_num):
                    for j in range(0,w_num):
                        if wt_idx in feature_cache[sentence_idx][j][i]: f_ind_mat[i][j] = 1
                
                y_star = [tag_indexer.get_index(tag) for tag in sentences[sentence_idx].get_bio_tags()]
                
                sum_f = sum([ f_ind_mat[(y_star[i],i)] for i in range(0,len(y_star))])
                exp_sum_f = np.multiply(f_ind_mat,np.exp(gamma)).sum();

                grad = sum_f - exp_sum_f
                wt[wt_idx]=wt[wt_idx]+lr*grad
            
            
    return CrfNerModel(tag_indexer, feature_indexer, wt)
    



# Extracts emission features for tagging the word at word_index with tag.
# add_to_indexer is a boolean variable indicating whether we should be expanding the indexer or not:
# this should be True at train time (since we want to learn weights for all features) and False at
# test time (to avoid creating any features we don't have weights for).
def extract_emission_features(sentence, word_index, tag, feature_indexer, add_to_indexer):
    feats = []
    curr_word = sentence.tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    
    for idx_offset in xrange(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence):
            active_word = "</s>"
        else:
            active_word = sentence.tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence):
            active_pos = "</S>"
        else:
            active_pos = sentence.tokens[word_index + idx_offset].pos
        
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
        
    
    
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in xrange(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    
    
       
    for i in xrange(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    
    
    
    return np.asarray(feats, dtype=int)

def extract_emission_experiments_features(sentence, word_index, tag, feature_indexer, brown_clusters, clark_clusters, add_to_indexer):
    feats = []
    curr_word = sentence.tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    word_context=[]
    pos_context=[]
    for idx_offset in xrange(-2, 3):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence):
            active_word = "</s>"
        else:
            active_word = sentence.tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence):
            active_pos = "</S>"
        else:
            active_pos = sentence.tokens[word_index + idx_offset].pos
        word_context.append(active_word)
#         pos_context.append(active_pos)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    
#     maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":PosContext3=" + reduce(lambda x,y:x+","+y, pos_context[1:-1]))
#     maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":PosContext5=" + reduce(lambda x,y:x+","+y, pos_context))
        
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Context3=" + reduce(lambda x,y:x+","+y, word_context[1:-1]))
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Context5=" + reduce(lambda x,y:x+","+y, word_context))
#     maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Context5=" + reduce(lambda x,y:x+","+y, word_context))
    
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in xrange(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    
    
    
    tri_word_shape=[]
    for idx_offset in xrange(-1, 2):
        new_word = []
        if word_index + idx_offset < 0:
            new_word += "b"
        elif word_index + idx_offset >= len(sentence):
            new_word += "e"
        else:
            curr_word = sentence.tokens[word_index + idx_offset ].word
            for i in xrange(0, len(curr_word)):
                if curr_word[i].isupper():
                    new_word += "X"
                elif curr_word[i].islower():
                    new_word += "x"
                elif curr_word[i].isdigit():
                    new_word += "0"
                else:
                    new_word += "?"
        tri_word_shape.append(new_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape_s" +repr(idx_offset) +"="+repr(new_word))
    
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape_s-1s0="+repr(tri_word_shape[0:-1]))
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape_s0s1="+repr(tri_word_shape[1:]))
    
#     maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape_s-2s-1s0="+repr(tri_word_shape[0:3]))
#     maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape_s0s1s2="+repr(tri_word_shape[2:]))
    
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape_s-1s0s1="+repr(tri_word_shape))
#     maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape_s-2s-1s0s1s2="+repr(tri_word_shape))
    
    curr_word = sentence.tokens[word_index].word
    b_cluster="-1"
    if brown_clusters.has_key(curr_word): b_cluster =brown_clusters[curr_word]
    
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":b_cluster="+b_cluster)
    
    c_cluster="-1"
    curr_word = sentence.tokens[word_index].word.lower()
    if clark_clusters.has_key(curr_word): c_cluster =clark_clusters[curr_word]
    
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":c_cluster="+c_cluster)
    
    
    return np.asarray(feats, dtype=int)

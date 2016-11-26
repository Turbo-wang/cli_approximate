import numpy as np
import sys
import os
import scipy.stats as scist
import config
import json
import pickle
import cct_to_clique
import scipy.spatial.distance as distance


corpus_dir = config.corpusDir    


def process_test_file(file_name):
    contents = []
    with open(file_name) as f:
        for line in f:
            content = {}
            line = line.strip().split('\t')
            content['id'] = line[0]
            content['word1'] = line[1]
            content['pos_of_word1'] = line[2]
            content['word2'] = line[3]
            content['pos_of_word']  = line[4]
            content['word1_context'] = line[5]
            content['word2_context'] = line[6]
            content['score_human'] = line[7]
            contents.append(content)
    return contents


def read_word_sense_context(file_name, file_type):
    word_sen_context = {}
    with open(os.path.join(corpus_dir, file_name)) as f:
        if file_type == "json":
            word_sen_context = json.load(f)
        elif file_type == "pickle":
            word_sen_context = pickle.load(f)
    return word_sen_context


def get_word_label(word, context, sense_contexts, model_no_label):
    if len(sense_contexts) == 1:
        return -1
    context_vect = np.zeros(model_no_label.vector_size)
    len_con = 0
    for con in context and con in model_no_label:
           context_vec += model_no_label[con] 
           len_con += 1
    context_vec_avg = context_vec / len_con   
    sen_vec = []
    distance = sys.maxsize 
    len_con = 0
    can_label = 0
    for label, sense_context in sense_contexts.items():
        sense_vec = np.zeros(model_no_label)
        for con in sense_context and con in model_no_label:
               sense_vec += model_no_label[con]  
               len_con += 1
        sen_vec_avg = sense_vec / len_con        
        dis = distance.cosine(sen_vec_avg, context_vec_avg)
        if dis < distance:
            distance = dis
            can_label = label
    return can_label


def filter_context(word, context):
    can_sen = ""
    sens = context.split(".")
    for sen in sens:
        if "<b>" in sen:
            can_sen = sen
    



def build_score_list():
    contents = process_test_file("SCWS/rating.txt")
    word_sen_context = read_word_sense_context('test_word_sense_context', 'json')
    human_score_list = []
    compute_score_list = []
    for content in contents:
        human_score_list.append(content['score_human'])



def build_x1_x2(contents):
    x1 = []
    x2 = []



def spearman_rand(x1, x2):
    assert x1.shape == x2.shape
    return scist.spearmanr(x1, x2)[0]

if __name__ == '__main__':
    contents = process_test_file('SCWS/ratings.txt')
    print(contents)
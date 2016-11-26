#!/usr/bin/python
# _*_ coding:utf-8 _*_
from __future__ import print_function
import nltk
import numpy as np
import networkx as nx
from collections import Counter
from collections import defaultdict
from collections import OrderedDict
from itertools import combinations
import types
import re
import string
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from sklearn import manifold
import os
import time
import pickle


def get_txt_line(file_name):
    translator = str.maketrans({key: None for key in string.punctuation})
    with open(file_name) as f:
        for line in f:
            line = line.strip().lower()
            line = line.translate(translator)
            tmp = re.split('\.|!|\?', line)
            for se in tmp:
                if len(se.split()) > 60:
                    subtmp = re.split(",|;", se) 
                    for subse in subtmp:
                        sen = nltk.tokenize.wordpunct_tokenize(se)
                        yield sen
                else:
                    sen = nltk.tokenize.wordpunct_tokenize(se)
                    yield sen 


def get_pku_ZN_line(file_name):
    #     chi_punc_list = ["。", "？", "！", "，", "、", "；", "：", "“", "”",\
    # "’", "‘", "（", "）", "[", "]", "{", "}", "【", "】", "——", "······", "···", "－", "～", "《》", "〈〉"]
    sentences = []
    with open(file_name) as f:
        for line in f:
            sentence = []
            for key in line.strip().split():
                sentence.append(key)
            sentences.append(sentence)
    return sentences

def count_word_fre(sents_list):
    sents = []
    for sen in sents_list:
        sents += sen
    word_fre = Counter(sents)
    return word_fre


def build_sent_list(file_name):
    sent_gene = get_txt_line(file_name)
    sent_list = []
    for sen in sent_gene:
        sent_list.append(sen)
    return sent_list

def build_graph(file_name, word_list, word_fre, language=None):
    if language == "CHINESE":
        sentences_list = get_pku_ZN_line(file_name)
    else:
        sentences_list = build_sent_list(file_name)
    G = nx.Graph()
    # word_fre = count_word_fre(sentences_list)
    # words_list = list(word_fre)
    # print("grandson "+str(word_fre["grandson"]))
    print('sentences_list_len', len(sentences_list))
    word_list_len = len(word_fre)
    print('word num', len(word_fre))
    for sent in sentences_list:

        # for word in sent:
            # if word_fre[word] <=2:
            #     sent.remove(word)
        H = nx.Graph()
        edges = combinations(sent, 2)
        H.add_edges_from(edges)
        for edge in H.edges(): 
            word1 = edge[0]
            word2 = edge[1]
            PMI_det = 1 / (word_fre[word1] * word_fre[word2])
            if G.has_edge(word1, word2):
                G[word1][word2]['weight'] += PMI_det
            else:
                G.add_edge(word1, word2, weight=PMI_det)

    return G, word_fre


def PMI_filter(G, pmi_threshhold):
    for edge in G.edges():
        if G[edge[0]][edge[1]]['weight'] < pmi_threshhold:
            G.remove_edge(edge[0], edge[1])


def build_cct(G, dict_word, word_list, word_fre, alpha=0.05, beta=0.05, gamma=0.05):
    cct_dict = {}
    for word in G.nodes():
        w = {}
        nei = G.neighbors(word)
        nei = sorted(nei, key=lambda x: G[word][x]['weight'] * word_fre[word] * word_fre[x], reverse=True)
        n = len(nei)
        w[word] = nei[:round(n*alpha)]
        for c in w[word]:
            nei = G.neighbors(c)
            nei = sorted(nei, key=lambda x: G[c][x]['weight'] * word_fre[c] * word_fre[x], reverse=True)
            n = len(nei)
            w[c] = nei[:round(n*beta)]
        cct_dict[word] = w
    return cct_dict


def corpus_to_word2vec(sentens, file_name, save_flag,para):
    model = Word2Vec(sentens, size = para['size'], window=para['window'], min_count=para['min_count'], workers=para['workers'])
    if save_flag == True:
        model.save(file_name)
    return model


def save_cct(cct_file_name, cct_dict):
    with open(cct_file_name, 'w') as f:
        for key, value in cct_dict.items():
            f.write(key)
            f.write("\t")
            for nei in value[key]:
                f.write(nei)
                f.write("\t")
            f.write("\n")
            for nei in value[key]:
                f.write(nei)
                f.write("\t")
                for w in value[nei]:
                    f.write(w)
                    f.write("\t")
                f.write("\n")    
            f.write("\n")    

 
# if __name__ == '__main__':
    # yaml_file = 'graph_text0.1%_no_pmi_en_wiki'
    # G = nx.read_yaml(yaml_file)
    # dict_word = {}
    # word_list = []
    # #fine_name = "corpus/text0.01%.out"
    # fine_name = "test.txt"
    # W_i_N = []
    # G, word_fre = build_graph(fine_name)
    # i=0
    # for word in G.nodes():
    #     if word_fre[word] >=3:
    #         dict_word[word] = i
    #         word_list.append(word)
    #         i += 1
    # print('cct')
    # cct_dict = build_cct(G, dict_word, word_list, word_fre,1,1,1)
    # save_cct("test_cct", cct_dict)
    # sentens = build_sent_list("part_wiki201004.txt")
    # para={"size":100, "window"=10,"min_count"=3, "workers"=4}
    # corpus_to_word2vec(sentens, "part_wiki2010_word2vec", save_flag=True, para)



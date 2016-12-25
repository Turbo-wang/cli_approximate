#!/usr/bin/python3
# _*_ coding:utf-8 _*_
import nltk
from nltk.tokenize import word_tokenize
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
import config
from sklearn.cluster import KMeans
import types
import os
import json
import sys
import gensim
from sklearn.cluster import AffinityPropagation
from gensim.models import Word2Vec
import scipy.spatial.distance as distance



corpusDir = config.corpusDir

# read raw file line by line.
def get_txt_line(file_name):
    # translator = str.maketrans({key: None for key in string.punctuation})
    with open(file_name) as f:
        for line in f:
            line = line.strip().lower()
            line = re.sub("[\"#$%&()*+-/:<=>@[\\]^_`{|}~]", " ", line)
            # line = line.translate(translator)
            tmp = re.split('\.|!|\?', line)
            for se in tmp:
                if len(se.split()) > 60:
                    subtmp = re.split(",|;", se)
                    for subse in subtmp:
                        sen = word_tokenize(se)
                        yield sen
                else:
                    sen = word_tokenize(se)
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

#statistic word frequency from sentences list
def count_word_fre(sents_list):
    sents = []
    for sen in sents_list:
        sents += sen
    word_fre = Counter(sents)
    return word_fre

# get sentences list from raw file
def build_sent_list(file_name):
    sent_gene = get_txt_line(os.path.join(corpusDir, file_name))
    sent_list = []
    for sen in sent_gene:
        sent_list.append(sen)
    return sent_list


def PMI_filter(G, pmi_threshhold):
    for edge in G.edges():
        if G[edge[0]][edge[1]]['weight'] < pmi_threshhold:
            G.remove_edge(edge[0], edge[1])


def build_cct(matrix_graph, word_to_index, index_to_word, word_fre, alpha=0.05, beta=0.05, gamma=0.05):
    cct_dict = {}
    weight_ = 10 ** len(str(len(index_to_word)))
    for word in index_to_word:
        w = {}
        word_index = word_to_index[word]
        neighbors_word = []
        for key, value in matrix_graph.items():
            if key // weight_ == word_index:
                nei_index = key % weight_
                neighbors_word.append(index_to_word[nei_index])
        nei = sorted(neighbors_word, key=lambda x: matrix_graph[word_to_index[x] * weight_ + word_index] , reverse=True)
        n = len(nei)
        w[word] = nei[:round(n*alpha)]
        for c in w[word]:
            nei_word = []
            c_index = word_to_index[c]
            for key, value in matrix_graph.items():
                if key // weight_ == c_index:
                    nei_index = key % weight_
                    nei_word.append(index_to_word[nei_index])
            c_nei = sorted(nei_word, key=lambda x: matrix_graph[word_to_index[x] * weight_ + c_index], reverse=True)
            n = len(c_nei)
            w[c] = c_nei[:round(n*beta)]
        cct_dict[word] = w
    return cct_dict


def corpus_to_word2vec(sentens, file_name, save_flag,para):
    model = Word2Vec(sentens, size = para['size'], window=para['window'], min_count=para['min_count'], workers=para['workers'])
    if save_flag == True:
        model.save(file_name)
    return model


def save_cct(cct_file_name, cct_dict):
    with open(os.path.join(corpusDir, cct_file_name), 'w') as f:
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


def build_sparse_matrix_graph(sen_list, min_count, word_fre):
    word_to_index = {}
    index_to_word = []
    i = 0

    for sen in sen_list:
        for word in sen:
            if word_fre[word] >= min_count and word not in word_to_index:
                word_to_index[word] = i
                index_to_word.append(word)
                i += 1

    matrix_graph = {}
    weight_ = 10 ** len(str(len(index_to_word)))
    for sen in sen_list:
        edges = combinations(sen, 2)
        for edge in edges:
            word1, word2 = edge
            if word_fre[word1] < min_count or word_fre[word2] < min_count:
                break
            else:
                key_word_pair1 = word_to_index[word1] * weight_ + word_to_index[word2]
                key_word_pair2 = word_to_index[word2] * weight_ + word_to_index[word1]
                if key_word_pair1 in matrix_graph:
                    matrix_graph[key_word_pair1] += 1
                else:
                    matrix_graph[key_word_pair1] = 1
                if key_word_pair2 in matrix_graph:
                    matrix_graph[key_word_pair2] += 1
                else:
                    matrix_graph[key_word_pair2] = 1
    return matrix_graph, word_to_index, index_to_word


def read_cct(file_name):
    cct_map = {}
    with open(os.path.join(corpusDir, file_name)) as f:
        another = False
        key_line = True
        w = {}
        key_W = ""
        for line in f:
            # print(len)
            line_con = line.strip().split()
            if len(line_con) == 0:
                another = True
                key_line = True
                cct_map[key_W] = w
                w = {}
                key_W = ""
            else:
                if key_line == True:
                    key_W = line_con[0]
                    key_line == False
                nei = line_con[1:]
                w[line_con[0]] = nei
                key_line = False
    return cct_map


def build_graph(content, word):
    G = nx.Graph(   )
    # nei_map = cct[word]
    G.add_node(word)
    for key, value in content.items():
        for nei in value:
            G.add_edge(key, nei)
    nei = G.neighbors(word)
    remove_nodes = []
    for node in G.nodes():
        if node not in nei:
            remove_nodes.append(node)
    G.remove_nodes_from(remove_nodes)
    return G


def constr_cliq_and_sense_con(word_cliques_dict, model):
    word_sense_context = {}
    for word, cliques in word_cliques_dict.items():
        word_sense_context[word] = cluster_kmeans_clique(word, cliques, model, 2)
    return word_sense_context



def construct_clique(cct, clique_file_name):
    word_cliques_dict = {}
    for word, content in cct.items():
        graph = build_graph(content, word)
        cliques = list(nx.clique.find_cliques(graph))
        word_cliques_dict[str(word)] = cliques
    save_json(word_cliques_dict, clique_file_name)
    return word_cliques_dict


def save_json(json_obj, file_name):
    with open(os.path.join(corpusDir, file_name), 'w') as f:
        json.dump(json_obj, f)


def corpus_to_word2vec(sentens, file_name, save_flag, para):
    model = gensim.models.Word2Vec(sentens, size = para['size'], window=para['window'], min_count=para['min_count'], workers=para['workers'])
    if save_flag == True:
        model.save(os.path.join(corpusDir, file_name))
    return model


def cluster_kmeans_clique(word, word_clique, model, n_cluster):
    vector_clique = []
    if len(word_clique) == 0:
        return []
    sense_context = {}
    if len(word_clique) == 1:
        sense_context['0'] = word_clique[0]
        return sense_context
    for clique in word_clique:
        contexts = clique
        con_vector = np.zeros(model.vector_size)
        len_con = 0
        for con in contexts:
            if con in model:
                len_con += 1
                con_vector += model[con]
        con_vector_avg = con_vector / len_con
        vector_clique.append(con_vector_avg)
    #af = AffinityPropagation(preference=-5).fit(vector_clique)
    kmeans = KMeans(n_clusters = n_cluster, random_state=0).fit(vector_clique)
    for index, vector in enumerate(vector_clique):
        #assert isinstance(vector,list)
        #assert isinstance(vector[0],int)
        #assert len(vector) == 100
        label_ = kmeans.predict([vector])[0]
        #print(label_)
        #label = str(label_)
        if label_ not in sense_context:
            sense_context[str(label_)] = set()
        sense_context[str(label_)].update(word_clique[index])
    for key, value in sense_context.items():
        sense_context[key] = list(value)
    # sense_context.remove(word)
    return sense_context


def cluster_ap_clique(word, word_clique, model):
    vector_clique = []
   # print(word_clique)
    if len(word_clique) == 0:
        return []
    sense_context = {}
    if len(word_clique) == 1:
        sense_context[0] = word_clique[0]
        return sense_context
    for clique in word_clique:
        contexts = clique
        con_vector = np.zeros(model.vector_size)
        for con in contexts:
            if con not in model:
                continue
            con_vector += model[con]
        con_vector_avg = con_vector / len(contexts)
        vector_clique.append(con_vector_avg)
    vector_clique = np.asarray(vector_clique, dtype='float32')
    #print(vector_clique)
    print(vector_clique.shape)
    print(word)
    af = AffinityPropagation().fit(vector_clique)
#    sense_context = {}
    for index, vector in enumerate(vector_clique):
        #assert vector.shape == (100,)
        #assert isinstance(vector,list)
        #assert isinstance(vector[0],int)
        #assert len(vector) == 100
    #    print(vector)
        label_ = af.predict(list(vector))[0]
        #label_ = 0
    #    print(type(label_))
        if label_ not in sense_context:
       # if len(sense_context[label_]) == 0:
            sense_context[label_] = set()
        #sense_context[label_].update(word_clique)
    # sense_context.remove(word)
    return sense_context


def label_corpus(sentens, word_sense_context, model, file_name):
    # new_sentens = []
    with open(os.path.join(corpusDir, file_name), 'w') as f:
        for senten in sentens:
            # new_senten = []
            for word in senten:
                new_word = ""
                if word not in word_sense_context or word_sense_context[word] == {} or word_sense_context[word] == []:
                    # new_senten.append(word)
                    new_word = word
                else:
                    context = senten
                    sense_context = word_sense_context[word]
                    if len(sense_context) == 1:
                        new_word = word
                    else:
                        context_vec = np.zeros(model.vector_size)
                        len_con = 0
                        for con in senten:
                            if con in model and con != word:
                                context_vec += model[con]
                                len_con += 1
                        context_vec_avg = context_vec / len_con
                        label = "0"
                        mindistance = sys.maxsize
                        for key, value in sense_context.items():
                            vector = np.zeros(model.vector_size)
                            len_con = 0
                            for wo in value:
                                if wo in model:
                                    vector += model[wo]
                                    len_con += 1
                            vector_avg = vector / len_con
                            dis = distance.cosine(vector_avg, context_vec_avg)
                            if mindistance > dis:
                                mindistance = dis
                                label = key
                        # new_senten.append(word+"_"+str(label))
                        new_word = word + "_" + str(label)
                f.write(new_word)
                f.write("\t")
            f.write("\n")
            # new_sentens.append(new_senten)


def load_model_file(file_name):
    path = os.path.join(corpusDir, file_name)
    model = Word2Vec.load(path)
    return model


def load_json_file(file_name):
    path = os.path.join(corpusDir, file_name)
    with open(path) as f:
        json_obj = json.load(f)
    return json_obj


def main():
    raw_file_name = config.file_name
    cct_file_name = config.cct_file_name
    clique_file_name = config.clique_file_name
    sen_list = build_sent_list(raw_file_name)
    print("current file_name is ", raw_file_name)
    word_fre = count_word_fre(sen_list)
    time_start = time.time()
    (matrix_graph, word_to_index, index_to_word) = build_sparse_matrix_graph(sen_list, 1, word_fre)
    # print(matrix_graph)
    save_json(matrix_graph, config.sparse_matrix_graph_file_name)
    with open(os.path.join(corpusDir, config.index_word_file_name), 'w') as f:
        for word in index_to_word:
            f.write(word)
            f.write("\n")
    print("build matrix time:")
    print(str(time.time()-time_start))
    # print(index_to_word)
    time_start = time.time()
    cct = build_cct(matrix_graph, word_to_index, index_to_word, word_fre, alpha=0.15, beta=0.15, gamma=0.15)
    print("build cct time:")
    print(str(time.time()-time_start))
    time_start = time.time()
    save_cct(cct_file_name, cct)
    print("save cct time:")
    print(str(time.time()-time_start))

    # cct = read_cct(cct_file_name)
    model = load_model_file(config.model_file_name)
    time_start = time.time()
    word_cliques_dict = construct_clique(cct, clique_file_name)
    print("find all cliques time:")
    print(str(time.time()-time_start))

    time_start = time.time()
    word_sense_context = constr_cliq_and_sense_con(word_cliques_dict, model)
    print("cluster sense for word time:")
    print(str(time.time()-time_start))
    save_json(word_sense_context, config.word_sense_dict_file_name)
    time_start = time.time()
    label_corpus(sen_list, word_sense_context, model, config.label_file_name)
    print("label corpus time:")
    print(str(time.time()-time_start))


if __name__ == '__main__':
    main()
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


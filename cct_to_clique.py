#from __future__ import print_function
from sklearn.cluster import KMeans
import types
import plour
import config
import networkx as nx
import time
import os
import json
import sys
import numpy as np
import gensim
from sklearn.cluster import AffinityPropagation
from gensim.models import Word2Vec
import scipy.spatial.distance as distance


corpusDir = config.corpusDir


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
                if word not in word_sense_context or word_sense_context[word] == {}:
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
    sentens = plour.build_sent_list(raw_file_name)
    cct_file_name = config.cct_file_name
    cct = read_cct(cct_file_name)
    model = load_model_file(config.model_file_name)
    clique_file_name = config.clique_file_name
    word_cliques_dict = construct_clique(cct, clique_file_name)
    word_sense_context = constr_cliq_and_sense_con(word_cliques_dict, model)
    save_json(word_sense_context, config.word_sense_dict_file_name)
    label_corpus(sentens, word_sense_context, model, config.label_file_name)

if __name__ == '__main__':
    main()
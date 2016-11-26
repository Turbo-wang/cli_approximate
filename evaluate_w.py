import os
from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf


def build_bnlm(srilm_path, file_path, order):
    os.system(srilm_path + "ngram-count -text " + file_path + " -order "+ order + " -write trainfile.count")
    os.system(srilm_path + "ngram-count -read trainfile.count -order " + order + " -lm  trainfile.lm -interpolate -kndiscount")


def gene_x_y(file_path, word_vector_path, order):
    with open(file_path) as f:



def build_cslm(word_vector_path, order):
    with open(word_vector_path) as f:
        base_para = f.readline().strip()
    word_num, dimension = base_para.split()

    # Parameters
    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100
    display_step = 1

    # Network Parameters
    n_hidden_1 = 256 # 1st layer number of features
    n_hidden_2 = 256 # 2nd layer number of features
    n_input = dimension * (order - 1)
    n_classes = word_num

    x = tf.placeholder("float", [None, n_input])
    # y = tf.placeholder("float", [None, n_output])
    
    # Create model
    def multilayer_perceptron(x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_laye

    def perplexity()
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = multilayer_perceptron(x, weights, biases)
    cost = -tf.reduce_mean(tf.log(tf.nn.softmax(pred)))
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(10):



def update_bnlm_by_cslm(lm_path)


if __name__ == '__main__':

    main()
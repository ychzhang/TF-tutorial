'''
Author: Yichi Zhang
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5, help='one epoch means feed all data into the train procedure one time')
parser.add_argument('--batch_size', type=int, default=64, help='number of images in one mini batch')

def data():
    # Prepare dataset. Format here: numpy array.
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print("The shape of x_train is: {}".format(x_train.shape))
    print("The shape of y_train is: {}".format(y_train.shape))
    print("Data type of the image is: {}".format(x_train.dtype))
    print("Data type of the label is: {}".format(y_train.dtype))
    return x_train, y_train, x_test, y_test

def model(inputs):
    flatten_inputs = tf.layers.flatten(inputs)
    fc0 = tf.contrib.layers.fully_connected(inputs=flatten_inputs, num_outputs=512, activation_fn=tf.nn.relu, scope='fc0')
    fc1 = tf.contrib.layers.fully_connected(inputs=fc0, num_outputs=10, activation_fn=tf.nn.softmax, scope='fc1')
    return fc1

def loss_func(ground_truth_labels, logits):
    onehot_label = tf.one_hot(indices=ground_truth_labels, depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_label, logits=logits, scope='loss_func')
    return loss

def opt_tool(loss, global_step):
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.005, momentum=0.9)
    opt_operation = optimizer.minimize(loss=loss, global_step=global_step)
    return opt_operation

def evaluation_metric(logits, ground_truth_label):
    # both arguments of this function should be numpy array
    prediction = np.argmax(logits, axis=1)
    accuracy = np.sum(prediction == ground_truth_label) / ground_truth_label.shape[0]
    return accuracy

def main(unused_argv):
    ''' Set visible GPU device '''
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
    
    ''' Step1: build the model '''
    inputs = tf.placeholder(dtype=tf.float64, shape=[None, 28, 28])
    true_labels = tf.placeholder(dtype=tf.int32, shape=[None])
    logits = model(inputs)

    ''' Step2: construct the loss function '''
    loss = loss_func(true_labels, logits)
    
    ''' Step3: create optimization operation '''
    global_step = tf.train.get_or_create_global_step()
    opt_operation = opt_tool(loss, global_step)

    ''' Step4: prepare dataset '''
    x_train, y_train, x_test, y_test = data()
    num_image = x_train.shape[0]

    ''' Step5: train '''
    batchsize = FLAGS.batch_size
    num_epoch = FLAGS.epochs
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epoch):
            for i in range(num_image//batchsize):
                logits_, loss_,  _ = sess.run( 
                                                [logits, loss,  opt_operation],
                                                feed_dict = { 
                                                                inputs: x_train[i*batchsize : (i+1)*batchsize, :, :],
                                                                true_labels: y_train[i*batchsize : (i+1)*batchsize] 
                                                            } 
                                             )
                logits_test = sess.run(logits, 
                                       feed_dict = { inputs: x_test } )
                train_accuracy = evaluation_metric(logits_, y_train[i*batchsize : (i+1)*batchsize])
                test_accuracy = evaluation_metric(logits_test, y_test)
                print("epoch {}    batch {}    training loss = {:.3f}    training accuracy = {:.3f}    test accuracy = {:.3f}".format(
                    epoch+1, i+1, loss_, train_accuracy, test_accuracy) )

if __name__ == '__main__':
    ''' entry point of the program '''
    FLAGS, unparsed = parser.parse_known_args()
    ''' run the main function with passed in arguments '''
    tf.app.run(argv=[sys.argv[0]] + unparsed)


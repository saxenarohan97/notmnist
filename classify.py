import os
from tqdm import tqdm
import progressbar as pb
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pprint import pprint
import tensorflow as tf

train_features = np.load('train_features.npy')
train_labels = np.load('train_labels.npy')

valid_features = np.load('validate_features.npy')
valid_labels = np.load('validate_labels.npy')

test_features = np.load('test_features.npy')
test_labels = np.load('test_labels.npy')

features = tf.placeholder(dtype = tf.float32)
labels = tf.placeholder(dtype = tf.float32)

weights = tf.Variable(tf.truncated_normal(shape = [784, 10]))
bias = tf.Variable(tf.zeros(shape = [10]))

logits = tf.matmul(features, weights) + bias

prediction = tf.nn.softmax(logits)

cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)

loss = tf.reduce_mean(cross_entropy)

tf.summary.scalar('Loss', loss)

train_feed_dict = {features: train_features, labels: train_labels}
valid_feed_dict = {features: valid_features, labels: valid_labels}
test_feed_dict = {features: test_features, labels: test_labels}

is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))

accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

tf.summary.scalar('Accuracy', accuracy)

epochs = 10000
learning_rate = 0.01

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

validation_accuracy = 0.0
previous_accuracy = 0.

bar = pb.ProgressBar(max_value = epochs, redirect_stdout = True)

saver = tf.train.Saver()

with tf.Session() as session:

    try:

        saver.restore(session, './tmp/progress.ckpt')
        bar_count = 0

        for epoch_i in range(epochs):

                bar_count += 1

                previous_accuracy = session.run(accuracy, feed_dict = valid_feed_dict) * 100

                session.run(optimizer, feed_dict = train_feed_dict)
                print("Train: ", end = "")
                print('{:.5f}'.format(session.run(loss, feed_dict = train_feed_dict)), end = ' | ')
                print("Valid: ", end = "")
                print('{:.5f}'.format(session.run(loss, feed_dict = valid_feed_dict)), end = ' | ')
                print("Test Accuracy: ", end = "")

                acc = session.run(accuracy, feed_dict = test_feed_dict) * 100

                if acc < previous_accuracy:

                    print('-----------ACCURACY HAS DECREASED-----------')
                    print(acc)
                    break

                print(acc)

                bar.update(bar_count)

    except KeyboardInterrupt:
        saver.save(session, './tmp/progress.ckpt')
        print('Progress saved to disk')

    else:
        saver.save(session, './tmp/progress.ckpt')
        print('Progress saved to disk')

import os
from tqdm import tqdm
import progressbar as pb
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pprint import pprint
import tensorflow as tf

image = mpimg.imread('./tmp/pic.png')
image = np.resize(image, [1, 784])

weights = tf.Variable(tf.truncated_normal(shape = [784, 10]))
bias = tf.Variable(tf.zeros(shape = [10]))

logits = tf.matmul(image, weights) + bias

prediction = tf.nn.softmax(logits)

index = tf.argmax(prediction, axis=1)

key = {0: 'a', 1: 'b', 2:'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j'}

saver = tf.train.Saver()

with tf.Session() as sess:

    saver.restore(sess, './tmp/progress.ckpt')

    print(key[sess.run(index)[0]])

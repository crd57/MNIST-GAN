# -*- coding: utf-8 -*-
# @Time    : 2018/7/8 16:05
# @Author  : Chen Ruida
# @Email   : crd57@outlook.com
# @File    : GAN1.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data

mb_size = 100
Z_dim = 100
CLIP = [-0.01, 0.01]

mnist = input_data.read_data_sets('D:\CODE\MNIST_GAN\MNIST_data', one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
YY_0 = labels[labels[:, 0] == 1][0:10]
YY_1 = labels[labels[:, 1] == 1][0:10]
YY_2 = labels[labels[:, 2] == 1][0:10]
YY_3 = labels[labels[:, 3] == 1][0:10]
YY_4 = labels[labels[:, 4] == 1][0:10]
YY_5 = labels[labels[:, 5] == 1][0:10]
YY_6 = labels[labels[:, 6] == 1][0:10]
YY_7 = labels[labels[:, 7] == 1][0:10]
YY_8 = labels[labels[:, 8] == 1][0:10]
YY_9 = labels[labels[:, 9] == 1][0:10]
YY = np.vstack([YY_0, YY_1, YY_2, YY_3, YY_4, YY_5, YY_6, YY_7, YY_8, YY_9])


def weight_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))


# discriminater net

X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y")

D_W1_x = weight_var([784, 128], 'D_W1_x')
D_b1_x = bias_var([128], 'D_b1_X')
D_W1_y = weight_var([10, 128], 'D_W1_y')
D_b1_y = bias_var([128], 'D_b1_y')

D_W2 = weight_var([256, 128], 'D_W2')
D_b2 = bias_var([128], 'D_b2')
D_W3 = weight_var([128, 1], 'D_W3')
D_b3 = bias_var([1], 'D_b3')

theta_D = [D_W1_x, D_b1_x, D_W1_y, D_b1_y, D_W2, D_b2, D_W3, D_b3]

# generator net

Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

G_W1_z = weight_var([100, 128], 'G_W1_z')
G_b1_z = bias_var([128], 'G_B1_z')

G_W1_y = weight_var([10, 128], 'G_W1_y')
G_b1_y = bias_var([128], 'G_B1_y')

G_W2 = weight_var([256, 512], 'G_W3')
G_b2 = bias_var([512], 'G_B2')

G_W3 = weight_var([512, 784], 'G_W2')
G_b3 = bias_var([784], 'G_B2')

theta_G = [G_W1_z, G_b1_z, G_b2, G_W1_y, G_b1_y, G_W2]


def generator(z, y):
    G_h1_z = tf.nn.relu(tf.matmul(z, G_W1_z) + G_b1_z)
    G_h1_y = tf.nn.relu(tf.matmul(y, G_W1_y) + G_b1_y)
    G_h1 = tf.concat(axis=1, values=[G_h1_z, G_h1_y])
    G_h2 = tf.nn.relu(tf.matmul(G_h1,G_W2)+G_b2)
    G_log_prob = tf.matmul(G_h2, G_W3) + G_b3
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x, y):
    D_h1_x = tf.nn.relu(tf.matmul(x, D_W1_x) + D_b1_x)
    D_h1_y = tf.nn.relu(tf.matmul(y, D_W1_y) + D_b1_y)
    D_h1 = tf.concat(axis=1, values=[D_h1_x, D_h1_y])
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit


G_sample = generator(Z, y_)
D_real, D_logit_real = discriminator(X, y_)
D_fake, D_logit_fake = discriminator(G_sample, y_)

# D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#     logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
# D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#     logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
# D_loss = D_loss_real + D_loss_fake
# G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#     logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_loss_real = tf.reduce_mean(tf.scalar_mul(-1.0, D_logit_real))
D_loss_fake = tf.reduce_mean(D_logit_fake)
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.scalar_mul(-1.0, D_logit_fake))

D_optimizer = tf.train.RMSPropOptimizer(0.001).minimize(D_loss, var_list=theta_D)
G_optimizer = tf.train.RMSPropOptimizer(0.001).minimize(G_loss, var_list=theta_G)
clip_D_op = [var.assign(tf.clip_by_value(var, CLIP[0], CLIP[1])) for var in theta_D]

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1.0, 1.0, size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):  # [i,samples[i]] imax=16
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(10000000):
    if it % 100 == 0:
        samples = sess.run(G_sample, feed_dict={
            Z: sample_Z(mb_size, Z_dim), y_: YY})  # 16*784
        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
    X_mb, y_mb = mnist.train.next_batch(mb_size)

    _, G_loss_curr = sess.run([G_optimizer, G_loss], feed_dict={
            Z: sample_Z(mb_size, Z_dim), y_: y_mb})
    if it % 2 == 0:
        _, D_loss_curr = sess.run([D_optimizer, D_loss], feed_dict={
                X: X_mb, Z: sample_Z(mb_size, Z_dim), y_: y_mb})

    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()

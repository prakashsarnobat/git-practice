# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 00:04:31 2019

@author: Sridhar Sanobat
"""

BOOKS = []

# prepare data
 R = np.array(BOOKS)
 N = len(BOOKS)
 M = len(BOOKS[0])
 K = 2 # number of hidden features
 P = np.random.rand(N,K)
 Q = np.random.rand(M,K)

# input placeholders
 ratings = tf.placeholder(tf.float32, name = 'ratings')

# model variables
 tP = tf.Variable(P, dtype=tf.float32, name='P')
 tQ = tf.Variable(Q, dtype=tf.float32, name='Q')

# build model
 pMultq = tf.matmul(tP, tQ, transpose_b=True)

squared_deltas = tf.square(pMultq - ratings)
 loss = tf.reduce_sum(squared_deltas)
 tf.summary.scalar('loss', loss)
 tf.summary.scalar('sumP', tf.reduce_sum(tP))
 tf.summary.scalar('sumQ', tf.reduce_sum(tQ))
import tensorflow as tf

"""
Graph 用来隔离 张量和计算，类似于命名空间
"""

g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v", initializer=tf.zeros_initializer(shape=[1]))

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", initializer=tf.zeros_initializer(shape=[1]))

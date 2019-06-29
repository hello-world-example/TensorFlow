import tensorflow as tf
import os
import shutil

"""
https://www.tensorflow.org/guide/summaries_and_tensorboard

tensorboard --logdir=9.1.tensor_board.log.d/
"""

# 定义一个简单的计算图，实现向量加法操作
input1 = tf.constant([1, 2, 3], dtype=tf.float32, name='input1')
input2 = tf.Variable(tf.random_uniform([3]), name='input2')
output = tf.add_n([input1, input2], name='add')

writer = tf.summary.FileWriter("9.1.tensor_board.log.d/", tf.get_default_graph())
writer.close()

"""
tensorboard --logdir=9.1.tensor_board.log.d/
"""
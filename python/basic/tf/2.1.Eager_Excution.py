import tensorflow as tf

"""
Session(会话)方式，静态图

Eager Excution(即时执行)，动态图

Eager Excution 与 Session 最大的区别就是，不需要构建好整个运算图，再统一执行；而是一边构建一边执行，这也标志着TensorFlow正式支持日益火爆的动态图
"""
tf.enable_eager_execution()

print(" TF version:{}".format(tf.VERSION))
print(" eager exec:{}".format(tf.executing_eagerly()))

x = [2]
y = [3]
print(tf.add(x, y))

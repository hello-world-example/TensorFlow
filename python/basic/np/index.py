import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
设置模型的参数，定义学习率、模型的迭代次数以及迭代多少次计算一次模型损失
"""
# 学习率
learning_rate = 0.01
# 模型的迭代次数
training_epochs = 5000
# 迭代多少次计算一次模型损失
display_step = 50

# Training Data
train_X = np.asarray(
    [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]
)

train_Y = np.asarray(
    [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]
)

data_len = train_X.shape[0]

print(train_X)
print(train_Y)
print(data_len)

# 定义输入参数名
xx = tf.placeholder(tf.float32)
yy = tf.placeholder(tf.float32)

# 定义 权重 和 偏差（初始值是随机数）
W = tf.Variable(np.random.randn(), name="weight")
B = tf.Variable(np.random.randn(), name="bias")

# 构建线性模型 「 yy = xx * W + B 」
linear_model = tf.add(tf.multiply(xx, W), B)

# Mean squared error 均方误差
cost = tf.reduce_sum(tf.pow(linear_model - yy, 2)) / (2 * data_len)
# Gradient descent 梯度下降
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# 开始训练
with tf.Session() as sess:
    # 执行初始化操作
    sess.run(init)

    # 拟合模型数据
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={xx: x, yy: y})

        # 每50次迭代后在控制台输出模型当前训练的loss以及权重大小
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={xx: train_X, yy: train_Y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(B))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={xx: train_X, yy: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(B), '\n')

    # 画出拟合图像
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(B), label='Fitted line')
    plt.legend()
    plt.show()

# 创建测试数据
test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

print("Testing... (Mean square loss Comparison)")
testing_cost = sess.run(
    tf.reduce_sum(tf.pow(linear_model - yy, 2)) / (2 * test_X.shape[0]),
    feed_dict={xx: test_X, yy: test_Y})  # same function as cost above
print("Testing cost=", testing_cost)
print("Absolute mean square loss difference:", abs(
    training_cost - testing_cost))

plt.plot(test_X, test_Y, 'bo', label='Testing data')
plt.plot(train_X, sess.run(W) * train_X + sess.run(B), label='Fitted line')
plt.legend()
plt.show()

import tensorflow as tf

a = tf.constant([1, 2], dtype=tf.float32, name='a')
b = tf.constant([2, 3], dtype=tf.float32, name='b')

print(a, ":::", b)

result = a + b
print(result)

# 创建一个会话
sess = tf.Session()
# 使用这个创建好的会话来得到关系的运算结果
run_result = sess.run(result)
# 打印结果
print(run_result)
# 关闭会话(缺点，这种关闭会话的方式在发生异常的时候可能不会执行)
sess.close()

print()
print("以下通过 Python 上下文管理器管理会话，类似于 Java 中 try...resources 语句")
print()

with tf.Session() as sess:
    # 使用这个创建好的会话来得到关系的运算结果
    run_result = sess.run(result)
    # 打印结果
    print(run_result)
    # 最后自动释放资源，不用显式调用 sess.close()



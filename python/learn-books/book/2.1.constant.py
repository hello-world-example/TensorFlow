import tensorflow as tf

a = tf.constant([1, 2], dtype=tf.float32, name='a')
b = tf.constant([2, 3], dtype=tf.float32, name='b')

print(a, ":::", b)

result = a + b
print(result)

sess = tf.Session()
run_result = sess.run(result)

print(run_result)

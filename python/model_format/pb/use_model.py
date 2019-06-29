import tensorflow as tf

sess = tf.Session()

with open('../src/main/resources/models/invode01-model.pb', 'rb') as file:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(file.read())
    sess.graph.as_default()
    # 导入计算图
    tf.import_graph_def(graph_def, name='')

# 需要有一个初始化的过程
sess.run(tf.global_variables_initializer())

# 需要先复原变量
print(sess.run('y:0'))
# 1

# 输入
input_x = sess.graph.get_tensor_by_name('x:0')

z = sess.graph.get_tensor_by_name('z:0')

ret = sess.run(z, feed_dict={input_x: 5})

print(ret)

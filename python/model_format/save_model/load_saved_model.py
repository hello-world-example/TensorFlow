import tensorflow as tf

export_dir = "./hello_model"

graph = tf.Graph()

with tf.Session(graph=graph) as sess:
    # 加载模型
    loader_load = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
    print(loader_load)

    #
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")

    # 调用模型进行运算，参数 x 设置为 7.0， 结果写入变量 y
    result = sess.run(y, feed_dict={x: 7.})
    # 打印模型计算的结果结果 14.0
    print(result)

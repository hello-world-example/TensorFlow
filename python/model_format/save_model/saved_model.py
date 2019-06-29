import tensorflow as tf

"""
saved_model_cli show --dir ./hello_model/ --all
"""

# 定义权重变量 W，初始值是 2.0
W = tf.get_variable('W', initializer=tf.constant(2.0), dtype=tf.float32)

# 定义一个占位符 x,
x = tf.placeholder(tf.float32, name='x')

# OP， y = W * x
y = tf.multiply(W, x, name='y')

with tf.Session() as sess:
    # 初始化所有变量
    sess.run(tf.global_variables_initializer())
    # 给 x 赋值 并运行
    print(sess.run(y, {x: 10.0}))

    # 已 save model 形式保存
    tf.saved_model.simple_save(
        sess,
        # 模型保存路径
        "./hello_model",
        # 指名输入变量
        inputs={"param_x": x},
        # 指名输出变量
        outputs={"result_y": y}
    )

# SAVE THE MODEL
# builder = tf.saved_model.builder.SavedModelBuilder("./hello_model")
#
# # 定义签名
# signature_def = tf.saved_model.predict_signature_def(
#     inputs={"x", x},
#     outputs={"result", tf.identity(y, name="result")}
# )
#
# builder.add_meta_graph_and_variables(
#     sess,
#     [
#         tf.saved_model.tag_constants.SERVING
#     ],
#     signature_def_map={"predict", signature_def}
# )
# builder.save()

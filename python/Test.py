import tensorflow as tf
import datetime as dt

# 定义权重变量 W，初始值是 2.0
W = tf.get_variable('W', initializer=tf.constant(2.0), dtype=tf.float32)

# 定义一个占位符 x,
x = tf.placeholder(tf.float32, name='x')

# OP， y = W * x
y = tf.multiply(W, x, name='y')

with tf.Session() as sess:
    # 初始化所有变量
    sess.run(tf.global_variables_initializer())

    # 已 saved model 形式保存
    tf.saved_model.simple_save(
        sess,
        # 模型保存路径
        "/tmp/tf_models/hello_model/" + str(int(dt.datetime.now().timestamp())),
        # 指名输入变量
        inputs={"param_x": x},
        # 指名输出变量
        outputs={"result_y": y}
    )
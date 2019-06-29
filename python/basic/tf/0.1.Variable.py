import tensorflow as tf

tf_constant = tf.constant(
    [
        [1, 2]
    ]
)

with tf.Session() as session:
    session_run = session.run(tf_constant)
    print(session_run)

import os
import pandas as pd
import tensorflow as tf

FUTURES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# /Users/kail/IdeaProjects/github/hello-world-example/tensorflow/python/iris/hello_iris.py
print(__file__)
# /Users/kail/IdeaProjects/github/hello-world-example/tensorflow/python/iris/hello_iris.py
print(os.path.realpath(__file__))
# /Users/kail/IdeaProjects/github/hello-world-example/tensorflow/python/iris
print(os.path.dirname(os.path.realpath(__file__)))

# 获取当前 py 脚本的文件夹
dir_path = os.path.dirname(os.path.realpath(__file__))
# 拼接文件路径
# /Users/kail/IdeaProjects/github/hello-world-example/tensorflow/python/iris/data/iris_training.csv
train_path = os.path.join(dir_path, 'data/iris_training.csv')
# /Users/kail/IdeaProjects/github/hello-world-example/tensorflow/python/iris/data/iris_test.csv
test_path = os.path.join(dir_path, 'data/iris_test.csv')

train_data = pd.read_csv(train_path, names=FUTURES, header=0)
train_x, train_y = train_data, train_data.pop('Species')

test_data = pd.read_csv(test_path, names=FUTURES, header=0)
test_x, test_y = test_data, test_data.pop('Species')

# 特征列
feature_columns = []
for key in train_x.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))

print(test_x.head())
print(test_y.head())

print("dict(test_x)")
print(dict(test_x))
print("dict(test_y)")
print(dict(test_y))
# exit(0)

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    # 设定深层神经网络分类器的复杂程度
    hidden_units=[10, 10],
    # 对应三种花朵类型
    n_classes=3
)


def train_input_fn(features, labels, batch_size):
    """

    :param features:
    :param labels:
    :param batch_size:
    :return:
    """
    features = dict(features)
    inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


tf.logging.set_verbosity(tf.logging.INFO)

# 训练
classifier.train(
    input_fn=lambda: train_input_fn(features=train_x, labels=train_y, batch_size=100),
    steps=100
)


# exit(0)


def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


# 评估
eval_result = classifier.evaluate(
    input_fn=lambda: eval_input_fn(test_x, test_y, 100)
)
# 打印评估结果
for key, value in eval_result.items():
    print(key, ":", value)


# ================================================================
# ================================================================
# ================================================================


def serving_input_receiver_fn2():
    receiver_tensors = {
        'SepalLength': tf.placeholder(tf.float32, [None, 1]),
        'SepalWidth': tf.placeholder(tf.float32, [None, 1]),
        'PetalLength': tf.placeholder(tf.float32, [None, 1]),
        'PetalWidth': tf.placeholder(tf.float32, [None, 1]),
    }
    # feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)

    # https://github.com/tensorflow/tensorflow/issues/12508
    # curl -d '{"signature_name": "predict", "instances": [{"SepalLength":[5.1],"SepalWidth":[3.3],"PetalLength":[1.7],"PetalWidth":[0.5]}]}'   -X POST http://localhost:8501/v1/models/iris:predict
    # curl -d '{"signature_name": "serving_default", "examples": [{"SepalLength":[5.1],"SepalWidth":[3.3],"PetalLength":[1.7],"PetalWidth":[0.5]}]}'   -X POST http://localhost:8501/v1/models/iris:classify
    return tf.estimator.export.ServingInputReceiver(features=receiver_tensors, receiver_tensors=receiver_tensors)


def serving_input_receiver_fn3():
    serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None], name='input_tensors')
    receiver_tensors = {"predictor_inputs": serialized_tf_example}

    feature_spec = {"words": tf.FixedLenFeature([25], tf.int64)}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


# region ================================================================
# ================================================================ https://medium.com/@yuu.ishikawa/introduction-to-restful-api-with-tensorflow-serving-9c60969b5b95
# ================================================================
# INPUT_FEATURE = 'x'
# NUM_CLASSES = 3
#
#
# def serving_input_receiver_fn():
#     """
#     This is used to define inputs to serve the model.
#     :return: ServingInputReciever
#     """
#     receiver_tensors = {
#         'sepal_length': tf.placeholder(tf.float32, [None, 1]),
#         'sepal_width': tf.placeholder(tf.float32, [None, 1]),
#         'petal_length': tf.placeholder(tf.float32, [None, 1]),
#         'petal_width': tf.placeholder(tf.float32, [None, 1]),
#     }
#
#     # Convert give inputs to adjust to the model.
#     features = {
#         INPUT_FEATURE: tf.concat([
#             receiver_tensors['sepal_length'],
#             receiver_tensors['sepal_width'],
#             receiver_tensors['petal_length'],
#             receiver_tensors['petal_width']
#         ], axis=1)
#     }
#     return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors,
#                                                     features=features)


# endregion ================================================================
# ================================================================
# ================================================================


def serving_input_receiver_fn4():
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    return tf.contrib.learn.build_parsing_serving_input_fn(feature_spec)()


def serving_input_receiver_fn5():
    inputs = {'x': tf.placeholder(tf.float32, [4])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


# 导出 [TensorFlow 模型如何对外提供服务](https://blog.csdn.net/zjerryj/article/details/80308713)
classifier.export_saved_model(
    export_dir_base='target/iris_model',
    serving_input_receiver_fn=serving_input_receiver_fn2)

exit(0)

# 预测
for i in range(0, 100):
    print('\nPlease enter features: SepalLength,SepalWidth,PetalLength,PetalWidth')
    a, b, c, d = map(float, input().split(','))
    predict_x = {
        'SepalLength': [a],
        'SepalWidth': [b],
        'PetalLength': [c],
        'PetalWidth': [d],
    }
    predictions = classifier.predict(
        input_fn=lambda: eval_input_fn(
            predict_x,
            labels=[0],
            batch_size=100
        )
    )

    print(predictions)

    for pred_dict in predictions:
        print(pred_dict)
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print(SPECIES[class_id], 100 * probability)

"""
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['classification']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_example_tensor:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['classes'] tensor_info:
        dtype: DT_STRING
        shape: (-1, 3)
        name: dnn/head/Tile:0
    outputs['scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 3)
        name: dnn/head/predictions/probabilities:0
  Method name is: tensorflow/serving/classify

signature_def['predict']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['examples'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_example_tensor:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['class_ids'] tensor_info:
        dtype: DT_INT64
        shape: (-1, 1)
        name: dnn/head/predictions/ExpandDims:0
    outputs['classes'] tensor_info:
        dtype: DT_STRING
        shape: (-1, 1)
        name: dnn/head/predictions/str_classes:0
    outputs['logits'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 3)
        name: dnn/logits/BiasAdd:0
    outputs['probabilities'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 3)
        name: dnn/head/predictions/probabilities:0
  Method name is: tensorflow/serving/predict

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_example_tensor:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['classes'] tensor_info:
        dtype: DT_STRING
        shape: (-1, 3)
        name: dnn/head/Tile:0
    outputs['scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 3)
        name: dnn/head/predictions/probabilities:0
  Method name is: tensorflow/serving/classify
"""

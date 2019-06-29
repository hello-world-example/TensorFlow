import pandas as pd
import tensorflow as tf

# 从导出目录中加载模型，并生成预测函数。
predict_fn = tf.contrib.predictor.from_saved_model(
    "target/iris_model/1547389348",
    signature_key="predict"
)

# 使用 Pandas 数据框定义测试数据。
inputs = pd.DataFrame(
    {
        'SepalLength': [5.1],
        'SepalWidth': [3.3],
        'PetalLength': [1.7],
        'PetalWidth': [0.5]
    }
)

# 将输入数据转换成序列化后的 Example 字符串。
examples = []
for index, row in inputs.iterrows():
    feature = {}
    for col, value in row.iteritems():
        feature[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    print(feature)
    print("===")
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    print(example)
    print("----")
    examples.append(example.SerializeToString())

# 开始预测
predictions = predict_fn({'examples': examples})

print(predictions)

"""
saved_model_cli show --dir 1547389348/ \
--tag_set serve --signature_def serving_default

saved_model_cli run --dir 1547389348/ --tag_set serve --signature_def serving_default --input_examples 'inputs=[{"SepalLength":[5.3],"SepalWidth":[2.9],"PetalLength":[5.6],"PetalWidth":[1.8]}]'

curl -d '{"signature_name": "serving_default", "examples": [{"SepalLength":[5.1],"SepalWidth":[3.3],"PetalLength":[1.7],"PetalWidth":[0.5]}]}'   -X POST http://localhost:8501/v1/models/iris:classify
"""


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

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 1600])
W = tf.Variable(tf.zeros([1600, 26]))
b = tf.Variable(tf.zeros([26]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "/Users/kail/PycharmProjects/py3/tf/learn/model/model.ckpt")
print("Model restored.")

data = [
    "56",
    "Self-emp-not-inc",
    "186651",
    "11th",
    "7",
    "Widowed",
    "Other-service",
    "Unmarried",
    "White",
    "Female",
    "0",
    "0",
    "50",
    "United-States"
]

result = sess.run("<=50K", feed_dict=data)
print(result)
code = ""
for i in result:
    temp = list(i)
    code += chr(temp.index(max(temp)) + 97)

# CSV_COLUMNS = [
#     "age", "workclass", "fnlwgt", "education", "education_num",
#     "marital_status", "occupation", "relationship", "race", "gender",
#     "capital_gain", "capital_loss", "hours_per_week", "native_country",
#     "income_bracket"
# ]

# age,workclass,        fnlwgt,education,education_num,marital_status, occupation,    relationship,race,  gender,capital_gain,capital_loss,hours_per_week,native_country, income_bracket
# 56, Self-emp-not-inc, 186651, 11th,    7,            Widowed,        Other-service, Unmarried,   White, Female, 0,         0,            50,            United-States,  <=50K.

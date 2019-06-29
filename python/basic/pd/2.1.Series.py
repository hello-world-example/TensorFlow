import pandas as pd

"""
Pandas入门2 -- Series类及创建: https://www.jianshu.com/p/1da8db98a073

一维数据结构：Series
二维数据结构: DataFrame
"""

# 每条数据带 label
S_with_user_label = pd.Series([2, 4, 6, 8, 10], ['a', 'b', 'c', 'd', 'e'])
print(S_with_user_label)
print()

# 默认 label
S_with_default_label = pd.Series([2, 4, 5, 6, 10])
print(S_with_default_label)
print()

# 通过字典创建（使用默认字典的顺序）
S_from_dict = pd.Series({
    'd': 1,
    'c': 2,
    'a': 4,
    'b': 3,
})
print(S_from_dict)
print()

# Series类 也类似Dictionary 类，有values 和 index
print(S_from_dict.values)
print(S_from_dict.index)
print(S_from_dict.take([1]))
print(S_from_dict.get("a"))
print()

# 通过标量来创建（Scalar）
S_from_scalar = pd.Series(0, ['a', 'b', 'c', 'd'])
print("创建一个一维数组，元素都是0\n", S_from_scalar)

import numpy as np
import pandas as pd

"""
Pandas入门3 -- Series基本操作 https://www.jianshu.com/p/0761906c4c04

Series对象可以作为大部分Numpy函数的输入参数。
"""

# 一维数组
rand_1 = np.random.rand(5)
print("一维数组")
print(rand_1)
print()

# 二维数组
rand_2 = np.random.rand(5, 5)
print("二维数组")
print(rand_2)
print()

# 3维数组
ran_3 = np.random.rand(5, 5, 5)
print("3维数组")
print(ran_3)
print()

# 4维数组
ran_4 = np.random.rand(5, 5, 5, 5)
print("4维数组")
print(ran_4)
print()

# 基本操作
S = pd.Series(data=rand_2[0], index=["a", "b", "c", "d", "e"])
print("---print(S)")
print(S)
print("---print(S[1])")
print(S[1])
print("---print(S[0:2])")
print(S[0:2])
print("---print(S[:2])")
print(S[:2])
print("Max %s, min %s, median %s, avg %s, " % (S.max(), S.min(), S.median(), S.mean()))
print("---print(S > S.mean())")
print(S > S.mean())
print("---大于均值的所有值")
print(S[S > S.mean()])
print("---S[[1,3]] 依次去第二 第四 个值")
print(S[[1,3]])
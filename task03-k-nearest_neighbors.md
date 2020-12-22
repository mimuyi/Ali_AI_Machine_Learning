#### K近邻居(k-nearest_neighbors)
----
> KNN是一个较懒的学习算法，只要有数据不用训练成模型就可以开干

> 可以解决分类、回归、填充问题

##### 概念
----
> 纬度上的距离计算(如:欧氏距离)，但会出现高纬度下数据样本稀疏、距离计算困难等问题——"维数灾难"(curse of dimensionality)。

> 降维(dimension reduction)缓解维数灾难的灾难的一个重要途径，将高纬度的空间转变为一个低纬度的空间，提高了数据集的密度并且降低了数据之间的距离计算。“多纬缩放”(Multiple Dimensional Scaling ,简称 MDS)方法可以使原始空间中高纬度样本之间的距离在低纬度中得以保持。

##### Demo & 鸢尾花
----
0. 使用KNN简单流程
>
> from sklearn.neighbors import KNeighborsClassifier
>
> clt = KNeighborsClassifier(n_neighbors=5, p=2), n_neighbors:K值， p：当p=1时，这等价于使用曼哈顿距离(L1)，p=2为欧几里得距离(L2)
>
> clt.fit(x,y)
>
> clt.predict(), 预测提供的数据的类标签
>
1. Demo
>
> \# 使用以下库
>
> import numpy as np
>
> import matplotlib.pyplot as plt
>
> from matplotlib.colors import ListedColormap  \# plt颜色
>
> from sklearn.neighbors import KNeighborsClassifier \# KNN
>
> from sklearn import datasets \# 数据提供

> \# 实现阶段

> \# KNN算法函数
>
> KNeighborsClassifier ( n_neighbors=5,  weights=’uniform’,  algorithm=’auto’,  leaf_size=30, p=2,  metric=’minkowski’,  metric_params=None,  n_jobs=1,  \*\*kwargs )
>
> n_neighbors：就是选取最近的点的个数：k , 
>
> `clf = KNeighborsClassifier(k)`  \# 在循环中构造不同K值的KNN
>
> KNeighborsClassifier.fit \# 以X为训练数据，y为目标值拟合模型
>
> `clf.fit(X,y)` \# 根据K值和x,y数据集生成模型
> 
> KNeighborsClassifier.predict() \# 使用模型预测提供的数据的类标签
>
> `clf.predict(np.c_[xx.ravel(), yy.ravel()])`

2. 鸢尾花
> \# 在Demo基础上增加以下库
> from sklearn.model_selection import train_test_split
>
> X = iris.data \# 使用完整数据集
>
> \# 训练集合和验证集合
>
> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
>
> \# 使用KNeighborsClassifier(n_neighbors, p, metric_params)生成KNN模型, n_neighbors：K值 , p：当p=1时，这等价于使用曼哈顿距离(L1)，p=2为欧几里得距离(L2) , 度量函数的附加关键字参数，设置应为dict（字典）形式
>
> clf = KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski") 
>
> 按照流程走即可






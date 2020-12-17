#### 学习记录
----
###### 预备知识
1. 西瓜书第三章线形模型
2. sklearn（python机器学习库）中文文档 https://sklearn.apachecn.org/
###### 逻辑回归（Logistic Regression）
> 逻辑回归(LR)是用于解决二分类任务的机器学习方法，可以认为得到是1或者0结果。LR在线性回归理论基础上增加Sigmoid函数完成非线性处理。
> sigmoid函数： 1 / (1 + e^(-x)) 取值范围[0,1] （重点）
###### Demo
> `from sklearn.linear_model import LogisticRegression` 导入逻辑回归模型

> `LogisticRegression.fit(x,y)` 调用拟合方程 `y=w0+w1*x1+w2*x2`
1. 画图库基本使用
> `plt.figure()`

> \# x_fearures:为即将绘制散点图的数据点

> \# c:表示的是颜色

> \# s:是一个实数或者是一个数组

> \# cmap:Colormap实体或者是一个colormap的名字，cmap仅仅当c是一个浮点数数组的时候才使用

> `plt.scatter(x_fearures[:,0],x_fearures[:,1], c=y_label, s=50, cmap='viridis')`

> `plt.title('Dataset')`

> `plt.show()`

> \# 可视化决策边界

> `plt.contour(x_grid, y_grid, z_proba, [0.5], linewidths=2., colors='blue')`

> \# 可视化预测新样本 增加样本

> `x_fearures_new1 = np.array([[0, -1]])`

> `plt.scatter(x_fearures_new1[:,0],x_fearures_new1[:,1], s=50, cmap='viridis')`

> `plt.annotate(s='New point 1',xy=(0,-1),xytext=(-2,0),color='blue',arrowprops=dict(arrowstyle='-|>',connectionstyle='arc3',color='red'))`

> \# 在训练集和测试集上分布利用训练好的模型进行预测

> `lr_clf.predict(x_fearures_new1)`

> \# 由于逻辑回归模型是概率预测模型（前文介绍的 p = p(y=1|x,\theta)）,所有我们可以利用 predict_proba 函数预测其概率

> `lr_clf.predict_proba(x_fearures_new1)`

###### 鸢尾花实战
> \# 导入鸢尾花数据集

> `from sklearn.datasets import load_iris`

> `data = load_iris()`

> `iris_target = data.target`

> \# 使用pandas转化为DataFrame格式

> `iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)`

> \# 查看数据整体信息、查看头部和尾部、对应标签、查看每个类别数量、特征统计描述

> `iris_features.info(), iris_features.head() iris_features.tail(), iris_target, pd.Series(iris_target).value_counts(), pd.Series(iris_target).value_counts()`

> \# 特征与标签组合的图 使用seaborn库

> `import seaborn as sns`

> `sns.pairplot(data=iris_all,diag_kind='hist', hue= 'target')`

> \# 使用plt展示

> `plt.show()`

> \# 进入训练阶段

> `from sklearn.model_selection import train_test_split`



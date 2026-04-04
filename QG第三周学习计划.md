# QG第三周学习计划

> **以下代码实现均为DeepML机器学习章节的题目答案**

## 机器学习的基本概念

### 三要素：模型 学习准则 优化算法

**模型**：本质是一个函数，根据输入的特征映射对应的预测结果y

- 线性模型：$f(x;\theta)=w^Tx+b$

  > 超参数$\theta$包含权重参数w和偏置b

- 非线性模型：$f(x;\theta)=w^T\phi(x)+b$

  > $\phi(x)$是非线性函数

**学习准则**：决定如何从假设空间中选择最优模型，本质是通过构造目标函数并最小化（或最大化）它来求解模型参数

- **期望风险最小化**：就是让损失最小，但是期望风险是用**真实结果**计算的，我们没有**真实结果**，无法计算
- **经验风险最小化**：期望风险算不了，用经验风险代替，无限接近**真实结果**

- **结构风险最小化**：用正则化项狠狠惩罚函数，防止过拟合

  > 过拟合：将题库答案吃干净了，但是出新题就不会

- **常用损失函数**：

  - 0-1损失函数
  - 平方损失函数
  - 交叉熵损失函数（负对数似然函数）
  - Hinge损失函数：处理二分类，标签y={1,-1}

**优化算法**：

- **梯度下降法**：常用，直观理解就是向**梯度**反方向走一个搜索步长$\alpha$，每次迭代优化所有样本的平均损失函数

  > 梯度是损失函数偏导结果，指示函数值在该点上升最快的方向

- **提前停止**：没用过暂时，每次迭代计算错误率，不再改变就**提前停止**

- **随机梯度下降**：常用，每次只计算当前的样本的损失函数，每次只优化一个损失函数
- **小批量梯度下降**：每次计算固定数量的平均损失函数而不是所有，收敛快开销小，常有

## 机器学习的算法类型

- **监督学习**：预测结果给标签判断对错
- **无监督学习**：从没有标签的数据集中学习
- **强化学习**：有交互，智能体在和环境的交互中不断学习并调整策略，以取得最大化的期望总回报
- **弱监督学习**：
- **半监督学习**：

## 泛化

> 模型在新的训练数据的表现情况

## 模型评估指标

- **预测问题**：
  - 均方根误差
  - 均方误差
  - 平均绝对误差
  - 平均绝对百分比误差
  - 对称绝对平均百分比误差
  - 均方对数误差
  - 中位绝对误差
- **分类问题**：
  - 准确率(accuracy)
  - 召回率（precision）
  - 综合分类Fscore：$F_A=\frac{(1+\beta^2)\times Precision \times Recall}{Precision+Recall}$
    - F1score即$\beta$为1
- **回归问题**：
  - 平均绝对误差：
  - 均方误差
  - 均方根误差
- **常用损失**
  - 交叉熵损失：
  - KL散度：衡量近似分布和真实分布的信息损失，量化意外程度
  - 3Js散度
- **鲁棒性和泛化性**：

## 监督学习

#### 线性模型

##### 线性回归

用于处理预测问题，因变量是预测的结果，通过正规方程梯度下降找到极值点

##### 逻辑回归

**本质**：将**线性回归**的输出映射到(0,1)区间，预测发生概率

用于处理分类问题，逻辑回归的激活函数是softmax的特殊情况sigmod
$$
对于向量x，有x=(x_1,x_2,\dots,x_k)\\
softmax(x_k)=\frac{exp(x_k)}{\sum_{i=1}^kexp(x_i)}\\
softmax以向量中概率最高的为预测结果
$$

```python
import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    arr1 = np.array(scores)
    size = np.shape(arr1)[0]
    res = []
    for i in arr1:
        temp = np.exp(i) / np.sum(np.exp(arr1))
        res.append(round(np.log(temp), 4))
    result = np.array(res)
    return result
```

而逻辑回归的核心是sigmod函数，处理二分类问题
$$
sigmod(x)=\frac{1}{1+exp(w^Tx+b)}
$$

```python
import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    arr1 = []
    for i in X:
        #题目把偏置放到矩阵外了
        predict = 1 / (1 + np.exp(-(weights.T @ i + bias)))
        if predict >= 0.5:
            arr1.append(1)
        else:
            arr1.append(0)
    res = np.array(arr1)
    return res
```

#### SVM支持向量机

> Pegasos算法实现

支持向量机是一种经典的二分类算法

**概念理解**：

- 超平面：在n维的空间中，n-1维的空间可以将空间分成两个不相交的空间
- 支持向量机就是找到一个超平面，将在高维空间中的样本点分开
- SVM不但分类，还让两个分类之间留下尽可能大的间隔
- 硬间隔：间隔完全没东西
- 软间隔：有一点

**公式推导**：……

反正是个$w^T + b=0$

**Pegasos**基于随机梯度下降实现，找到最优的w和b

**核函数**（kernel_func）将原始数据映射到更高维的空间，使之可能线性可分

推导省略

## 无监督学习

#### K簇聚类模型

> 最常用聚类算法

**流程**:

1. 从n个样本中随机取k个对象作为初始聚类中心

2. 分别计算其他点到中心的距离，分配给最近的中心点

3. 分配后根据簇内各对象在各维度均值定中心点

4. 如果与上次比较中心点发生变化，转到step2，否则step5
5. 中心点不再变后，输出聚类结果

```python
import math

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    '''
    实现一个简单的k聚类算法
    :param points: 样本点列表
    :param k: 目标生成的k簇数量
    :param initial_centroids: 初始化的中心点都给我了
    :param max_iterations: 最大迭代次数
    :return: 返回最后得到的类中心点列表
    '''
    res = []
    centroids = initial_centroids

    for _ in range(max_iterations):
        # 存入当前簇的点
        clusters = [[] for _ in range(k)]
        for point in points:
            dist_min = math.inf
            k_index = -1
            for i in range(k):
                if dist_min > math.dist(point, centroids[i]):
                    dist_min = math.dist(point, centroids[i])
                    k_index = i
            clusters[k_index].append(point)
        # 更新中心点
        temp_centroids = []
        for cluster in clusters:
            point_sum = []
            for i in range(len(cluster[0])):
                temp = 0
                for point in cluster:
                    temp += point[i]
                temp /= len(cluster)
                point_sum.append(round(temp, 4))
            point_sum_tuple = tuple(point_sum)
            temp_centroids.append(point_sum_tuple)
        # 检查变化
        judge = 1
        for i in range(k):
            if centroids[i] != temp_centroids[i]:
                centroids = temp_centroids
                judge = 0
                break
        if judge == 1:
            return centroids
    return centroids
```

**难点**：初始中心点的选择和数量是很重要的最简单的办法是定两个距离最大的点，再距离他们最大找新的中心点，直到满足数量需求

#### PCA降维技术

> 又叫做主成分分析，当两个特征相关的时候，合并成一个主成分实现降维

**[前置]协方差矩阵**：

- 方差度量单个随机变量的离散程度

- 协方差刻画两个随机变量的相似度

$$
\sigma(x,y)=\frac{1}{n-1}\sum^n_{i=1}(x_i-\overline{x})(y_i- \overline{y})
$$

所以协方差矩阵是个对称的矩阵，而**实对称矩阵X**可以被一个**正交矩阵Q**对角化
$$
QQ^T=Q^T=I
$$
设一个对角矩阵A
$$
X=QAQ^T
$$
……

**流程**：

- 从2D样本中取k个主成分
- 中心化，使每个特征的均值为0，标准差为1
- 得协方差矩阵，然后用np.linalg.eigh特征分解
- 对分解得到得特征值与对应向量排序，选k个最大的为主元


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


def kmeans(X, k, max_iters=100, random_state=None):
    """
    K-Means 聚类算法

    参数:
    X : ndarray, 形状 (n_samples, n_features)
        输入数据矩阵，每行一个样本
    k : int
        聚类簇数
    max_iters : int, 默认 100
        最大迭代次数
    random_state : int, 默认 None
        随机种子

    返回:
    labels : ndarray, 形状 (n_samples,)
        每个样本所属的簇标签，从 0 到 k-1
    centers : ndarray, 形状 (k, n_features)
        最终的中心点坐标
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape

    # 1. 初始化中心点：随机选择 k 个样本作为初始中心
    indices = np.random.choice(n_samples, k, replace=False)
    centers = X[indices].copy()

    for iteration in range(max_iters):
        # 2. 计算每个样本到各中心的距离
        distances = np.zeros((n_samples, k))
        for i in range(k):
            # 使用欧氏距离
            distances[:, i] = np.sqrt(((X - centers[i]) ** 2).sum(axis=1))

        # 3. 分配标签：每个样本分配到最近的中心
        labels = np.argmin(distances, axis=1)

        # 4. 更新中心点：计算每个簇的均值
        new_centers = np.zeros((k, n_features))
        for i in range(k):
            if np.sum(labels == i) > 0:
                new_centers[i] = X[labels == i].mean(axis=0)
            else:
                # 如果某个簇没有样本，随机重新初始化
                new_centers[i] = X[np.random.randint(0, n_samples)]

        # 5. 检查收敛：如果中心点不再变化，提前停止
        if np.allclose(centers, new_centers):
            print(f"迭代 {iteration + 1} 次后收敛")
            break

        centers = new_centers

    return labels, centers


def test_kmeans():
    """测试 K-Means 算法在鸢尾花数据集上的表现"""
    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data
    y = iris.target

    print("数据集形状:", X.shape)
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    print(f"真实类别数: {len(np.unique(y))}")

    # 运行 K-Means
    k = 3
    labels, centers = kmeans(X, k, max_iters=100, random_state=42)

    print(f"\nK-Means 聚类结果 (k={k}):")
    print("簇标签:", labels[:20], "...")  # 只打印前20个
    print("簇中心形状:", centers.shape)

    # 统计每个簇的样本数
    for i in range(k):
        count = np.sum(labels == i)
        print(f"簇 {i}: {count} 个样本")

    # 与真实标签的简单比较（注意：聚类标签是任意的，需要对齐）
    from sklearn.metrics import adjusted_rand_score

    # 计算调整兰德指数
    ari = adjusted_rand_score(y, labels)
    print(f"\n调整兰德指数 (与真实标签比较): {ari:.4f}")

    return labels, centers


if __name__ == "__main__":
    # 运行测试
    labels, centers = test_kmeans()

    # 示例：使用用户提供的数据读取方式（需要 Iris.csv 文件）
    # 如果文件存在，可以取消注释以下代码
    """
    import pandas as pd
    df = pd.read_csv("Iris.csv")
    df = df.drop(['Id'], axis=1)
    X = df.drop(['Species'], axis=1).values
    Y = df.iloc[:, -1].values

    k = 3
    labels, centers = kmeans(X, k, max_iters=100, random_state=42)
    print("聚类完成，标签数量:", len(labels))
    """

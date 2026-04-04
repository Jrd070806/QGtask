import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文字体和更大的图表尺寸
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

# 设置随机种子以确保可重复性
np.random.seed(42)

# 确保图表在Notebook中内联显示
try:
    from IPython import get_ipython

    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic('matplotlib', 'inline')
except:
    pass


# 1. 加载和准备数据
def load_wine_quality_data():
    """加载红酒质量数据集"""
    # 尝试从本地或网络加载数据集
    try:
        # 如果本地有文件
        df = pd.read_csv('winequality-red.csv', delimiter=';')
    except:
        # 从网络加载数据集
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df = pd.read_csv(url, delimiter=';')

    print(f"数据集形状: {df.shape}")
    print(f"特征列: {df.columns.tolist()}")
    print(f"质量分数分布:\n{df['quality'].value_counts().sort_index()}")

    return df


# 2. 数据预处理
def preprocess_data(df, task='regression'):
    """预处理数据，根据任务类型返回相应数据"""
    # 分离特征和标签
    if task == 'regression':
        # 回归任务：直接使用质量分数作为目标
        X = df.iloc[:, :-1].values  # 所有特征
        y = df.iloc[:, -1].values.reshape(-1, 1)  # 质量分数
    else:
        # 分类任务：将质量>6的酒标记为好酒(1)，否则为坏酒(0)
        X = df.iloc[:, :-1].values  # 所有特征
        y = (df.iloc[:, -1] > 6).astype(int).values.reshape(-1, 1)

    # 数据标准化
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normalized = (X - X_mean) / X_std

    # 添加偏置项（截距项）
    X_with_bias = np.c_[np.ones((X_normalized.shape[0], 1)), X_normalized]

    return X_with_bias, y, X_mean, X_std


# 3. 线性回归模型实现
class LinearRegression:
    """从零实现的线性回归模型"""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.loss_history = []

    def compute_loss(self, X, y):
        """计算均方误差损失"""
        m = len(y)
        predictions = X.dot(self.weights)
        loss = (1 / (2 * m)) * np.sum(np.square(predictions - y))
        return loss

    def fit(self, X, y):
        """使用梯度下降训练模型"""
        m, n = X.shape
        self.weights = np.random.randn(n, 1) * 0.01

        for i in range(self.n_iterations):
            # 计算预测值
            predictions = X.dot(self.weights)

            # 计算梯度
            gradient = (1 / m) * X.T.dot(predictions - y)

            # 更新权重
            self.weights -= self.learning_rate * gradient

            # 记录损失
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)

            if i % 100 == 0:
                print(f"迭代 {i}: 损失 = {loss:.4f}")

        return self

    def predict(self, X):
        """预测连续值"""
        return X.dot(self.weights)

    def evaluate_regression(self, X, y):
        """评估回归模型的性能"""
        predictions = self.predict(X)
        m = len(y)

        # 计算MSE, RMSE, MAE
        mse = np.mean((predictions - y) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y))

        # 计算R²分数
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        print("=" * 50)
        print("线性回归模型评估结果:")
        print("=" * 50)
        print(f"均方误差(MSE): {mse:.4f}")
        print(f"均方根误差(RMSE): {rmse:.4f}")
        print(f"平均绝对误差(MAE): {mae:.4f}")
        print(f"R²分数: {r2:.4f}")
        print("=" * 50)

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'predictions': predictions
        }


# 4. 逻辑回归模型实现
class LogisticRegression:
    """从零实现的逻辑回归模型"""

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.loss_history = []

    def sigmoid(self, z):
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def compute_loss(self, X, y):
        """计算逻辑损失（交叉熵损失）"""
        m = len(y)
        h = self.sigmoid(X.dot(self.weights))
        epsilon = 1e-15  # 避免log(0)
        h = np.clip(h, epsilon, 1 - epsilon)
        loss = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return loss

    def fit(self, X, y):
        """使用梯度下降训练模型"""
        m, n = X.shape
        self.weights = np.random.randn(n, 1) * 0.01

        for i in range(self.n_iterations):
            # 计算预测概率
            h = self.sigmoid(X.dot(self.weights))

            # 计算梯度
            gradient = (1 / m) * X.T.dot(h - y)

            # 更新权重
            self.weights -= self.learning_rate * gradient

            # 记录损失
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)

            if i % 100 == 0:
                print(f"迭代 {i}: 损失 = {loss:.4f}")

        return self

    def predict_proba(self, X):
        """预测概率"""
        return self.sigmoid(X.dot(self.weights))

    def predict(self, X, threshold=0.5):
        """预测类别"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def evaluate_classification(self, X, y_true, threshold=0.5, show_plot=True):
        """全面评估分类模型性能"""
        y_pred = self.predict(X, threshold)
        y_prob = self.predict_proba(X)

        # 计算混淆矩阵
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # 计算评估指标
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # 计算ROC-AUC曲线
        auc_score = self.calculate_roc_auc(y_true, y_prob)

        print("=" * 50)
        print("逻辑回归模型评估结果:")
        print("=" * 50)
        print(f"混淆矩阵:")
        print(f"          预测")
        print(f"         正    负")
        print(f"实  正  {tp:4d}  {fn:4d}")
        print(f"际  负  {fp:4d}  {tn:4d}")
        print()
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数 (F1-score): {f1_score:.4f}")
        print(f"ROC-AUC分数: {auc_score:.4f}")
        print("=" * 50)

        # 绘制ROC曲线
        if show_plot:
            self.plot_roc_curve(y_true, y_prob, auc_score)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'auc': auc_score,
            'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
        }

    def calculate_roc_auc(self, y_true, y_prob):
        """计算ROC-AUC分数"""
        # 获取所有可能的阈值
        thresholds = np.sort(np.unique(y_prob))[::-1]

        tpr_list = []  # 真正例率
        fpr_list = []  # 假正例率

        # 计算每个阈值下的TPR和FPR
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        # 添加起点(0,0)和终点(1,1)
        tpr_list = [0] + tpr_list + [1]
        fpr_list = [0] + fpr_list + [1]

        # 计算AUC（使用梯形法则）
        auc = 0
        for i in range(1, len(fpr_list)):
            auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

        return auc

    def plot_roc_curve(self, y_true, y_prob, auc_score):
        """绘制ROC曲线"""
        # 获取所有可能的阈值
        thresholds = np.sort(np.unique(y_prob))[::-1]

        tpr_list = []
        fpr_list = []

        # 计算每个阈值下的TPR和FPR
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        # 添加起点(0,0)和终点(1,1)
        tpr_list = [0] + tpr_list + [1]
        fpr_list = [0] + fpr_list + [1]

        # 绘制ROC曲线
        plt.figure(figsize=(10, 6))
        plt.plot(fpr_list, tpr_list, 'b-', linewidth=2, label=f'ROC曲线 (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='随机分类器')
        plt.xlabel('假正例率 (FPR)', fontsize=12)
        plt.ylabel('真正例率 (TPR)', fontsize=12)
        plt.title('ROC曲线', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        # 确保图表显示
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)  # 短暂的暂停以确保图表显示


# 5. 分析两个模型的异同点
def analyze_model_differences(linear_model, logistic_model, X_train_lin, y_train_lin, X_train_log, y_train_log):
    """分析线性回归和逻辑回归的异同点"""
    print("\n" + "=" * 60)
    print("线性回归与逻辑回归模型对比分析")
    print("=" * 60)

    print("\n1. 相同点:")
    print("   - 两种模型都是线性模型，基于特征加权和的形式进行预测")
    print("   - 都使用梯度下降法优化模型参数")
    print("   - 都需要对特征进行标准化处理以提高训练效果")
    print("   - 都可以处理多特征问题")

    print("\n2. 不同点:")
    print("   - 输出类型:")
    print("     * 线性回归: 连续值输出，用于回归任务")
    print("     * 逻辑回归: 概率值输出，通过阈值转换为离散类别，用于分类任务")
    print("   - 激活函数:")
    print("     * 线性回归: 无激活函数，直接输出加权和")
    print("     * 逻辑回归: 使用Sigmoid函数将输出压缩到(0,1)区间")
    print("   - 损失函数:")
    print("     * 线性回归: 均方误差(MSE)")
    print("     * 逻辑回归: 交叉熵损失(Log Loss)")
    print("   - 优化目标:")
    print("     * 线性回归: 最小化预测值与真实值的平方差")
    print("     * 逻辑回归: 最大化似然函数，即分类正确的概率")
    print("   - 评估指标:")
    print("     * 线性回归: MSE, RMSE, MAE, R²等")
    print("     * 逻辑回归: 准确率、精确率、召回率、F1分数、AUC等")

    print("\n3. 模型复杂度对比:")
    print(f"   - 线性回归参数数量: {linear_model.weights.shape[0]}")
    print(f"   - 逻辑回归参数数量: {logistic_model.weights.shape[0]}")
    print(f"   - 线性回归最终损失: {linear_model.loss_history[-1]:.4f}")
    print(f"   - 逻辑回归最终损失: {logistic_model.loss_history[-1]:.4f}")

    # 绘制损失函数下降曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 线性回归损失曲线
    axes[0].plot(linear_model.loss_history[:100], 'b-', linewidth=2)
    axes[0].set_xlabel('迭代次数', fontsize=12)
    axes[0].set_ylabel('损失值', fontsize=12)
    axes[0].set_title('线性回归损失函数下降曲线', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # 逻辑回归损失曲线
    axes[1].plot(logistic_model.loss_history[:100], 'r-', linewidth=2)
    axes[1].set_xlabel('迭代次数', fontsize=12)
    axes[1].set_ylabel('损失值', fontsize=12)
    axes[1].set_title('逻辑回归损失函数下降曲线', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)  # 短暂的暂停以确保图表显示

    return fig


# 6. 可视化结果函数
def visualize_results(linear_model, logistic_model, linear_results, logistic_results, X_test_lin, y_test_lin,
                      X_test_log, y_test_log):
    """可视化模型结果"""

    # 创建一个大图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 线性回归预测结果散点图
    linear_predictions = linear_model.predict(X_test_lin)
    axes[0, 0].scatter(y_test_lin, linear_predictions, alpha=0.6, edgecolors='black', linewidth=0.5)

    # 绘制理想预测线
    min_val = min(np.min(y_test_lin), np.min(linear_predictions))
    max_val = max(np.max(y_test_lin), np.max(linear_predictions))
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测线')

    axes[0, 0].set_xlabel('实际质量分数', fontsize=12)
    axes[0, 0].set_ylabel('预测质量分数', fontsize=12)
    axes[0, 0].set_title('线性回归: 预测 vs 实际', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. 线性回归误差分布
    linear_errors = y_test_lin - linear_predictions
    axes[0, 1].hist(linear_errors.flatten(), bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('预测误差', fontsize=12)
    axes[0, 1].set_ylabel('频数', fontsize=12)
    axes[0, 1].set_title('线性回归误差分布', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 逻辑回归概率分布
    logistic_proba = logistic_model.predict_proba(X_test_log)
    y_pred_log = logistic_model.predict(X_test_log)

    # 获取正类和负类的预测概率
    positive_proba = logistic_proba[y_test_log.flatten() == 1]
    negative_proba = logistic_proba[y_test_log.flatten() == 0]

    axes[1, 0].hist(positive_proba.flatten(), bins=20, alpha=0.6, label='好酒(正类)', color='green', edgecolor='black')
    axes[1, 0].hist(negative_proba.flatten(), bins=20, alpha=0.6, label='坏酒(负类)', color='red', edgecolor='black')
    axes[1, 0].axvline(x=0.5, color='blue', linestyle='--', linewidth=2, label='阈值(0.5)')
    axes[1, 0].set_xlabel('预测概率', fontsize=12)
    axes[1, 0].set_ylabel('频数', fontsize=12)
    axes[1, 0].set_title('逻辑回归预测概率分布', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 混淆矩阵热力图
    cm = [[logistic_results['confusion_matrix']['TN'], logistic_results['confusion_matrix']['FP']],
          [logistic_results['confusion_matrix']['FN'], logistic_results['confusion_matrix']['TP']]]

    im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 1].set_title('逻辑回归混淆矩阵', fontsize=14)

    # 添加颜色条
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 设置坐标轴标签
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(['预测坏酒', '预测好酒'])
    axes[1, 1].set_yticklabels(['实际坏酒', '实际好酒'])

    # 在图中显示数值
    thresh = np.max(cm) / 2
    for i in range(2):
        for j in range(2):
            axes[1, 1].text(j, i, format(cm[i][j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i][j] > thresh else "black",
                            fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    return fig


# 7. 主函数
def main():
    print("红酒质量预测与分类任务")
    print("=" * 60)

    # 加载数据
    df = load_wine_quality_data()

    # 划分训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f"\n训练集大小: {train_df.shape}")
    print(f"测试集大小: {test_df.shape}")

    # 1. 线性回归任务：预测红酒质量评分
    print("\n" + "=" * 60)
    print("线性回归任务：预测红酒质量评分")
    print("=" * 60)

    # 准备数据（回归）
    X_train_lin, y_train_lin, X_mean_lin, X_std_lin = preprocess_data(train_df, 'regression')
    X_test_lin, y_test_lin, _, _ = preprocess_data(test_df, 'regression')

    # 训练线性回归模型
    print("\n训练线性回归模型...")
    linear_model = LinearRegression(learning_rate=0.1, n_iterations=500)
    linear_model.fit(X_train_lin, y_train_lin)

    # 评估线性回归模型
    print("\n评估线性回归模型在测试集上的表现...")
    linear_results = linear_model.evaluate_regression(X_test_lin, y_test_lin)

    # 2. 逻辑回归任务：分类好/坏酒
    print("\n" + "=" * 60)
    print("逻辑回归任务：分类好酒与坏酒（质量>6为好酒）")
    print("=" * 60)

    # 准备数据（分类）
    X_train_log, y_train_log, X_mean_log, X_std_log = preprocess_data(train_df, 'classification')
    X_test_log, y_test_log, _, _ = preprocess_data(test_df, 'classification')

    print(f"\n类别分布:")
    print(f"训练集 - 好酒: {np.sum(y_train_log == 1)}, 坏酒: {np.sum(y_train_log == 0)}")
    print(f"测试集 - 好酒: {np.sum(y_test_log == 1)}, 坏酒: {np.sum(y_test_log == 0)}")

    # 训练逻辑回归模型
    print("\n训练逻辑回归模型...")
    logistic_model = LogisticRegression(learning_rate=0.1, n_iterations=500)
    logistic_model.fit(X_train_log, y_train_log)

    # 评估逻辑回归模型
    print("\n评估逻辑回归模型在测试集上的表现...")
    logistic_results = logistic_model.evaluate_classification(X_test_log, y_test_log, show_plot=True)

    # 3. 分析两个模型的异同点
    analyze_model_differences(linear_model, logistic_model, X_train_lin, y_train_lin, X_train_log, y_train_log)

    # 4. 可视化结果
    print("\n" + "=" * 60)
    print("模型结果可视化")
    print("=" * 60)
    fig = visualize_results(linear_model, logistic_model, linear_results, logistic_results,
                            X_test_lin, y_test_lin, X_test_log, y_test_log)

    # 5. 模型评估总结
    print("\n" + "=" * 60)
    print("模型评估总结")
    print("=" * 60)

    print("\n线性回归模型总结:")
    print(f"  模型可以预测红酒质量评分在{df['quality'].min()}-{df['quality'].max()}之间")
    print(f"  在测试集上的R²分数为: {linear_results['R2']:.4f}")
    print(f"  平均预测误差约为: {linear_results['MAE']:.4f}分")

    print("\n逻辑回归模型总结:")
    print(f"  模型准确率为: {logistic_results['accuracy']:.4f}")
    print(f"  模型精确率为: {logistic_results['precision']:.4f}，即预测为好酒的样本中，实际为好酒的比例")
    print(f"  模型召回率为: {logistic_results['recall']:.4f}，即实际为好酒的样本中，被正确预测的比例")
    print(f"  模型F1分数为: {logistic_results['f1_score']:.4f}，是精确率和召回率的调和平均")
    print(f"  模型AUC分数为: {logistic_results['auc']:.4f}，表示模型区分好酒和坏酒的能力")

    if logistic_results['auc'] > 0.8:
        print("  AUC > 0.8，模型具有较好的分类能力")
    elif logistic_results['auc'] > 0.7:
        print("  AUC > 0.7，模型具有一般的分类能力")
    else:
        print("  AUC较低，模型分类能力有待提高")

    # 保存图表
    try:
        plt.figure(fig.number)
        plt.savefig('wine_quality_models_results.png', dpi=150, bbox_inches='tight')
        print("\n图表已保存为 'wine_quality_models_results.png'")
    except:
        pass

    return linear_model, logistic_model, linear_results, logistic_results, fig


# 8. 运行主函数
if __name__ == "__main__":
    # 检查是否在交互式环境中运行
    import sys

    if 'ipykernel' in sys.modules:
        print("检测到Jupyter Notebook环境，图表将内联显示")

    linear_model, logistic_model, linear_results, logistic_results, fig = main()

    # 在非交互式环境中保持图表显示
    if 'ipykernel' not in sys.modules:
        print("\n程序执行完成！按任意键关闭所有图表...")
        plt.show(block=True)

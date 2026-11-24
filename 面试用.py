
# import os
# os.system("pip install pandas numpy matplotlib scikit-learn")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 解决画图时中文显示成方块的问题
plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows系统用黑体
plt.rcParams['axes.unicode_minus'] = False

# 1. 模拟生成电商数据（假装是你从后台导出来的）
np.random.seed(42)
data = {
    'UserID': range(1, 501),
    'OrderDate': pd.date_range(start='2025-01-01', periods=500, freq='D'),
    'OrderValue': np.random.uniform(50, 500, 500), # 假设客单价50-500元
    'Frequency': np.random.randint(1, 15, 500)     # 假设购买频次1-15次
}
df = pd.DataFrame(data)

# 计算R（最近一次购买距离基准日的天数）
base_date = pd.Timestamp('2025-08-01')
df['Recency'] = (base_date - df['OrderDate']).dt.days

# 提取RFM特征，并进行标准化（Kmeans聚类必须做的一步）
X = df[['Recency', 'Frequency', 'OrderValue']]
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(X)

# 2. 跑Kmeans聚类（把用户分成4类）
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(rfm_scaled)

# 3. 画图（这就是你要的证据）
plt.figure(figsize=(10, 6))
# 用不同颜色把4类用户画出来
scatter = plt.scatter(df['Frequency'], df['OrderValue'], c=df['Cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='User Cluster (用户群体)')
plt.title('电商用户 RFM 与 Kmeans 聚类结果图')
plt.xlabel('购买频次 (Frequency)')
plt.ylabel('客单价 (Order Value)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
import numpy as np
from sklearn.cluster import KMeans
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

# 读取点云数据文件
data = np.loadtxt('../PCD/points.txt', delimiter=' ')
file_path = '../PCD/point_cloud_labeled.txt'
# 自适应确定聚类数
y_coords = data[:, 1]
var_y = np.var(y_coords)
n_clusters = int(np.sqrt(var_y) * 30)

# 按点的y轴坐标进行聚类
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(data[:, 1:2])

# 获取聚类结果
labels = kmeans.labels_

# 将每个点的聚类结果添加到点云数据中
data_labeled = np.column_stack((data, labels))

# 绘制聚类结果
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

with open(file_path, 'w') as f:
    f.truncate(0)

with open(file_path, 'a') as f:
    # 循环遍历每个类别
    for i in range(kmeans.n_clusters):
        # 选择属于当前类别的点
        cluster_points = data[labels == i]
        # 按z排序
        cluster_sorted = cluster_points[np.argsort(cluster_points[:, 2])]

        # 将点保存到单独的文件中
        cluster_name = "cluster{}".format(i)
        cluster_name = np.array([cluster_name])
        np.savetxt(f, cluster_name, header='', fmt='%s')
        np.savetxt(f, cluster_points, header='', fmt='%.6f')

        # 绘制点和线
        ax.scatter(cluster_sorted[:, 0], cluster_sorted[:, 1], cluster_sorted[:, 2], label=f'Cluster {i}')
        ax.plot(cluster_sorted[:, 0], cluster_sorted[:, 1], cluster_sorted[:, 2])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

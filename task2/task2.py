from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Функция для загрузки данных из CSV и преобразования их в массив NumPy
def load_data(file_path):
    data_file = pd.read_csv(file_path, delimiter="\t", header=None)
    result_array = []
    for index, row in data_file.iterrows():
        result_array.append([float(row[0]), float(row[1])])
    return pd.DataFrame(result_array, columns=["X1", "X2"])


def make_cluster_graphic(labels, data, n_clusters, best_silhouette, name):
    unique_labels = set(labels)

    # Определяем цвета для каждого кластера, включая выбросы
    colors = {
        -1: 'black',  # Цвет для выбросов
        0: 'red',
        1: 'green',
        2: 'blue',
        3: 'cyan',
        4: 'magenta'
        # Добавьте больше цветов по необходимости
    }

    # Визуализация результатов кластеризации
    plt.figure(figsize=(10, 6))
    for label in unique_labels:
        class_member_mask = (labels == label)
        xy = data[class_member_mask]
        plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=colors[label], markersize=12, markeredgecolor='k')

    plt.title(f'{name}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig(f"{name}.png")


# Функция для выполнения k-means кластеризации и оценки с помощью силуэтного коэффициента
def kmeans_clustering(data, range_n_clusters, data_name):
    best_silhouette = -1
    best_n_clusters = 0
    best_labels = None
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_init=10, n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, labels)
        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_n_clusters = n_clusters
            best_labels = labels

    make_cluster_graphic(best_labels, data, best_n_clusters, best_silhouette, f"{data_name}_kmeans")
    return best_n_clusters, best_labels, best_silhouette


# Функция для выполнения DBSCAN кластеризации и оценки с помощью силуэтного коэффициента
def dbscan_clustering(data, eps_values, data_name):
    best_silhouette = -1
    best_eps = 0
    best_labels = None
    best_n_clusters = None
    for min in range(2, 200):
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=min)
            labels = dbscan.fit_predict(data)
            if len(np.unique(labels)) > 1:
                silhouette_avg = silhouette_score(data, labels)
                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_eps = eps
                    best_n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    best_labels = labels
    make_cluster_graphic(best_labels, data, best_n_clusters, best_silhouette, f"{data_name}_dbscan")
    return best_eps, best_labels, best_silhouette



def hierarchical_clustering(data, range_n_clusters, data_name):
    best_silhouette = -1
    best_n_clusters = 0
    best_cluster_assignments = None
    # Вычисляем матрицу связей
    Z = linkage(data, method='ward')
    for n_clusters in range_n_clusters:
        # Получение меток кластеров
        cluster_assignments = fcluster(Z, n_clusters, criterion='maxclust')
        # Вычисление силуэтного коэффициента
        silhouette_avg = silhouette_score(data, cluster_assignments)
        # Проверка и сохранение лучшего силуэтного коэффициента
        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_n_clusters = n_clusters
            best_cluster_assignments = cluster_assignments
    print(best_cluster_assignments)
    make_cluster_graphic(best_cluster_assignments, data, best_n_clusters, best_silhouette, f"{data_name}_hierarchy")
    return best_n_clusters, best_cluster_assignments, best_silhouette


# Функция для выполнения иерархической кластеризации и оценки с помощью силуэтного коэффициента

# Загрузка данных
data1 = load_data('clustering_1.csv')
data2 = load_data('clustering_2.csv')
data3 = load_data('clustering_3.csv')

# Выбор диапазона количества кластеров для k-means и иерархической кластеризации
range_n_clusters = range(2, 11)

# Выбор диапазона значений eps для DBSCAN
eps_values = np.arange(0.01, 4.01, 0.01)

for name, data in {"cluster1": data1, "cluster2": data2, "cluster3": data3}.items():
    # Применение k-means кластеризации
    kmeans_n_clusters, kmeans_labels, kmeans_silhouette = kmeans_clustering(data, range_n_clusters, name)
    dbscan_eps, dbscan_labels, dbscan_silhouette = dbscan_clustering(data, [0.30805843757575], name)  # 0.308058437

    # Применение иерархической кластеризации
    hierarchical_n_clusters, hierarchical_labels, hierarchical_silhouette = hierarchical_clustering(data,
                                                                                                    range_n_clusters,
                                                                                                    name)
    print(f"Kmeans silhouette {name}  for {kmeans_n_clusters} clusters : {kmeans_silhouette}",
          f"DBscan silhouette {name}   eps={dbscan_eps}              : {dbscan_silhouette}",
          f"Hierarchical silhouette {name} for {hierarchical_n_clusters} clusters : {hierarchical_silhouette}",
          sep="\n")
    print()

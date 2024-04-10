from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import numpy as np
import pandas as pd


# Функция для загрузки данных из CSV и преобразования их в массив NumPy
def load_data(file_path):
    data_file = pd.read_csv(file_path, delimiter="\t", header=None)
    result_array = []
    for index, row in data_file.iterrows():
        result_array.append([float(row[0]), float(row[1])])
    return pd.DataFrame(result_array, columns=["X1", "X2"])


# Функция для выполнения k-means кластеризации и оценки с помощью силуэтного коэффициента
def kmeans_clustering(data, range_n_clusters):
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
    return best_n_clusters, best_labels, best_silhouette


# Функция для выполнения DBSCAN кластеризации и оценки с помощью силуэтного коэффициента
def dbscan_clustering(data, eps_values):
    best_silhouette = -1
    best_eps = 0
    best_labels = None
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps)
        labels = dbscan.fit_predict(data)
        if len(np.unique(labels)) > 1:
            silhouette_avg = silhouette_score(data, labels)
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_eps = eps
                best_labels = labels
    return best_eps, best_labels, best_silhouette


def k_distance_plot(data, n_neighbors):
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(data)

    # Нахождение k-ближайших соседей для каждой точки в данных
    distances, indices = nn.kneighbors(data)

    # Сортировка расстояний
    distances = np.sort(distances[:, n_neighbors - 1])

    # Использование KneeLocator для нахождения точки излома
    kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
    optimal_eps = distances[kneedle.knee] if kneedle.knee else None

    return optimal_eps


def hierarchical_clustering(data, range_n_clusters):
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
    return best_n_clusters, best_cluster_assignments, best_silhouette




# Функция для выполнения иерархической кластеризации и оценки с помощью силуэтного коэффициента

# Загрузка данных
data1 = load_data('clustering_1.csv')
data2 = load_data('clustering_2.csv')
data3 = load_data('clustering_3.csv')


# Выбор диапазона количества кластеров для k-means и иерархической кластеризации
range_n_clusters = range(2, 11)

# Выбор диапазона значений eps для DBSCAN
eps_values = np.arange(0.00000001, 4.0, 0.01)

for name, data in {"cluster1": data1, "cluster2": data2, "cluster3": data3}.items():
    # Применение k-means кластеризации
    kmeans_n_clusters, kmeans_labels, kmeans_silhouette = kmeans_clustering(data, range_n_clusters)
    dbscan_eps, dbscan_labels, dbscan_silhouette = dbscan_clustering(data, [k_distance_plot(data, 4)])

    # Применение иерархической кластеризации
    hierarchical_n_clusters, hierarchical_labels, hierarchical_silhouette = hierarchical_clustering(data,
                                                                                                    range_n_clusters)
    print(f"Kmeans silhouette {name}  for {kmeans_n_clusters} clusters : {kmeans_silhouette}",
          f"DBscan silhouette {name}                    : {dbscan_silhouette}",
          f"Hierarchical silhouette {name} for {hierarchical_n_clusters} clusters : {hierarchical_silhouette}", sep="\n")
    print()

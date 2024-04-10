import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Загрузка данных
pluton_df = pd.read_csv('pluton.csv')
print(pluton_df)

# Стандартизация входных данных
scaler = StandardScaler()
pluton_scaled = scaler.fit_transform(pluton_df)

# Функция для выполнения кластеризации и оценки качества
def cluster_and_evaluate(data, max_iter, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=10, random_state=42)
    kmeans.fit(data)
    inertia = kmeans.inertia_
    silhouette = silhouette_score(data, kmeans.labels_)
    return inertia, silhouette, kmeans.labels_

# Кластеризация стандартизированных данных
inertia_scaled, silhouette_scaled, labels_scaled = cluster_and_evaluate(pluton_scaled, max_iter=300)

# Кластеризация нестандартизированных данных
inertia_unscaled, silhouette_unscaled,labels_unscaled = cluster_and_evaluate(pluton_df, max_iter=300)

# Добавление меток кластеров к исходным данным
pluton_df['Cluster_Unscaled'] = labels_unscaled

# Создание DataFrame для стандартизированных данных с метками кластеров
pluton_scaled_df = pd.DataFrame(pluton_scaled, columns=pluton_df.columns[:-1])
pluton_scaled_df['Cluster_Scaled'] = labels_scaled

# Сохранение результатов в CSV-файлы
pluton_df.to_csv('pluton_with_clusters_unscaled.csv', index=False)
pluton_scaled_df.to_csv('pluton_with_clusters_scaled.csv', index=False)
print(f"Scaled inertia  :{inertia_scaled}")
print(f"Unscaled inertia:{inertia_unscaled}")

print(f"Scaled silhouette  :{silhouette_scaled}")
print(f"Unscaled silhouette:{silhouette_unscaled}")
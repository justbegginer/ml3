from sklearn.cluster import KMeans
from PIL import Image
import numpy as np

# Загрузка изображения
image_path = 'img_1.png'  # Замените на путь к вашему изображению
image = Image.open(image_path)
image_np = np.array(image)

# Преобразование изображения в двумерный массив пикселей
pixels = image_np.reshape(-1, 3)

# Кластеризация пикселей
kmeans = KMeans(n_clusters=5)
kmeans.fit(pixels)

# Получение центров кластеров
centroids = kmeans.cluster_centers_

# Создание нового изображения из центров кластеров
clustered_image = np.array([centroids[label] for label in kmeans.labels_], dtype=np.uint8)
clustered_image = clustered_image.reshape(image_np.shape)
clustered_image = Image.fromarray(clustered_image)

# Сохранение изображения
clustered_image.save(f'clustered_image.png')

# Создание и сохранение палитры
palette_image = Image.new('RGB', (centroids.shape[0], 50), color='white')
palette_pixels = palette_image.load()
for i, color in enumerate(centroids.astype(int)):
    for j in range(50):
        palette_pixels[i, j] = tuple(color)
palette_image.save('palette_image.png')

# Визуализация исходного и кластеризованного изображений

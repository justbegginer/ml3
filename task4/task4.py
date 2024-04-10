import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

# Предположим, что файл 'votes.csv' находится в той же директории, что и ваш скрипт.
votes_data = pd.read_csv('votes.csv')

# Построение дендрограммы для набора данных
# Вычисляем связи между данными
votes_data = votes_data.replace([np.inf, -np.inf], np.nan)
votes_data = votes_data.apply(lambda x: x.fillna(x.mean()), axis=0)

linked = linkage(votes_data, 'ward')
# Построение дендрограммы
plt.figure(figsize=(25, 10))
dendrogram(linked,
           orientation='top',
           labels=votes_data.index,
           distance_sort='descending',
           show_leaf_counts=True)
plt.xticks(fontsize=8)
plt.xlabel('States')
plt.ylabel('Distance (Ward)')
plt.show()

# При успешном выполнении, сохраняем график в файл
dendrogram_path = 'dendrogram.png'
plt.savefig(dendrogram_path)
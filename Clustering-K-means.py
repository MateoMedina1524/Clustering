# -*- coding: utf-8 -*-

    https://colab.research.google.com/notebooks/intro.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generar datos de ejemplo
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# Implementación de K-means con 4 clusters
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Predicciones (labels de los clusters)
labels = kmeans.predict(X)

# Visualización de los resultados
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')
plt.title('K-means Clustering')
plt.show()


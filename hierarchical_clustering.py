# Importar las librerías necesarias
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generar un conjunto de datos de ejemplo
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Aplicar el clustering jerárquico
agg_clust = AgglomerativeClustering(n_clusters=4)
labels = agg_clust.fit_predict(X)

# Visualizar los resultados
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Resultado de Clustering Jerárquico')
plt.show()

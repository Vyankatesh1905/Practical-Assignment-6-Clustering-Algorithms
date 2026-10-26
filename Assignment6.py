
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

df = pd.read_csv("Mall_Customers.csv")

X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

X_scaled = StandardScaler().fit_transform(X)

models = {
    "K-Means": KMeans(n_clusters=5, n_init=10, random_state=42),
    "Hierarchical": AgglomerativeClustering(n_clusters=5),
    "GMM": GaussianMixture(n_components=5, random_state=42),
    "DBSCAN": DBSCAN(eps=0.8, min_samples=5)
}

results = {}
print("CLUSTERING RESULTS")
print("-" * 60)

for name, model in models.items():
    if name == "GMM":
        labels = model.fit_predict(X_scaled)
    else:
        labels = model.fit_predict(X_scaled)

    results[name] = labels
    mask = labels != -1
    clusters = set(labels[mask])

    print("\nAlgorithm:", name)
    print("Number of clusters:", len(clusters))
    print("Noise points:", sum(labels == -1))

    if len(clusters) >= 2 and mask.sum() > len(clusters):
        print("Silhouette Score:",
              round(silhouette_score(X_scaled[mask], labels[mask]), 4))
        print("Davies-Bouldin Index:",
              round(davies_bouldin_score(X_scaled[mask], labels[mask]), 4))
    else:
        print("Metrics unavailable for this clustering result.")

X_2d = PCA(n_components=2).fit_transform(X_scaled)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, (name, labels) in zip(axes.flat, results.items()):
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="viridis", s=25)
    ax.set_title(name)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

plt.tight_layout()
plt.savefig("clustering_comparison.png", dpi=150)
plt.show()

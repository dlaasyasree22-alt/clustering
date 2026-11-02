import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# ---------------------------------------------------------
# 1. DATA PREPROCESSING & FEATURE ENGINEERING
# ---------------------------------------------------------
print("Loading dataset and aggregating customer features...")
df = pd.read_csv('train.csv')

# Create Customer Profiles
cust_df = df.groupby('Customer ID').agg(
    Total_Sales=('Sales', 'sum'),
    Order_Count=('Order ID', 'nunique'),
    Avg_Order_Value=('Sales', 'mean'),
    Unique_Categories=('Category', 'nunique')
).reset_index()

# Log transformation + Standard Scaling
X = cust_df[['Total_Sales', 'Order_Count', 'Avg_Order_Value', 'Unique_Categories']]
X_scaled = StandardScaler().fit_transform(np.log1p(X))

print(f"Data ready: {X_scaled.shape[0]} customer records processed.\n")

# ---------------------------------------------------------
# 2. RUNNING CLUSTERING ALGORITHMS
# ---------------------------------------------------------

# A. K-Means (Random Init)
km_rand = KMeans(n_clusters=3, init='random', n_init=10, random_state=42)
cust_df['KMeans_Random'] = km_rand.fit_predict(X_scaled)

# B. K-Means++ Initialization
km_pp = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
cust_df['KMeans_PP'] = km_pp.fit_predict(X_scaled)

# C. K-Medoids (PAM Implementation)
def fit_kmedoids(X, k=3, max_iter=100, random_state=42):
    np.random.seed(random_state)
    D = pairwise_distances(X, metric='euclidean')
    medoids = np.random.choice(X.shape[0], size=k, replace=False)
    labels = np.argmin(D[:, medoids], axis=1)
    
    for _ in range(max_iter):
        old_medoids = medoids.copy()
        for i in range(k):
            members = np.where(labels == i)[0]
            if len(members) > 0:
                sub_D = D[np.ix_(members, members)]
                medoids[i] = members[np.argmin(sub_D.sum(axis=1))]
        labels = np.argmin(D[:, medoids], axis=1)
        if np.array_equal(old_medoids, medoids):
            break
    return medoids, labels

medoid_indices, kmedoids_labels = fit_kmedoids(X_scaled, k=3)
cust_df['KMedoids'] = kmedoids_labels

# D. Hierarchical Agglomerative Clustering (Ward Linkage)
agg = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
cust_df['Agglomerative'] = agg.fit_predict(X_scaled)

# E. DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=5)
cust_df['DBSCAN'] = dbscan.fit_predict(X_scaled)

# ---------------------------------------------------------
# 3. METRICS COMPUTATION & SUMMARY
# ---------------------------------------------------------
def eval_model(labels, name):
    mask = labels != -1
    if len(set(labels[mask])) > 1:
        sil = silhouette_score(X_scaled[mask], labels[mask])
        db = davies_bouldin_score(X_scaled[mask], labels[mask])
    else:
        sil, db = np.nan, np.nan
    return {
        "Algorithm": name,
        "Clusters": len(set(labels)) - (1 if -1 in labels else 0),
        "Silhouette Score": round(sil, 4),
        "Davies-Bouldin Index": round(db, 4)
    }

results = [
    eval_model(cust_df['KMeans_Random'].values, "K-Means (Random Init)"),
    eval_model(cust_df['KMeans_PP'].values, "K-Means++"),
    eval_model(cust_df['KMedoids'].values, "K-Medoids (PAM)"),
    eval_model(cust_df['Agglomerative'].values, "Agglomerative (Ward)"),
    eval_model(cust_df['DBSCAN'].values, "DBSCAN (eps=0.8, MinPts=5)")
]

print("=== OVERALL CLUSTERING PERFORMANCE MATRIX ===")
print(pd.DataFrame(results).to_string(index=False))

# ---------------------------------------------------------
# 4. GENERATING AND SAVING VISUALIZATIONS
# ---------------------------------------------------------
print("\nGenerating visual outputs...")

# Chart 1: Dendrogram
plt.figure(figsize=(10, 5))
linked = linkage(X_scaled, method='ward')
dendrogram(linked, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=8.)
plt.title("Hierarchical Agglomerative Clustering Dendrogram (Ward Linkage)")
plt.xlabel("Cluster Size / Sample Index")
plt.ylabel("Euclidean Distance")
plt.tight_layout()
plt.savefig("dendrogram_output.png")
plt.close()

# Chart 2: DBSCAN Clusters & Outliers
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cust_df['DBSCAN'], palette='Set1', style=cust_df['DBSCAN'] == -1)
plt.title("DBSCAN Clustering Projection (Noise/Outliers in Red/X)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.tight_layout()
plt.savefig("dbscan_output.png")
plt.close()

# Chart 3: PCA vs. t-SNE Projections
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cust_df['KMeans_PP'], palette='Set1', ax=axes[0])
axes[0].set_title("PCA Projection (2D)")

sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=cust_df['KMeans_PP'], palette='Set1', ax=axes[1])
axes[1].set_title("t-SNE Projection (2D, Perplexity=30)")

plt.tight_layout()
plt.savefig("pca_vs_tsne_output.png")
plt.close()

print("Execution complete! Images saved:")
print(" - dendrogram_output.png")
print(" - dbscan_output.png")
print(" - pca_vs_tsne_output.png")
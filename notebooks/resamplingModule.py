import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from collections import Counter

# Utility to convert pandas objects to numpy
try:
    import pandas as pd
except ImportError:
    pd = None

def to_numpy(arr):
    """
    Convert pandas Series/DataFrame or array-like to numpy array.
    """
    if pd is not None and isinstance(arr, (pd.Series, pd.DataFrame)):
        return arr.values
    return np.array(arr)


def cluster_centroid_undersample(X, y, target_count=None, random_state=42):
    """
    Undersample majority class by clustering and using centroids to create a balanced dataset.

    Parameters:
    - X, y: input features and labels
    - target_count: number of centroids to generate; by default equals minority class size
    - random_state: reproducibility
    """
    X = to_numpy(X)
    y = to_numpy(y)
    maj = Counter(y).most_common()[0][0]
    minc = 1 - maj
    X_maj = X[y == maj]
    X_min = X[y == minc]

    # Default target_count = size of minority class to balance
    if target_count is None:
        target_count = len(X_min)

    n_clusters = min(target_count, len(X_maj))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X_maj)
    centroids = kmeans.cluster_centers_

    # Build new dataset: all minority + centroids labelled as majority
    X_new = np.vstack([X_min, centroids])
    y_new = np.concatenate([
        np.full(len(X_min), minc),
        np.full(len(centroids), maj)
    ])
    return X_new, y_new


def simple_smote(X, y, k=5, n_samples=None, random_state=42):
    """
    Generate synthetic minority samples via SMOTE-style interpolation until balanced by default.

    Parameters:
    - k: number of nearest neighbors
    - n_samples: number of synthetic samples; default = difference between classes
    """
    X = to_numpy(X)
    y = to_numpy(y)
    np.random.seed(random_state)

    maj = Counter(y).most_common()[0][0]
    minc = 1 - maj
    X_min = X[y == minc]
    X_maj = X[y == maj]

    if n_samples is None:
        n_samples = max(0, len(X_maj) - len(X_min))
    if n_samples == 0:
        return X.copy(), y.copy()

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(X_min))).fit(X_min)
    indices = nbrs.kneighbors(X_min, return_distance=False)[:, 1:]

    synthetic = []
    for _ in range(n_samples):
        i = np.random.randint(len(X_min))
        j = np.random.choice(indices[i])
        gap = np.random.rand()
        synthetic.append(X_min[i] + gap * (X_min[j] - X_min[i]))

    X_syn = np.vstack(synthetic)
    y_syn = np.full(len(X_syn), minc)

    return np.vstack([X, X_syn]), np.concatenate([y, y_syn])


def kmeans_smote(X, y, n_clusters=None, k=5, random_state=42):
    """
    Cluster minority class then apply SMOTE within each cluster until balanced.

    Parameters:
    - n_clusters: number of KMeans clusters; default = sqrt(n_minority)
    - k: neighbors for SMOTE
    """
    X = to_numpy(X)
    y = to_numpy(y)
    np.random.seed(random_state)

    maj = Counter(y).most_common()[0][0]
    minc = 1 - maj
    X_min = X[y == minc]
    X_maj = X[y == maj]

    # Default clusters = sqrt(minority count)
    if n_clusters is None:
        n_clusters = max(2, int(np.sqrt(len(X_min))))
    n_clusters = min(n_clusters, len(X_min))

    km = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X_min)
    labels = km.labels_

    # Compute how many to sample: balance target
    total_needed = max(0, len(X_maj) - len(X_min))
    per_cluster = total_needed // n_clusters + 1

    synthetic = []
    for c in range(n_clusters):
        pts = X_min[labels == c]
        if len(pts) < 2:
            continue
        nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(pts))).fit(pts)
        nn_idx = nbrs.kneighbors(pts, return_distance=False)[:, 1:]
        for _ in range(per_cluster):
            i = np.random.randint(len(pts))
            j = np.random.choice(nn_idx[i])
            gap = np.random.rand()
            synthetic.append(pts[i] + gap * (pts[j] - pts[i]))

    if not synthetic:
        return X.copy(), y.copy()

    X_syn = np.vstack(synthetic)
    y_syn = np.full(len(X_syn), minc)
    return np.vstack([X, X_syn]), np.concatenate([y, y_syn])


def plot_class_balance(datasets, title="", save_path=None):
    """
    Plot class distribution for different datasets with consistent y-axis scale.
    """
    plt.figure(figsize=(4 * len(datasets), 4))

    # Compute the global max count for consistent y-axis scaling
    global_max = 0
    for _, y_i, _ in datasets:
        y_arr = to_numpy(y_i)
        class_counts = Counter(y_arr)
        max_count = max(class_counts.values())
        global_max = max(global_max, max_count)

    for i, (X_i, y_i, label) in enumerate(datasets, start=1):
        ax = plt.subplot(1, len(datasets), i)
        y_arr = to_numpy(y_i)

        sns.countplot(x=y_arr, hue=y_arr, palette=['#77DD76', '#FF6962'], ax=ax, legend=False)


        ax.set_title(label, fontsize=14)
        for p in ax.patches:
            cnt = p.get_height()
            pct = 100 * cnt / len(y_arr)
            ax.text(p.get_x() + p.get_width()/2, cnt + global_max * 0.01, f"{pct:.2f}%", ha='center')

        ax.set_ylim(0, global_max * 1.15)  # uniform height
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path)
    plt.show()


def resample_and_plot(X, y, title_suffix="", save_path=None):
    """Apply original + three resampling, plot, and return dict of results."""
    X_arr = to_numpy(X)
    y_arr = to_numpy(y)

    datasets = [
        (X_arr, y_arr, "Original"),
        (*cluster_centroid_undersample(X_arr, y_arr), "Cluster Centroids"),
        (*simple_smote(X_arr, y_arr), "SMOTE"),
        (*kmeans_smote(X_arr, y_arr), "KMeans SMOTE"),
    ]

    plot_class_balance(datasets, title=f"Class Distribution {title_suffix}", save_path=save_path)
    return {label: (X_i, y_i) for X_i, y_i, label in datasets}

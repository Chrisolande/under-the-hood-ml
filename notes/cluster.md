# Unsupervised Learning: Clustering Algorithms

Unsupervised learning discovers hidden structure in unlabeled data where no response variable exists. Unlike supervised learning's clear objective functions, unsupervised methods optimize subjective criteria, making evaluation inherently more challenging and domain-dependent.

## I. Core Mathematical Framework

### The Clustering Problem

Given dataset $\mathbf{X} = \{x_1, x_2, ..., x_n\}$ where $x_i \in \mathbb{R}^p$, partition into $K$ clusters $\{C_1, C_2, ..., C_K\}$ such that:

1. **Homogeneity:** Within-cluster similarity is maximized
2. **Separation:** Between-cluster dissimilarity is maximized
3. **Completeness:** $\bigcup_{k=1}^{K} C_k = \{1, ..., n\}$ and $C_k \cap C_j = \emptyset$ for $k \neq j$

**Key insight:** No universal "best" clustering exists, quality depends on problem context and distance metric choice.

### Distance Metrics

The foundation of clustering is measuring dissimilarity between observations.

#### 1. Euclidean Distance (L2 Norm)

$$d(x_i, x_j) = \sqrt{\sum_{l=1}^{p} (x_{il} - x_{jl})^2} = ||x_i - x_j||_2$$

- **Properties:** Sensitive to scale and outliers
- **When to use:** Features have similar units and spherical clusters expected

#### 2. Manhattan Distance (L1 Norm)

$$d(x_i, x_j) = \sum_{l=1}^{p} |x_{il} - x_{jl}|$$

- **Properties:** More robust to outliers than Euclidean
- **When to use:** High-dimensional sparse data, grid-like structures

#### 3. Correlation-Based Distance

$$d(x_i, x_j) = 1 - \text{corr}(x_i, x_j) = 1 - \frac{\sum_{l=1}^{p}(x_{il} - \bar{x}_i)(x_{jl} - \bar{x}_j)}{\sqrt{\sum_{l=1}^{p}(x_{il} - \bar{x}_i)^2}\sqrt{\sum_{l=1}^{p}(x_{jl} - \bar{x}_j)^2}}$$

- **Properties:** Focuses on pattern/shape rather than magnitude
- **When to use:** Gene expression, time series where profile shape matters
- **Range:** $[0, 2]$ where 0 = perfectly correlated, 2 = perfectly anti-correlated

#### 4. Cosine Distance

$$d(x_i, x_j) = 1 - \frac{x_i \cdot x_j}{||x_i|| \cdot ||x_j||} = 1 - \frac{\sum_{l=1}^{p} x_{il} x_{jl}}{\sqrt{\sum_{l=1}^{p} x_{il}^2}\sqrt{\sum_{l=1}^{p} x_{jl}^2}}$$

- **When to use:** Text data, high-dimensional sparse vectors (TF-IDF)

## II. K-Means Clustering

### Mathematical Formulation

**Objective:** Minimize total within-cluster sum of squares (WCSS):

$$\min_{C_1, ..., C_K} \sum_{k=1}^{K} \sum_{i \in C_k} ||x_i - \mu_k||^2 = \min_{C_1, ..., C_K} \sum_{k=1}^{K} |C_k| \cdot \text{Var}(C_k)$$

where $\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x_i$ is the centroid of cluster $k$.

**Alternative formulation (pairwise distances):**

$$\min_{C_1, ..., C_K} \sum_{k=1}^{K} \frac{1}{|C_k|} \sum_{i, j \in C_k} ||x_i - x_j||^2$$

This shows K-means minimizes average pairwise squared distance within clusters.

### Lloyd's Algorithm

The standard K-means algorithm alternates between two steps:

```
Initialize: Randomly select K centroids {μ₁, ..., μₖ}

Repeat until convergence:
    # E-step: Assign points to nearest centroid
    For each point xᵢ:
        Cₖ ← {i : ||xᵢ - μₖ|| ≤ ||xᵢ - μⱼ|| for all j}
    
    # M-step: Update centroids
    For each cluster k:
        μₖ ← (1/|Cₖ|) Σᵢ∈Cₖ xᵢ
```

**Convergence properties:**

- Guaranteed to converge to local minimum
- Each iteration decreases WCSS: $\text{WCSS}^{(t+1)} \leq \text{WCSS}^{(t)}$
- Typically converges in < 100 iterations
- **No guarantee of global optimum**

**Computational complexity:**

- Per iteration: $O(nKp)$ where $n$ = samples, $K$ = clusters, $p$ = features
- Total: $O(nKpi)$ where $i$ = iterations

### Initialization Methods

Poor initialization can trap K-means in bad local minima.

#### 1. Random Initialization

Simple but unreliable: randomly select $K$ points as initial centroids.

#### 2. K-Means++ (Recommended)

**Algorithm:**

1. Choose first centroid $\mu_1$ uniformly at random from $\mathbf{X}$
2. For each remaining centroid $k = 2, ..., K$:
   - Compute $D(x_i)^2 = \min_{j<k} ||x_i - \mu_j||^2$ for each point
   - Select $\mu_k$ with probability $\propto D(x_i)^2$
3. Run standard K-means

**Why it works:** Spreads initial centroids far apart, ensuring better coverage of data space.

**Theoretical guarantee:** Expected WCSS is $O(\log K)$ times optimal (Arthur & Vassilvitskii, 2007).

```python
from sklearn.cluster import KMeans

# K-means++ initialization (default in scikit-learn)
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
labels = kmeans.fit_predict(X)
```

### Determining Optimal K

#### 1. Elbow Method

Plot WCSS vs. $K$ and look for "elbow" where marginal improvement decreases sharply.

**Inertia (WCSS):**

$$\text{Inertia} = \sum_{k=1}^{K} \sum_{i \in C_k} ||x_i - \mu_k||^2$$

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, marker='o')
plt.xlabel('Number of clusters K')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method')
plt.show()
```

**Limitation:** Elbow often ambiguous, especially with high-dimensional data.

#### 2. Silhouette Analysis

Measures how similar a point is to its own cluster compared to other clusters.

**Silhouette coefficient for point $i$:**

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

where:

- $a(i)$ = average distance to points in same cluster (cohesion)
- $b(i)$ = average distance to points in nearest different cluster (separation)

**Range:** $[-1, 1]$

- $s(i) \approx 1$: Point well-clustered
- $s(i) \approx 0$: Point on cluster boundary
- $s(i) < 0$: Point likely misclassified

**Average silhouette score:**

$$\bar{s} = \frac{1}{n} \sum_{i=1}^{n} s(i)$$

```python
from sklearn.metrics import silhouette_score, silhouette_samples

silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# Visualize per-sample silhouette coefficients
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X)
silhouette_vals = silhouette_samples(X, labels)

y_lower = 10
fig, ax = plt.subplots(figsize=(10, 6))

for i in range(k):
    cluster_silhouette_vals = silhouette_vals[labels == i]
    cluster_silhouette_vals.sort()
    
    size_cluster = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster
    
    color = cm.nipy_spectral(float(i) / k)
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                      0, cluster_silhouette_vals,
                      facecolor=color, alpha=0.7)
    y_lower = y_upper + 10

ax.set_xlabel("Silhouette coefficient")
ax.set_ylabel("Cluster label")
ax.axvline(x=silhouette_score(X, labels), color="red", linestyle="--")
plt.show()
```

#### 3. Gap Statistic

Compares WCSS to expected WCSS under null reference distribution.

$$\text{Gap}(K) = \mathbb{E}_{\text{ref}}[\log(\text{WCSS}_K)] - \log(\text{WCSS}_K)$$

**Algorithm:**

1. Cluster observed data for $K = 1, ..., K_{\max}$
2. Generate $B$ reference datasets (uniform random over data range)
3. Cluster each reference dataset
4. Compute Gap statistic for each $K$
5. Choose $K$ where Gap is largest and exceeds Gap$(K+1) - s_{K+1}$

```python
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

def compute_gap(X, K_range, B=10):
    gaps = []
    s_k = []
    
    for k in K_range:
        # Observed WCSS
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss_obs = kmeans.inertia_
        
        # Reference WCSS
        wcss_refs = []
        for _ in range(B):
            X_ref = np.random.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
            kmeans_ref = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_ref.fit(X_ref)
            wcss_refs.append(kmeans_ref.inertia_)
        
        gap = np.log(np.mean(wcss_refs)) - np.log(wcss_obs)
        gaps.append(gap)
        s_k.append(np.std(np.log(wcss_refs)) * np.sqrt(1 + 1/B))
    
    return np.array(gaps), np.array(s_k)

K_range = range(1, 11)
gaps, s_k = compute_gap(X, K_range, B=10)

# Optimal K: first k where Gap(k) >= Gap(k+1) - s_{k+1}
optimal_k = next(k for k in K_range[:-1] if gaps[k-1] >= gaps[k] - s_k[k])
```

### K-Means Variants

#### 1. Mini-Batch K-Means

Uses random subsets (mini-batches) for faster convergence on large datasets.

**Complexity:** $O(nKp) \rightarrow O(bKpi)$ where $b << n$ is batch size.

```python
from sklearn.cluster import MiniBatchKMeans

mbkmeans = MiniBatchKMeans(n_clusters=5, batch_size=100, random_state=42)
labels = mbkmeans.fit_predict(X)
```

**Trade-off:** 3-10x speedup with slightly worse cluster quality.

#### 2. K-Medoids (PAM - Partitioning Around Medoids)

Uses actual data points as cluster centers (medoids) instead of centroids.

**Advantages:**

- More robust to outliers
- Works with arbitrary distance metrics (not just Euclidean)
- Medoids are interpretable (actual observations)

**Disadvantages:**

- Higher computational cost: $O(K(n-K)^2pi)$

```python
from sklearn_extra.cluster import KMedoids

kmedoids = KMedoids(n_clusters=5, metric='euclidean', random_state=42)
labels = kmedoids.fit_predict(X)
medoid_indices = kmedoids.medoid_indices_
```

### K-Means Limitations

1. **Assumes spherical clusters:** Struggles with elongated or irregular shapes
2. **Sensitive to scale:** Features with large variance dominate distance calculations
3. **Requires pre-specified K:** No automatic determination
4. **Sensitive to outliers:** Single outlier can distort centroid
5. **Hard assignments:** Each point belongs to exactly one cluster
6. **Local minima:** Results depend on initialization

**Solutions:**

- Spherical clusters → Use DBSCAN, GMM, or spectral clustering
- Scale sensitivity → Standardize features before clustering
- Unknown K → Use elbow/silhouette methods or hierarchical clustering
- Outliers → Use K-medoids or remove outliers first
- Hard assignments → Use Gaussian Mixture Models for soft clustering

## III. Hierarchical Clustering

Builds nested hierarchy of clusters without pre-specifying $K$, visualized as dendrogram.

### Mathematical Framework

#### Linkage Criteria

Define dissimilarity $d(C_A, C_B)$ between clusters $C_A$ and $C_B$:

**1. Complete Linkage (Maximum)**

$$d(C_A, C_B) = \max_{i \in C_A, j \in C_B} ||x_i - x_j||$$

- **Properties:** Tends to produce compact, balanced clusters
- **Sensitivity:** Less sensitive to outliers than single linkage

**2. Single Linkage (Minimum)**

$$d(C_A, C_B) = \min_{i \in C_A, j \in C_B} ||x_i - x_j||$$

- **Properties:** Can detect elongated clusters
- **Problem:** Chain effect - produces trailing clusters

**3. Average Linkage (UPGMA)**

$$d(C_A, C_B) = \frac{1}{|C_A| \cdot |C_B|} \sum_{i \in C_A} \sum_{j \in C_B} ||x_i - x_j||$$

- **Properties:** Compromise between single and complete
- **Most commonly used** in practice

**4. Ward's Linkage (Minimum Variance)**

Minimizes within-cluster variance increase when merging clusters.

$$d(C_A, C_B) = \frac{|C_A| \cdot |C_B|}{|C_A| + |C_B|} ||\mu_A - \mu_B||^2$$

- **Properties:** Tends to produce equal-sized clusters
- **Connection to K-means:** Minimizes same objective function (WCSS)

### Agglomerative Algorithm

```
Initialize: Each point is its own cluster (n clusters)

Repeat until 1 cluster remains:
    1. Compute pairwise dissimilarities between all clusters
    2. Merge two closest clusters C_A and C_B
    3. Update dissimilarity matrix
    4. Record merge height h = d(C_A, C_B)
    
Output: Dendrogram encoding all merges
```

**Computational complexity:**

- Naive: $O(n^3)$
- Optimized (with priority queue): $O(n^2 \log n)$

### Python Implementation

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

# Compute linkage matrix
Z = linkage(X, method='ward', metric='euclidean')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(Z, 
           truncate_mode='lastp',
           p=30,
           leaf_font_size=10,
           show_contracted=True)
plt.xlabel('Sample index or (cluster size)')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram (Ward)')
plt.show()

# Cut dendrogram to get K clusters
K = 5
labels = fcluster(Z, K, criterion='maxclust')

# Alternative: cut at specific height
height = 10.0
labels = fcluster(Z, height, criterion='distance')
```

### Dendrogram Interpretation

**Reading dendrograms:**

- **Height of merge:** Dissimilarity between clusters
- **Horizontal line length:** Duration cluster remains intact
- **Cutting dendrogram:** Horizontal line intersecting $K$ vertical lines yields $K$ clusters

**Example interpretation:**

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# Generate dendrogram with labels
fig, ax = plt.subplots(figsize=(12, 6))
dend = dendrogram(Z, labels=sample_names, leaf_font_size=8)

# Add horizontal line at cut height
cut_height = 15.0
ax.axhline(y=cut_height, color='r', linestyle='--', label=f'Cut at h={cut_height}')
ax.legend()
plt.show()

# Identify clusters at cut height
clusters = fcluster(Z, cut_height, criterion='distance')
print(f"Number of clusters: {len(np.unique(clusters))}")
```

### Cophenetic Correlation

Measures how well dendrogram preserves original pairwise distances.

$$r_c = \text{corr}(d_{ij}, c_{ij})$$

where:

- $d_{ij}$ = original distance between points $i$ and $j$
- $c_{ij}$ = cophenetic distance (height where $i$ and $j$ first merge)

**Interpretation:**

- $r_c$ close to 1: Dendrogram accurately represents data structure
- $r_c < 0.8$: Dendrogram may be misleading

```python
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

# Compute cophenetic correlation
c, coph_dists = cophenet(Z, pdist(X))
print(f"Cophenetic correlation: {c:.3f}")
```

### Hierarchical vs. K-Means

| Aspect | Hierarchical | K-Means |
|--------|-------------|---------|
| **K specification** | Not required | Required |
| **Scalability** | $O(n^2 \log n)$ | $O(nKpi)$ |
| **Flexibility** | Cut dendrogram at any level | Fixed K |
| **Reproducibility** | Deterministic | Depends on initialization |
| **Cluster shape** | Any shape (with single linkage) | Spherical |
| **Visualization** | Dendrogram | Scatter plots |
| **Best for** | Small datasets, exploration | Large datasets, speed |

## IV. DBSCAN (Density-Based Clustering)

**Density-Based Spatial Clustering of Applications with Noise** discovers clusters of arbitrary shape and automatically detects outliers.

### Core Concepts

#### 1. $\epsilon$-Neighborhood

$$N_\epsilon(x_i) = \{x_j \in \mathbf{X} : d(x_i, x_j) \leq \epsilon\}$$

The set of points within distance $\epsilon$ from point $x_i$.

#### 2. Core Points

Point $x_i$ is a **core point** if:

$$|N_\epsilon(x_i)| \geq \text{min\_samples}$$

Core points are in dense regions.

#### 3. Border Points

Point $x_i$ is a **border point** if:

- $|N_\epsilon(x_i)| < \text{min\_samples}$ (not core)
- $x_i \in N_\epsilon(x_j)$ for some core point $x_j$

Border points are on cluster edges.

#### 4. Noise Points

Points that are neither core nor border points.

#### 5. Density Reachability

Point $x_j$ is **directly density-reachable** from $x_i$ if:

- $x_j \in N_\epsilon(x_i)$
- $x_i$ is a core point

Point $x_j$ is **density-reachable** from $x_i$ if there exists a chain:

$$x_i = p_1, p_2, ..., p_m = x_j$$

where each $p_{k+1}$ is directly density-reachable from $p_k$.

#### 6. Density Connectivity

Points $x_i$ and $x_j$ are **density-connected** if there exists point $x_k$ such that both $x_i$ and $x_j$ are density-reachable from $x_k$.

### DBSCAN Algorithm

```
Mark all points as unvisited

For each unvisited point p:
    Mark p as visited
    
    If |N_ε(p)| < min_samples:
        Mark p as noise (tentatively)
        Continue
    
    # p is core point - start new cluster
    Create new cluster C
    Add p to C
    
    SeedSet = N_ε(p) \ {p}
    
    For each point q in SeedSet:
        If q is unvisited:
            Mark q as visited
            If |N_ε(q)| >= min_samples:
                SeedSet = SeedSet ∪ N_ε(q)
        
        If q not yet in any cluster:
            Add q to C
```

**Key properties:**

- **Deterministic** (given fixed point ordering)
- **One pass** through data
- **Complexity:** $O(n \log n)$ with spatial indexing, $O(n^2)$ naive

### Python Implementation

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

# Standardize features (important for DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
labels = dbscan.fit_predict(X_scaled)

# Identify noise points
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# Core sample indices
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
```

### Parameter Selection

#### Finding Optimal $\epsilon$

**K-distance plot method:**

1. For each point, compute distance to its $k$-th nearest neighbor
2. Sort distances in ascending order
3. Plot sorted $k$-distances
4. Look for "elbow" point - rapid increase indicates good $\epsilon$

```python
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Choose k = min_samples
k = 5
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X_scaled)
distances, indices = neighbors.kneighbors(X_scaled)

# Sort distances to k-th nearest neighbor
distances = np.sort(distances[:, k-1], axis=0)

# Plot k-distance graph
plt.plot(distances)
plt.xlabel('Points sorted by distance')
plt.ylabel(f'{k}-th nearest neighbor distance')
plt.title('K-distance Graph')
plt.grid(True)
plt.show()

# Elbow typically suggests good epsilon
epsilon = distances[int(len(distances) * 0.95)]  # 95th percentile
print(f"Suggested epsilon: {epsilon:.3f}")
```

#### Choosing min_samples

**Rule of thumb:** $\text{min\_samples} \geq \text{dimensionality} + 1$

- Lower values: More small clusters, sensitive to noise
- Higher values: Fewer clusters, more points classified as noise
- Common choice: 4-10 for most applications

### DBSCAN Advantages and Limitations

**Strengths:**

1. **No K specification:** Automatically determines number of clusters
2. **Arbitrary shapes:** Handles non-convex, elongated clusters
3. **Noise detection:** Explicitly identifies outliers
4. **Robust to outliers:** Outliers don't influence cluster formation
5. **Single pass:** Efficient for large datasets with spatial indexing

**Weaknesses:**

1. **Parameter sensitivity:** Results highly dependent on $\epsilon$ and min_samples
2. **Varying densities:** Struggles when clusters have different densities
3. **High dimensions:** Distance metrics become less meaningful (curse of dimensionality)
4. **Memory intensive:** Requires distance matrix or spatial index

### DBSCAN Variants

#### HDBSCAN (Hierarchical DBSCAN)

Extends DBSCAN to handle varying density clusters.

**Key idea:** Build hierarchy of DBSCAN clusterings at all $\epsilon$ values, then extract stable clusters.

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5)
labels = clusterer.fit_predict(X_scaled)

# Outlier scores (higher = more likely outlier)
outlier_scores = clusterer.outlier_scores_

# Cluster persistence (stability)
persistence = clusterer.cluster_persistence_
```

**Advantages over DBSCAN:**

- Handles varying density clusters
- More robust parameter: only min_cluster_size needed
- Provides soft clustering via membership probabilities

## V. Gaussian Mixture Models (GMM)

GMMs perform **soft clustering** by modeling data as mixture of Gaussian distributions.

### Mathematical Framework

**Model assumption:** Data generated from mixture of $K$ Gaussian components:

$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

where:

- $\pi_k$ = mixing coefficient (weight) of component $k$, $\sum_{k=1}^{K} \pi_k = 1$
- $\mu_k$ = mean vector of component $k$
- $\Sigma_k$ = covariance matrix of component $k$
- $\mathcal{N}(x | \mu, \Sigma) = \frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)$

**Latent variable:** $z_i \in \{1, ..., K\}$ indicates which component generated $x_i$.

### Expectation-Maximization (EM) Algorithm

Since latent variables $z_i$ are unknown, use EM to find maximum likelihood estimates.

#### E-Step: Compute Responsibilities

Compute posterior probability that point $x_i$ belongs to component $k$:

$$\gamma_{ik} = p(z_i = k | x_i) = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}$$

This is the **responsibility** of component $k$ for point $i$.

#### M-Step: Update Parameters

Given responsibilities, update parameters:

$$\mu_k = \frac{\sum_{i=1}^{n} \gamma_{ik} x_i}{\sum_{i=1}^{n} \gamma_{ik}}$$

$$\Sigma_k = \frac{\sum_{i=1}^{n} \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_{i=1}^{n} \gamma_{ik}}$$

$$\pi_k = \frac{1}{n} \sum_{i=1}^{n} \gamma_{ik}$$

**Iterate** E and M steps until convergence (log-likelihood stops improving).

### Python Implementation

```python
from sklearn.mixture import GaussianMixture
import numpy as np

# Fit GMM
gmm = GaussianMixture(
    n_components=5,
    covariance_type='full',  # full, tied, diag, spherical
    n_init=10,
    random_state=42
)
gmm.fit(X)

# Hard clustering (assign to most probable component)
labels = gmm.predict(X)

# Soft clustering (responsibilities)
probabilities = gmm.predict_proba(X)

# Component parameters
means = gmm.means_
covariances = gmm.covariances_
weights = gmm.weights_

print(f"Converged in {gmm.n_iter_} iterations")
print(f"Log-likelihood: {gmm.score(X):.3f}")
```

### Covariance Types

| Type | Parameters | Constraint | Use Case |
|------|-----------|------------|----------|
| `full` | $K \cdot p(p+1)/2$ | None | General ellipsoidal clusters |
| `tied` | $p(p+1)/2$ | $\Sigma_k = \Sigma$ for all $k$ | Clusters same shape/orientation |
| `diag` | $K \cdot p$ | $\Sigma_k$ diagonal | Axis-aligned ellipsoids |
| `spherical` | $K$ | $\Sigma_k = \sigma_k^2 I$ | Spherical clusters |

**Trade-off:** More parameters → better fit but higher variance and longer training.

### Model Selection: BIC and AIC

**Akaike Information Criterion (AIC):**

$$\text{AIC} = 2m - 2\ln(\hat{L})$$

**Bayesian Information Criterion (BIC):**

$$\text{BIC} = m \ln(n) - 2\ln(\hat{L})$$

where:

- $m$ = number of parameters
- $n$ = number of samples
- $\hat{L}$ = maximum likelihood of the model

**Lower values indicate better models.** BIC penalizes complexity more heavily than AIC.

```python
from sklearn.mixture import GaussianMixture
import numpy as np

n_components_range = range(1, 11)
bic_scores = []
aic_scores = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
    aic_scores.append(gmm.aic(X))

# Optimal K minimizes BIC/AIC
optimal_k_bic = n_components_range[np.argmin(bic_scores)]
optimal_k_aic = n_components_range[np.argmin(aic_scores)]

import matplotlib.pyplot as plt
plt.plot(n_components_range, bic_scores, label='BIC', marker='o')
plt.plot(n_components_range, aic_scores, label='AIC', marker='s')
plt.xlabel('Number of components')
plt.ylabel('Information criterion')
plt.legend()
plt.show()
```

### Anomaly Detection with GMM

Low probability density indicates anomalies.

```python
# Compute log probability density for each point
densities = gmm.score_samples(X)

# Identify anomalies (lowest 5% densities)
threshold = np.percentile(densities, 5)
anomalies = densities < threshold

print(f"Number of anomalies: {anomalies.sum()}")

# Visualize anomalies
plt.scatter(X[~anomalies, 0], X[~anomalies, 1], c='blue', alpha=0.5, label='Normal')
plt.scatter(X[anomalies, 0], X[anomalies, 1], c='red', marker='x', s=100, label='Anomaly')
plt.legend()
plt.show()
```

### GMM vs. K-Means

| Aspect | GMM | K-Means |
|--------|-----|---------|
| **Assignment** | Soft (probabilistic) | Hard (deterministic) |
| **Cluster shape** | Ellipsoidal | Spherical |
| **Flexibility** | Different sizes/orientations | Equal size preferred |
| **Output** | Probabilities | Labels only |
| **Complexity** | Higher | Lower |
| **Convergence** | Slower | Faster |
| **Use case** | Complex clusters, anomaly detection | Simple, fast clustering |

**When to use GMM:**

- Need uncertainty estimates
- Clusters have different shapes/sizes
- Anomaly detection required
- Want generative model

## VI. Spectral Clustering

Handles non-convex clusters by transforming data into graph representation.

### Algorithm Overview

1. **Build similarity graph:** Compute affinity matrix $W$ where $W_{ij} = \exp(-||x_i - x_j||^2 / 2\sigma^2)$
2. **Compute graph Laplacian:** $L = D - W$ where $D$ is degree matrix
3. **Eigen-decomposition:** Find first $K$ eigenvectors of $L$
4. **Cluster in eigenspace:** Run K-means on eigenvector matrix

### Mathematical Framework

**Normalized graph Laplacian:**

$L_{\text{norm}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}$

**Key insight:** Eigenvectors corresponding to smallest eigenvalues provide low-dimensional embedding that preserves cluster structure.

### Python Implementation

```python
from sklearn.cluster import SpectralClustering
import numpy as np

# Spectral clustering
spectral = SpectralClustering(
    n_clusters=5,
    affinity='rbf',  # or 'nearest_neighbors', 'precomputed'
    gamma=1.0,       # RBF kernel parameter
    n_init=10,
    random_state=42
)
labels = spectral.fit_predict(X)

# Custom affinity matrix
from sklearn.metrics.pairwise import rbf_kernel
affinity_matrix = rbf_kernel(X, gamma=0.5)

spectral_custom = SpectralClustering(
    n_clusters=5,
    affinity='precomputed',
    random_state=42
)
labels = spectral_custom.fit_predict(affinity_matrix)
```

### Advantages and Limitations

**Strengths:**

- Handles non-convex clusters
- Based on graph theory (interpretable)
- Works well for image segmentation

**Weaknesses:**

- Computationally expensive: $O(n^3)$ for eigen-decomposition
- Requires similarity metric choice
- Memory intensive for large $n$

## VII. Clustering Evaluation Metrics

### Internal Metrics (No Ground Truth)

#### 1. Silhouette Score

Already covered in K-means section. Range: $[-1, 1]$, higher is better.

#### 2. Davies-Bouldin Index

Measures average similarity between each cluster and its most similar cluster:

$\text{DB} = \frac{1}{K} \sum_{k=1}^{K} \max_{j \neq k} \left( \frac{\sigma_k + \sigma_j}{d(c_k, c_j)} \right)$

where:

- $\sigma_k$ = average distance of points in cluster $k$ to centroid $c_k$
- $d(c_k, c_j)$ = distance between centroids

**Lower values indicate better clustering.**

```python
from sklearn.metrics import davies_bouldin_score

db_score = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Index: {db_score:.3f}")
```

#### 3. Calinski-Harabasz Index (Variance Ratio)

Ratio of between-cluster dispersion to within-cluster dispersion:

$\text{CH} = \frac{\text{Tr}(B_K)}{\text{Tr}(W_K)} \cdot \frac{n - K}{K - 1}$

where:

- $B_K$ = between-cluster dispersion matrix
- $W_K$ = within-cluster dispersion matrix

**Higher values indicate better clustering.**

```python
from sklearn.metrics import calinski_harabasz_score

ch_score = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz Index: {ch_score:.3f}")
```

### External Metrics (With Ground Truth)

When true labels $y$ are available (e.g., for validation).

#### 1. Adjusted Rand Index (ARI)

Measures similarity between two clusterings, adjusted for chance:

$\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}$

**Range:** $[-1, 1]$ where:

- 1 = perfect match
- 0 = random labeling
- Negative = worse than random

```python
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(y_true, labels)
print(f"Adjusted Rand Index: {ari:.3f}")
```

#### 2. Normalized Mutual Information (NMI)

Information-theoretic measure of clustering agreement:

$\text{NMI}(Y, C) = \frac{2 \cdot I(Y; C)}{H(Y) + H(C)}$

where $I(Y; C)$ is mutual information and $H(\cdot)$ is entropy.

**Range:** $[0, 1]$ where 1 = perfect match.

```python
from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score(y_true, labels)
print(f"Normalized Mutual Information: {nmi:.3f}")
```

#### 3. Fowlkes-Mallows Index

Geometric mean of precision and recall:

$\text{FMI} = \sqrt{\frac{TP}{TP + FP} \cdot \frac{TP}{TP + FN}}$

**Range:** $[0, 1]$ where 1 = perfect match.

```python
from sklearn.metrics import fowlkes_mallows_score

fmi = fowlkes_mallows_score(y_true, labels)
print(f"Fowlkes-Mallows Index: {fmi:.3f}")
```

## VIII. Complete Clustering Pipeline

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from dataclasses import dataclass
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    labels: np.ndarray
    n_clusters: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    algorithm: str


class ClusteringPipeline:
    def __init__(self, X: np.ndarray, scale: bool = True):
        self.X_raw = X
        self.scaler = StandardScaler() if scale else None
        self.X = self.scaler.fit_transform(X) if scale else X
        logger.info(f"Initialized pipeline with {X.shape[0]} samples, {X.shape[1]} features")
    
    def kmeans_clustering(self, k: int, **kwargs) -> ClusteringResult:
        logger.info(f"Running K-Means with k={k}")
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42, **kwargs)
        labels = kmeans.fit_predict(self.X)
        return self._evaluate(labels, 'K-Means')
    
    def dbscan_clustering(self, eps: float, min_samples: int, **kwargs) -> ClusteringResult:
        logger.info(f"Running DBSCAN with eps={eps}, min_samples={min_samples}")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        labels = dbscan.fit_predict(self.X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"Found {n_clusters} clusters, {list(labels).count(-1)} noise points")
        return self._evaluate(labels, 'DBSCAN')
    
    def gmm_clustering(self, k: int, **kwargs) -> ClusteringResult:
        logger.info(f"Running GMM with k={k}")
        gmm = GaussianMixture(n_components=k, random_state=42, **kwargs)
        labels = gmm.fit_predict(self.X)
        return self._evaluate(labels, 'GMM')
    
    def hierarchical_clustering(self, k: int, linkage: str = 'ward', **kwargs) -> ClusteringResult:
        logger.info(f"Running Hierarchical clustering with k={k}, linkage={linkage}")
        hier = AgglomerativeClustering(n_clusters=k, linkage=linkage, **kwargs)
        labels = hier.fit_predict(self.X)
        return self._evaluate(labels, f'Hierarchical-{linkage}')
    
    def _evaluate(self, labels: np.ndarray, algorithm: str) -> ClusteringResult:
        # Filter out noise points for DBSCAN
        valid_mask = labels != -1
        n_clusters = len(set(labels[valid_mask]))
        
        if n_clusters > 1 and valid_mask.sum() > n_clusters:
            sil = silhouette_score(self.X[valid_mask], labels[valid_mask])
            db = davies_bouldin_score(self.X[valid_mask], labels[valid_mask])
            ch = calinski_harabasz_score(self.X[valid_mask], labels[valid_mask])
        else:
            sil = db = ch = np.nan
        
        logger.info(f"{algorithm}: Silhouette={sil:.3f}, DB={db:.3f}, CH={ch:.1f}")
        
        return ClusteringResult(
            labels=labels,
            n_clusters=n_clusters,
            silhouette=sil,
            davies_bouldin=db,
            calinski_harabasz=ch,
            algorithm=algorithm
        )
    
    def find_optimal_k_kmeans(self, k_range: range) -> Tuple[int, np.ndarray]:
        silhouette_scores = []
        
        for k in k_range:
            result = self.kmeans_clustering(k)
            silhouette_scores.append(result.silhouette)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        logger.info(f"Optimal K (by silhouette): {optimal_k}")
        
        return optimal_k, np.array(silhouette_scores)
    
    def compare_algorithms(self, k: int = 5) -> pd.DataFrame:
        results = []
        
        results.append(self.kmeans_clustering(k))
        results.append(self.gmm_clustering(k))
        results.append(self.hierarchical_clustering(k, linkage='ward'))
        results.append(self.hierarchical_clustering(k, linkage='average'))
        
        # DBSCAN with auto-tuned epsilon
        eps = self._estimate_epsilon()
        results.append(self.dbscan_clustering(eps=eps, min_samples=5))
        
        df = pd.DataFrame([
            {
                'Algorithm': r.algorithm,
                'N_Clusters': r.n_clusters,
                'Silhouette': r.silhouette,
                'Davies_Bouldin': r.davies_bouldin,
                'Calinski_Harabasz': r.calinski_harabasz
            }
            for r in results
        ])
        
        return df.sort_values('Silhouette', ascending=False)
    
    def _estimate_epsilon(self) -> float:
        from sklearn.neighbors import NearestNeighbors
        
        k = 5
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(self.X)
        distances, _ = neighbors.kneighbors(self.X)
        distances = np.sort(distances[:, k-1])
        
        # Use 95th percentile as epsilon
        eps = distances[int(len(distances) * 0.95)]
        logger.info(f"Estimated epsilon: {eps:.3f}")
        return eps


if __name__ == "__main__":
    # Generate sample data
    from sklearn.datasets import make_blobs
    
    X, y_true = make_blobs(n_samples=500, centers=5, n_features=2, 
                           cluster_std=1.0, random_state=42)
    
    # Initialize pipeline
    pipeline = ClusteringPipeline(X, scale=True)
    
    # Compare algorithms
    comparison = pipeline.compare_algorithms(k=5)
    print("\nAlgorithm Comparison:")
    print(comparison.to_string(index=False))
    
    # Find optimal K for K-Means
    optimal_k, silhouettes = pipeline.find_optimal_k_kmeans(range(2, 11))
    print(f"\nOptimal K: {optimal_k}")
    
    # Run best algorithm
    best_result = pipeline.kmeans_clustering(optimal_k)
    
    # Visualize
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
    plt.title('True Labels')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=best_result.labels, cmap='viridis', alpha=0.6)
    plt.title(f'{best_result.algorithm} (K={optimal_k})')
    plt.tight_layout()
    plt.show()
```

## IX. Practical Recommendations

### Choosing the Right Algorithm

**Decision tree:**

```
Is K known?
├─ YES
│  ├─ Need probabilities/soft clustering? → GMM
│  ├─ Fast and simple? → K-Means
│  └─ Want hierarchy/dendrogram? → Hierarchical
└─ NO
   ├─ Arbitrary cluster shapes? → DBSCAN/HDBSCAN
   ├─ Non-convex clusters? → Spectral Clustering
   └─ Want to explore different K? → Hierarchical + Dendrogram
```

### Pre-processing Checklist

1. **Feature scaling:** Essential for distance-based methods (K-means, DBSCAN, hierarchical)
2. **Handle missing values:** Impute or remove before clustering
3. **Dimensionality reduction:** Consider PCA if $p > 50$ to combat curse of dimensionality
4. **Outlier removal:** Remove extreme outliers before K-means/GMM; keep for DBSCAN

### Common Pitfalls

1. **Not scaling features:** Causes features with large ranges to dominate
2. **Ignoring cluster validation:** Always use multiple metrics
3. **Over-interpreting small differences:** Clustering is subjective
4. **Forcing inappropriate K:** Use elbow/silhouette methods
5. **Using Euclidean distance for categorical data:** Use appropriate distance metrics

### Algorithm Summary Table

| Algorithm | Complexity | Cluster Shape | K Required | Handles Outliers | Scalability |
|-----------|------------|---------------|------------|------------------|-------------|
| **K-Means** | $O(nKpi)$ | Spherical | Yes | No | Excellent |
| **Mini-Batch K-Means** | $O(bKpi)$ | Spherical | Yes | No | Excellent |
| **Hierarchical** | $O(n^2 \log n)$ | Any | No | No | Poor |
| **DBSCAN** | $O(n \log n)$ | Arbitrary | No | Yes | Good |
| **HDBSCAN** | $O(n \log n)$ | Arbitrary | No | Yes | Good |
| **GMM** | $O(nK^2pi)$ | Ellipsoidal | Yes | Moderate | Good |
| **Spectral** | $O(n^3)$ | Non-convex | Yes | No | Poor |

### Summary

Clustering is an exploratory technique with no single "correct" answer. Success depends on:

1. **Domain knowledge:** Understanding what makes a "good" cluster in your context
2. **Algorithm selection:** Match algorithm assumptions to data characteristics
3. **Validation:** Use multiple metrics and visual inspection
4. **Iteration:** Clustering is iterative, refine based on results

**Key principle:** The best clustering is one that provides actionable insights for your specific problem, not necessarily the one with the highest metric scores.

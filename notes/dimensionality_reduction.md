# Dimensionality Reduction

Dimensionality reduction (DR) transforms high-dimensional data into lower-dimensional representations while preserving essential information. These unsupervised learning techniques combat the curse of dimensionality, accelerate model training, and enable data visualization.

## I. The Curse of Dimensionality

High-dimensional datasets ($p \gg n$ or $p$ very large) create fundamental statistical and computational challenges.

### Mathematical Foundation

**Distance Concentration:** In high dimensions, distances between points become similar. For random points in a $p$-dimensional unit hypercube:

$$\frac{\text{max}_i d_i - \text{min}_i d_i}{\text{min}_i d_i} \to 0 \text{ as } p \to \infty$$

**Volume Paradox:** A hypercube with edge length 0.001 in 100 dimensions occupies:

$$\text{Volume}_{\text{small}} = (0.001)^{100} = 10^{-300}$$

While a hypercube with edge 0.999 occupies $(0.999)^{100} \approx 0.905$ of the unit hypercube.

### Practical Consequences

1. **Sparsity:** Training instances become increasingly isolated, making reliable predictions require large extrapolations
2. **Sample Size Requirements:** To maintain density $\rho$, samples needed grow as $n \propto \rho^{-p}$ (exponential in $p$)
3. **Overfitting Risk:** Models fit noise rather than signal when $p$ approaches or exceeds $n$
4. **Computational Burden:** Training time scales as $O(n \cdot p^2)$ or worse for many algorithms

## II. Mathematical Framework

### Projection vs. Manifold Learning

**Projection** assumes data lies in/near a lower-dimensional linear subspace:

$$\mathbf{Z} = \mathbf{X}\mathbf{W}$$

where $\mathbf{X} \in \mathbb{R}^{n \times p}$, $\mathbf{W} \in \mathbb{R}^{p \times M}$, $\mathbf{Z} \in \mathbb{R}^{n \times M}$, and $M \ll p$.

**Manifold Learning** assumes data lies on a curved $d$-dimensional manifold embedded in $\mathbb{R}^p$ where $d \ll p$:

$$\mathcal{M} = \{\mathbf{x} \in \mathbb{R}^p : \mathbf{x} = f(\mathbf{t}), \mathbf{t} \in \mathbb{R}^d\}$$

where $f: \mathbb{R}^d \to \mathbb{R}^p$ is a smooth, continuous mapping.

**Key Insight:** Projection fails when the manifold is highly nonlinear (e.g., Swiss roll). Manifold methods "unroll" curved structures.

## III. Principal Component Analysis (PCA)

PCA finds orthogonal directions of maximum variance through eigendecomposition or SVD.

### Mathematical Formulation

**Objective:** Find $M$ orthonormal directions $\{\phi_m\}_{m=1}^M$ that maximize captured variance.

**The $m$-th Principal Component:**

$$Z_m = \sum_{j=1}^p \phi_{jm}X_j = \mathbf{X}\phi_m$$

where $\phi_m = (\phi_{1m}, \phi_{2m}, \ldots, \phi_{pm})^T$ are the **loadings** with $\|\phi_m\|^2 = \sum_{j=1}^p \phi_{jm}^2 = 1$.

**Optimization Problem (First PC):**

$$\phi_1 = \underset{\|\phi\|=1}{\text{argmax}} \left\{\frac{1}{n}\sum_{i=1}^n \left(\sum_{j=1}^p \phi_j x_{ij}\right)^2\right\} = \underset{\|\phi\|=1}{\text{argmax}} \left\{\frac{1}{n}\|\mathbf{X}\phi\|^2\right\}$$

This simplifies to:

$$\phi_1 = \underset{\|\phi\|=1}{\text{argmax}} \{\phi^T \mathbf{X}^T\mathbf{X}\phi\}$$

**Solution via Eigendecomposition:**

The loading vectors $\phi_1, \phi_2, \ldots, \phi_p$ are the eigenvectors of the covariance matrix $\mathbf{X}^T\mathbf{X}/(n-1)$:

$$\mathbf{X}^T\mathbf{X}\phi_m = \lambda_m \phi_m$$

where $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p \geq 0$ are eigenvalues.

**Variance of $m$-th PC:**

$$\text{Var}(Z_m) = \frac{1}{n}\sum_{i=1}^n z_{im}^2 = \frac{\lambda_m}{n}$$

### SVD Implementation

PCA is typically computed via **Singular Value Decomposition** of the centered data matrix:

$$\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$

where:
- $\mathbf{U} \in \mathbb{R}^{n \times n}$: Left singular vectors (principal component scores)
- $\mathbf{\Sigma} \in \mathbb{R}^{n \times p}$: Diagonal matrix with singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $\mathbf{V} \in \mathbb{R}^{p \times p}$: Right singular vectors (principal component directions/loadings)

**Relationship to eigenvalues:**

$$\lambda_m = \sigma_m^2$$

**Computational Complexity:**
- Full SVD: $O(\min(np^2, n^2p))$
- First $M$ components: $O(npM)$ using randomized algorithms

### Proportion of Variance Explained (PVE)

**Total Variance:**

$$\text{TSS} = \sum_{j=1}^p \text{Var}(X_j) = \frac{1}{n}\sum_{j=1}^p \sum_{i=1}^n x_{ij}^2 = \frac{1}{n}\text{tr}(\mathbf{X}^T\mathbf{X})$$

**PVE by $m$-th PC:**

$$\text{PVE}_m = \frac{\lambda_m}{\sum_{j=1}^p \lambda_j} = \frac{\sigma_m^2}{\sum_{j=1}^p \sigma_j^2}$$

**Cumulative PVE:**

$$\text{CPVE}_M = \sum_{m=1}^M \text{PVE}_m = \frac{\sum_{m=1}^M \lambda_m}{\sum_{j=1}^p \lambda_j}$$

**Key Property (Variance-Error Decomposition):**

$$\underbrace{\sum_{j=1}^p \frac{1}{n}\sum_{i=1}^n x_{ij}^2}_{\text{Total Variance}} = \underbrace{\sum_{m=1}^M \frac{1}{n}\sum_{i=1}^n z_{im}^2}_{\text{Explained Variance}} + \underbrace{\sum_{j=1}^p \frac{1}{n}\sum_{i=1}^n \left(x_{ij} - \sum_{m=1}^M z_{im}\phi_{jm}\right)^2}_{\text{Reconstruction Error}}$$

This shows maximizing variance ≡ minimizing reconstruction MSE.

### Python Implementation

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load high-dimensional data
digits = load_digits()
X = digits.data  # 1797 samples × 64 features
y = digits.target

# Preprocessing: Center and scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA
pca = PCA()
pca.fit(X_scaled)

# Extract components
loadings = pca.components_  # Shape: (n_components, n_features)
explained_var = pca.explained_variance_
explained_var_ratio = pca.explained_variance_ratio_

# Cumulative variance
cumsum_var = np.cumsum(explained_var_ratio)

print(f"Original dimensions: {X.shape[1]}")
print(f"Components for 95% variance: {np.argmax(cumsum_var >= 0.95) + 1}")
print(f"Components for 99% variance: {np.argmax(cumsum_var >= 0.99) + 1}")
```

### Choosing Number of Components

**Method 1: Variance Threshold**

```python
# Find M for desired variance
target_variance = 0.95
n_components = np.argmax(cumsum_var >= target_variance) + 1

# Transform data
pca_reduced = PCA(n_components=n_components)
X_reduced = pca_reduced.fit_transform(X_scaled)
print(f"Reduced from {X.shape[1]} to {X_reduced.shape[1]} dimensions")
```

**Method 2: Scree Plot (Elbow Method)**

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Individual variance
ax1.bar(range(1, 21), explained_var_ratio[:20])
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Variance Explained')
ax1.set_title('Scree Plot')
ax1.axhline(y=1/len(explained_var_ratio), color='r', linestyle='--', 
            label='Average variance')

# Cumulative variance
ax2.plot(range(1, len(cumsum_var)+1), cumsum_var, marker='o')
ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
ax2.set_xlabel('Number of Components')
ax2.set_ylabel('Cumulative Variance Explained')
ax2.set_title('Cumulative Variance')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Method 3: Cross-Validation (Supervised Context)**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Test different numbers of components
n_components_range = range(5, 65, 5)
cv_scores = []

for n in n_components_range:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n)),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_n = n_components_range[np.argmax(cv_scores)]
print(f"Optimal components via CV: {optimal_n}")
print(f"CV Accuracy: {max(cv_scores):.3f}")
```

### Projection and Reconstruction

```python
# Project to lower dimensions
pca = PCA(n_components=20)
X_reduced = pca.fit_transform(X_scaled)
print(f"Reduced shape: {X_reduced.shape}")

# Reconstruct (inverse transform)
X_reconstructed = pca.inverse_transform(X_reduced)

# Calculate reconstruction error
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"Reconstruction MSE: {reconstruction_error:.4f}")

# Visualize original vs reconstructed
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    # Original
    if i < 5:
        ax.imshow(X[i].reshape(8, 8), cmap='gray')
        ax.set_title(f'Original {i}')
    # Reconstructed
    else:
        idx = i - 5
        reconstructed = scaler.inverse_transform(
            X_reconstructed[idx].reshape(1, -1)
        )
        ax.imshow(reconstructed.reshape(8, 8), cmap='gray')
        ax.set_title(f'Reconstructed {idx}')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

### Visualizing Principal Components

```python
# 2D visualization
X_2d = PCA(n_components=2).fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', 
                     alpha=0.6, edgecolors='k', linewidth=0.5)
plt.xlabel(f'PC1 ({explained_var_ratio[0]:.1%} variance)')
plt.ylabel(f'PC2 ({explained_var_ratio[1]:.1%} variance)')
plt.title('First Two Principal Components')
plt.colorbar(scatter, label='Digit Class')
plt.grid(True, alpha=0.3)
plt.show()

# Loadings interpretation (first PC)
first_pc_loadings = loadings[0].reshape(8, 8)
plt.figure(figsize=(6, 6))
plt.imshow(first_pc_loadings, cmap='RdBu_r')
plt.colorbar(label='Loading value')
plt.title('First Principal Component Loadings')
plt.show()
```

## IV. PCA Variants

### A. Randomized PCA

Efficient approximation for large datasets when $M \ll \min(n, p)$.

**Algorithm:** Uses randomized SVD based on random projections and power iterations.

**Complexity:** $O(M(n + p)M) + O(M^3)$ vs. $O(\min(n^2p, np^2))$ for full PCA.

```python
from sklearn.decomposition import PCA
import time

# Compare timing
n_components = 20

# Randomized PCA
start = time.time()
pca_random = PCA(n_components=n_components, svd_solver='randomized', 
                 random_state=42)
X_random = pca_random.fit_transform(X_scaled)
time_random = time.time() - start

# Full PCA
start = time.time()
pca_full = PCA(n_components=n_components, svd_solver='full')
X_full = pca_full.fit_transform(X_scaled)
time_full = time.time() - start

print(f"Randomized PCA: {time_random:.4f}s")
print(f"Full PCA: {time_full:.4f}s")
print(f"Speedup: {time_full/time_random:.2f}x")
print(f"Approximation error: {np.mean((X_random - X_full)**2):.2e}")
```

### B. Incremental PCA

Processes data in mini-batches for memory efficiency.

```python
from sklearn.decomposition import IncrementalPCA

# Incremental PCA for large datasets
n_batches = 10
inc_pca = IncrementalPCA(n_components=20)

for batch in np.array_split(X_scaled, n_batches):
    inc_pca.partial_fit(batch)

X_inc = inc_pca.transform(X_scaled)
print(f"Incremental PCA shape: {X_inc.shape}")
print(f"Variance explained: {inc_pca.explained_variance_ratio_.sum():.3f}")
```

### C. Kernel PCA

Performs PCA in high-dimensional feature space via kernel trick, enabling nonlinear dimensionality reduction.

**Kernel Functions:**

$$k(\mathbf{x}_i, \mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle$$

Common kernels:
- **Linear:** $k(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T\mathbf{x}_j$
- **Polynomial:** $k(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T\mathbf{x}_j + r)^d$
- **RBF (Gaussian):** $k(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_swiss_roll

# Generate nonlinear data
X_swiss, color = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)

# Linear PCA (fails to unroll)
pca_linear = PCA(n_components=2)
X_pca = pca_linear.fit_transform(X_swiss)

# Kernel PCA with RBF kernel
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1, random_state=42)
X_kpca = kpca.fit_transform(X_swiss)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=color, cmap='viridis')
ax1.set_title('Linear PCA')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')

ax2.scatter(X_kpca[:, 0], X_kpca[:, 1], c=color, cmap='viridis')
ax2.set_title('Kernel PCA (RBF)')
ax2.set_xlabel('Component 1')
ax2.set_ylabel('Component 2')

plt.tight_layout()
plt.show()
```

## V. Random Projection

Leverages the **Johnson-Lindenstrauss Lemma** for fast, approximate dimensionality reduction.

### Theoretical Foundation

**Johnson-Lindenstrauss Lemma:** For any $0 < \epsilon < 1$ and integer $m$, let $M$ be a set of $m$ points in $\mathbb{R}^n$. Then there exists a linear map $f: \mathbb{R}^n \to \mathbb{R}^d$ where:

$$d \geq \frac{4\log(m)}{\epsilon^2/2 - \epsilon^3/3}$$

such that for all $\mathbf{u}, \mathbf{v} \in M$:

$$(1-\epsilon)\|\mathbf{u} - \mathbf{v}\|^2 \leq \|f(\mathbf{u}) - f(\mathbf{v})\|^2 \leq (1+\epsilon)\|\mathbf{u} - \mathbf{v}\|^2$$

**Key Insight:** Required dimension $d$ depends only on number of samples $m$ and tolerance $\epsilon$, **not** original dimension $n$.

### Implementation

```python
from sklearn.random_projection import GaussianRandomProjection, johnson_lindenstrauss_min_dim

# Calculate minimum dimensions
n_samples = X.shape[0]
eps = 0.1
d_min = johnson_lindenstrauss_min_dim(n_samples, eps=eps)
print(f"Minimum dimensions for ε={eps}: {d_min}")

# Gaussian Random Projection
grp = GaussianRandomProjection(n_components=d_min, random_state=42)
X_projected = grp.fit_transform(X_scaled)

print(f"Original shape: {X_scaled.shape}")
print(f"Projected shape: {X_projected.shape}")

# Verify distance preservation
from scipy.spatial.distance import pdist

original_dists = pdist(X_scaled[:100])
projected_dists = pdist(X_projected[:100])

distortion = np.abs(original_dists - projected_dists) / original_dists
print(f"Mean relative distortion: {distortion.mean():.4f}")
print(f"Max relative distortion: {distortion.max():.4f}")
```

## VI. Manifold Learning

### A. Locally Linear Embedding (LLE)

Preserves local neighborhood structure in lower dimensions.

**Algorithm:**

1. **Find neighbors:** For each $\mathbf{x}^{(i)}$, identify $k$ nearest neighbors
2. **Compute weights:** Minimize reconstruction error:
   $$\min_{\mathbf{W}} \sum_{i=1}^n \left\|\mathbf{x}^{(i)} - \sum_{j=1}^n w_{ij}\mathbf{x}^{(j)}\right\|^2$$
   subject to: $w_{ij} = 0$ if $\mathbf{x}^{(j)} \notin \text{neighbors}(\mathbf{x}^{(i)})$ and $\sum_j w_{ij} = 1$
3. **Map to low dimension:** Find $\mathbf{z}^{(i)} \in \mathbb{R}^d$ minimizing:
   $$\min_{\mathbf{Z}} \sum_{i=1}^n \left\|\mathbf{z}^{(i)} - \sum_{j=1}^n w_{ij}\mathbf{z}^{(j)}\right\|^2$$

```python
from sklearn.manifold import LocallyLinearEmbedding

# LLE on Swiss roll
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_lle = lle.fit_transform(X_swiss)

plt.figure(figsize=(8, 6))
plt.scatter(X_lle[:, 0], X_lle[:, 1], c=color, cmap='viridis')
plt.title('LLE on Swiss Roll')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Position along r—oll')
plt.show()

print(f"Reconstruction error: {lle.reconstruction_error_:.4f}")
```

### B. t-SNE

Preserves local structure for visualization by minimizing divergence between high- and low-dimensional probability distributions.

**Algorithm:**

1. **High-dimensional similarities:** Compute pairwise conditional probabilities:
   $$p_{j|i} = \frac{\exp(-\|\mathbf{x}^{(i)} - \mathbf{x}^{(j)}\|^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-\|\mathbf{x}^{(i)} - \mathbf{x}^{(k)}\|^2 / 2\sigma_i^2)}$$
   $$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

2. **Low-dimensional similarities:** Use t-distribution (heavy tails):
   $$q_{ij} = \frac{(1 + \|\mathbf{z}^{(i)} - \mathbf{z}^{(j)}\|^2)^{-1}}{\sum_{k \neq l}(1 + \|\mathbf{z}^{(k)} - \mathbf{z}^{(l)}\|^2)^{-1}}$$

3. **Minimize KL divergence:**
   $$\text{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

```python
from sklearn.manifold import TSNE

# t-SNE on digits
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', 
                     alpha=0.7, edgecolors='k', linewidth=0.5)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization of Digits Dataset')
plt.colorbar(scatter, label='Digit Class')
plt.show()

print(f"Final KL divergence: {tsne.kl_divergence_:.4f}")
```

**Hyperparameters:**
- `perplexity` (5-50): Balances local vs. global structure
- `learning_rate` (10-1000): Step size for gradient descent
- `n_iter` (≥250): Number of optimization iterations

**Warning:** t-SNE is **non-deterministic** and primarily for visualization, not preprocessing for ML pipelines.

### C. UMAP

Faster alternative to t-SNE with better preservation of global structure.

```python
from umap import UMAP

# UMAP on digits
umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = umap.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10',
                     alpha=0.7, edgecolors='k', linewidth=0.5)
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.title('UMAP Visualization of Digits Dataset')
plt.colorbar(scatter, label='Digit Class')
plt.show()
```

### D. Isomap

Preserves geodesic distances (distances along manifold surface).

**Algorithm:**

1. Construct $k$-NN graph
2. Compute shortest path distances (geodesics) using Dijkstra/Floyd-Warshall
3. Apply MDS to geodesic distance matrix

```python
from sklearn.manifold import Isomap

# Isomap on Swiss roll
isomap = Isomap(n_components=2, n_neighbors=10)
X_isomap = isomap.fit_transform(X_swiss)

plt.figure(figsize=(8, 6))
plt.scatter(X_isomap[:, 0], X_isomap[:, 1], c=color, cmap='viridis')
plt.title('Isomap on Swiss Roll')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Position along roll')
plt.show()

print(f"Reconstruction error: {isomap.reconstruction_error():.4f}")
```

## VII. Supervised Dimensionality Reduction

### A. Principal Components Regression (PCR)

Two-stage process: unsupervised PCA + supervised regression.

```python
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Generate regression data
X_reg, y_reg = make_regression(n_samples=500, n_features=100, 
                                n_informative=10, noise=5, random_state=42)

# PCR Pipeline
pcr = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=15)),
    ('regression', LinearRegression())
])

# Cross-validation
scores = cross_val_score(pcr, X_reg, y_reg, cv=5, 
                        scoring='neg_mean_squared_error')
rmse = np.sqrt(-scores.mean())
print(f"PCR RMSE: {rmse:.3f}")

# Compare with different numbers of components
n_components_range = range(1, 51, 2)
cv_scores = []

for n in n_components_range:
    pcr.set_params(pca__n_components=n)
    scores = cross_val_score(pcr, X_reg, y_reg, cv=5,
                            scoring='neg_mean_squared_error')
    cv_scores.append(np.sqrt(-scores.mean()))

plt.figure(figsize=(10, 6))
plt.plot(n_components_range, cv_scores, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('RMSE (Cross-Validation)')
plt.title('PCR Performance vs. Number of Components')
plt.grid(True, alpha=0.3)
plt.show()

optimal_n = n_components_range[np.argmin(cv_scores)]
print(f"Optimal components: {optimal_n}")
```

**Limitation:** PCs are chosen to maximize variance, not correlation with response. May discard relevant but low-variance directions.

### B. Partial Least Squares (PLS)

Finds directions maximizing **covariance** with response.

**Algorithm:** For $m = 1, \ldots, M$:

1. Compute direction $\phi_m$ proportional to: $\mathbf{X}^T\mathbf{y}$
2. Compute scores: $z_m = \mathbf{X}\phi_m$
3. Regress $\mathbf{X}$ and $\mathbf{y}$ on $z_m$: $\mathbf{X} \leftarrow \mathbf{X} - z_m\phi_m^T$, $\mathbf{y} \leftarrow \mathbf{y} - z_m\theta_m$

**Final prediction:** $\hat{y} = \sum_{m=1}^M \theta_m z_m$


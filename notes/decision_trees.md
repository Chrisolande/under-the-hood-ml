# Decision Trees

Decision Trees (DTs) are foundational supervised learning algorithms capable of both classification and regression. They partition the feature space into non-overlapping regions through recursive binary splits, creating interpretable rule-based models that form the basis for powerful ensemble methods.

## I. Core Mathematical Framework

### The CART Algorithm

Scikit-learn implements the **Classification and Regression Tree (CART)** algorithm, which constructs binary trees using a **greedy, top-down recursive partitioning** strategy.

**Key insight:** The algorithm is "greedy" because it optimizes locally at each split without considering future splits, this makes it computationally tractable but doesn't guarantee a globally optimal tree.

At each node, CART searches for the optimal split by evaluating all possible combinations of:

- Feature $X_j$ where $j \in \{1, 2, ..., p\}$
- Threshold $t_k$

This creates two child regions:

- $R_1(j, t_k) = \{X | X_j < t_k\}$
- $R_2(j, t_k) = \{X | X_j \geq t_k\}$

### Cost Functions

#### Regression Trees: Mean Squared Error

For continuous targets, each region $R_m$ predicts the mean of its training observations:

$$\hat{y}_{R_m} = \frac{1}{|R_m|} \sum_{i \in R_m} y_i$$

The CART cost function minimizes weighted MSE across child nodes:

$$J(j, t_k) = \frac{n_{\text{left}}}{n} \text{MSE}_{\text{left}} + \frac{n_{\text{right}}}{n} \text{MSE}_{\text{right}}$$

where:
$$\text{MSE}_{\text{node}} = \frac{1}{n_{\text{node}}} \sum_{i \in \text{node}} (y_i - \hat{y}_{\text{node}})^2$$

**Interpretation:** We weight each child's error by its proportion of samples, ensuring splits that create highly unbalanced nodes aren't artificially favored.

#### Classification Trees: Impurity Measures

For categorical targets, we maximize node **purity** (homogeneity of class labels).

**1. Gini Impurity (Default in scikit-learn)**

$$G_i = 1 - \sum_{k=1}^{K} p_{i,k}^2$$

where $p_{i,k}$ is the proportion of class $k$ samples in node $i$.

- **Range:** $[0, 1 - 1/K]$
- **Pure node:** $G_i = 0$ (all samples same class)
- **Maximum impurity:** $G_i = 1 - 1/K$ (equal class distribution)
- **Computational advantage:** Faster than entropy (no logarithms)

**2. Entropy (Information Theory)**

$$H_i = -\sum_{k=1}^{K} p_{i,k} \log_2(p_{i,k})$$

Convention: $0 \log_2(0) = 0$

- **Interpretation:** Expected number of bits needed to encode class information
- **Pure node:** $H_i = 0$
- **Maximum entropy:** $H_i = \log_2(K)$ bits

**3. Information Gain**

Used in algorithms like ID3/C4.5:

$$\text{IG}(j, t_k) = H_{\text{parent}} - \left(\frac{n_{\text{left}}}{n} H_{\text{left}} + \frac{n_{\text{right}}}{n} H_{\text{right}}\right)$$

This measures the reduction in entropy from a split, higher values indicate better splits.

**Practical Note:** Gini vs. Entropy usually yields similar trees. Gini tends to isolate the most frequent class in pure nodes, while entropy produces slightly more balanced trees. Choose based on domain requirements or cross-validation performance.

## II. Tree Construction Algorithm

### Step 1: Recursive Binary Splitting

**Pseudocode:**

```
function BuildTree(node, depth):
    if stopping_criterion_met(node, depth):
        return leaf_node with prediction
    
    best_cost = infinity
    for each feature j in features:
        for each threshold t in unique_values(feature_j):
            split data into left and right
            cost = weighted_impurity(left, right)
            if cost < best_cost:
                best_cost = cost
                best_split = (j, t)
    
    split node using best_split
    left_child = BuildTree(left_samples, depth+1)
    right_child = BuildTree(right_samples, depth+1)
    return node
```

**Computational Complexity:**

- For each split: $O(n \cdot p \cdot n \log n)$ where sorting dominates
- Total tree construction: $O(n^2 \cdot p \cdot \log n)$ in worst case

### Step 2: Regularization via Stopping Criteria

Prevent overfitting by constraining tree growth:

| Hyperparameter | Effect | Typical Values |
|----------------|--------|----------------|
| `max_depth` | Limits tree depth | 3-10 for interpretability, 10-20 for ensembles |
| `min_samples_split` | Minimum samples to split a node | 2-20 |
| `min_samples_leaf` | Minimum samples in leaf nodes | 1-10 |
| `max_leaf_nodes` | Maximum number of terminal nodes | Depends on problem |
| `min_impurity_decrease` | Minimum impurity reduction for split | 0.0-0.01 |

**Pro tip:** Start with shallow trees (`max_depth=3-5`) for interpretability, then increase depth while monitoring validation performance.

### Step 3: Cost Complexity Pruning

**Problem:** Pre-pruning (stopping early) may miss beneficial future splits. Post-pruning addresses this.

**Algorithm:** Grow a full tree $T_0$, then prune back by minimizing:

$$C_{\alpha}(T) = \sum_{m=1}^{|T|} \sum_{i \in R_m} (y_i - \hat{y}_{R_m})^2 + \alpha |T|$$

where:

- $|T|$ = number of terminal nodes (tree complexity)
- $\alpha$ = complexity parameter (tuning hyperparameter)

**Process:**

1. For increasing $\alpha$, compute optimal subtree $T_{\alpha}$
2. Use K-fold CV to select $\alpha^*$ with best validation performance
3. Return $T_{\alpha^*}$ trained on full data

**When to use:** Post-pruning (`ccp_alpha` in scikit-learn) often yields better trees than pre-pruning alone, especially for noisy data.

## III. Python Implementation Guide

### Basic Classification Tree

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

# Load data
iris = load_iris(as_frame=True)
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train classifier
clf = DecisionTreeClassifier(
    criterion='gini',        # or 'entropy'
    max_depth=3,            # regularization
    min_samples_split=5,    # regularization
    random_state=42
)
clf.fit(X_train, y_train)

# Evaluate
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
cv_scores = cross_val_score(clf, X_train, y_train, cv=5)

print(f"Training accuracy: {train_acc:.3f}")
print(f"Test accuracy: {test_acc:.3f}")
print(f"CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Interpret predictions
sample = np.array([[5.0, 1.5]])
prediction = clf.predict(sample)
probabilities = clf.predict_proba(sample)
print(f"\nPrediction: Class {prediction[0]}")
print(f"Class probabilities: {probabilities[0].round(3)}")

# Extract decision path
leaf_id = clf.apply(sample)
print(f"Sample ended in leaf node: {leaf_id[0]}")
```

### Visualizing the Tree

```python
# Text representation
feature_names = ["petal length (cm)", "petal width (cm)"]
tree_rules = export_text(clf, feature_names=feature_names)
print("\nDecision Rules:\n", tree_rules)

# Graphical plot
plt.figure(figsize=(12, 8))
plot_tree(clf, 
          feature_names=feature_names,
          class_names=iris.target_names.tolist(),
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree Visualization")
plt.show()
```

### Regression Tree Example

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate synthetic data
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Train regressor
reg = DecisionTreeRegressor(max_depth=3, random_state=42)
reg.fit(X, y)

# Predict
X_test = np.linspace(0, 5, 100).reshape(-1, 1)
y_pred = reg.predict(X_test)

# Evaluate
print(f"R² score: {r2_score(y, reg.predict(X)):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, reg.predict(X))):.3f}")
```

### Cost Complexity Pruning

```python
# Get pruning path
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # exclude max alpha

# Train trees with different alpha values
clfs = []
for ccp_alpha in ccp_alphas:
    clf_pruned = DecisionTreeClassifier(
        random_state=42, ccp_alpha=ccp_alpha
    )
    clf_pruned.fit(X_train, y_train)
    clfs.append(clf_pruned)

# Find best alpha via validation
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

# Plot
fig, ax = plt.subplots()
ax.plot(ccp_alphas, train_scores, label="train", marker='o')
ax.plot(ccp_alphas, test_scores, label="test", marker='o')
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.legend()
plt.show()
```

## IV. Feature Importance

Decision trees provide natural feature importance scores:

$$\text{Importance}(X_j) = \sum_{t: \text{split on } X_j} \frac{n_t}{n} \Delta I(t)$$

where:

- $n_t$ = number of samples at node $t$
- $\Delta I(t)$ = impurity decrease from the split

```python
# Extract feature importance
importances = clf.feature_importances_
for name, imp in zip(feature_names, importances):
    print(f"{name}: {imp:.3f}")
```

```python
from sklearn.inspection import permutation_importance

# Compute permutation importance on test data
result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)

# Display results
for name, imp, std in zip(feature_names, result.importances_mean, result.importances_std):
    print(f"{name}: {imp:.3f} ± {std:.3f}")
```

**Warning:** Feature importance can be biased toward high-cardinality features (those with many unique values). Consider permutation importance for unbiased estimates.

## V. Advantages and Limitations

### Strengths

1. **Interpretability:** "White-box" models with clear if-then rules
2. **No feature scaling required:** Decisions based on thresholds, not distances
3. **Handles mixed data types:** Categorical and numerical features (with proper encoding)
4. **Non-linear relationships:** Captures complex interactions automatically
5. **Robust to outliers:** Splits based on order, not magnitude
6. **Handles missing data:** Can use surrogate splits (not in scikit-learn by default)
7. **Feature selection:** Implicitly selects relevant features

### Weaknesses

1. **High variance:** Small data changes → completely different trees
2. **Overfitting tendency:** Without regularization, memorizes training data
3. **Orthogonal boundaries:** Struggles with diagonal decision boundaries
   - Example: XOR problem requires deep tree, while simple rotated feature space needs shallow tree
4. **Instability:** Non-robust to training set perturbations
5. **Prediction smoothness:** Piecewise constant predictions (step functions)
6. **Biased splits:** May favor high-cardinality features
7. **Lower accuracy:** Often outperformed by ensembles and other methods

### When to Use Decision Trees

**Best for:**

- Exploratory analysis and feature importance
- Problems requiring model interpretability (healthcare, finance, legal)
- Quick baseline models
- Building blocks for ensembles

**Avoid when:**

- Need highest predictive accuracy (use ensembles instead)
- Feature space has many irrelevant features
- Require smooth prediction functions
- Data is very noisy

## VI. Connection to Ensemble Methods

Single trees are **high-variance, low-bias** learners ideal for ensembling.

### Random Forests (Bagging + Feature Randomness)

**Algorithm:**

1. Generate $B$ bootstrap samples from training data
2. For each sample, grow a deep tree (low bias, high variance)
3. **Key difference from bagging:** At each split, consider only $m = \sqrt{p}$ random features (decorrelates trees)
4. Aggregate predictions: average (regression) or majority vote (classification)

**Why it works:**

- Variance of average of $B$ independent variables: $\sigma^2 / B$
- With correlation $\rho$: $\rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$
- Feature randomness reduces $\rho$, dramatically lowering variance

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,           # number of trees
    max_features='sqrt',        # m = sqrt(p)
    max_depth=None,             # grow deep trees
    min_samples_split=2,
    bootstrap=True,             # bagging
    oob_score=True,            # out-of-bag evaluation
    n_jobs=-1,                 # parallelize
    random_state=42
)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_:.3f}")
```

### Gradient Boosting (Sequential Error Correction)

**Algorithm (Regression):**

1. Initialize: $F_0(x) = \bar{y}$
2. For $m = 1$ to $M$:
   - Compute residuals: $r_i = y_i - F_{m-1}(x_i)$
   - Fit tree $h_m$ to residuals
   - Update: $F_m(x) = F_{m-1}(x) + \lambda h_m(x)$
3. Return $F_M(x)$

**Key parameters:**

- $\lambda$ (learning rate): Controls step size (0.01-0.1)
- $M$ (n_estimators): Number of boosting rounds
- $d$ (max_depth): Tree complexity (1-8, often use stumps $d=1$)

**Why it works:** Gradient descent in function space, each tree corrects predecessor's errors.

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,         # shrinkage parameter
    max_depth=3,               # shallow trees
    subsample=0.8,             # stochastic GB
    random_state=42
)
gb.fit(X_train, y_train)
```

**Modern implementations:** XGBoost, LightGBM, CatBoost add:

- Regularization terms in objective
- Efficient handling of sparse data
- Built-in cross-validation
- Distributed computing support

## VII. Practical Recommendations

### Hyperparameter Tuning Strategy

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 9, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
```

### Diagnostic Checklist

1. **Overfitting check:** Compare train vs. validation performance
2. **Tree complexity:** Visualize tree if too deep, increase regularization
3. **Class imbalance:** Use `class_weight='balanced'` or resampling
4. **Feature scaling:** Not needed, but standardize if using with other models in pipeline
5. **Missing data:** Impute or use algorithms supporting missingness (XGBoost, LightGBM)

### Summary

Decision trees trade predictive power for interpretability. For production ML:

- **Single trees:** Use when interpretability is paramount
- **Random Forests:** Use when want balance of accuracy and some interpretability (feature importance)
- **Gradient Boosting:** Use when need maximum accuracy and can sacrifice interpretability

The key insight: Trees partition feature space using axis-aligned cuts. This is both their strength (interpretability) and weakness (inefficiency for diagonal boundaries). Ensembles compensate by averaging many diverse partitions, achieving state-of-the-art performance while maintaining reasonable computational costs.

# Ensemble Learning

---
A group of predictors is called an **ensemble**. The technique of making predictions using ensembles is called **ensemble learning**.
**Example**
We can train a group of decision tree classifiers, each on a different random subset of the training set. We then obtain the predictions of all the individual trees, the class with th most votes is the ensemble's prediction. Now, this ensemble of decision trees is what we call **RandomForest**

## I. Theoretical Foundation: The Bias-Variance Trade-Off

### 1.1 Mathematical Decomposition

Ensemble methods fundamentally aim to minimize
the expected test error. For a squared-error loss function, the expected test Mean Squared Error (MSE) for an estimate $\hat{f}(\mathbf{x})$ decomposes as:

$$\mathbb{E}[(\hat{f}(\mathbf{x}) - y)^2] = \underbrace{\text{Var}(\hat{f}(\mathbf{x}))}_{\text{Variance}} + \underbrace{[\text{Bias}(\hat{f}(\mathbf{x}))]^2}_{\text{Squared Bias}} + \underbrace{\sigma^2}_{\text{Irreducible Error}}$$

where $\mathbb{E}[\cdot]$ denotes expectation over training sets.

**Component Analysis:**

- **Variance** $\text{Var}(\hat{f}(\mathbf{x}))$: Quantifies how much $\hat{f}$ changes across different training datasets. High-capacity models (deep decision trees, high-degree polynomials) exhibit high variance-they fit training data closely but predictions fluctuate dramatically with different samples.

- **Squared Bias** $[\text{Bias}(\hat{f}(\mathbf{x}))]^2$: Measures systematic error from approximating complex reality with simplified models. Low-capacity models (shallow trees, linear regression) have high bias-they systematically underfit.

- **Irreducible Error** $\sigma^2 = \text{Var}(\epsilon)$: Inherent noise in the data-generating process, irreducible by any model.

### 1.2 Ensemble Strategy

Ensemble methods exploit this decomposition through two primary strategies:

1. **Variance Reduction** (Bagging, Random Forests): Average predictions from multiple high-variance models to reduce overall variance without increasing bias.

2. **Bias Reduction** (Boosting): Sequentially fit models to residuals, progressively reducing bias while carefully controlling variance through regularization.

**Statistical Insight:**  
Consider $B$ independent predictors $\hat{f}_1, \ldots, \hat{f}_B$, each with variance $\sigma^2$. The variance of their average is:

$$\text{Var}\left(\frac{1}{B}\sum_{b=1}^{B}\hat{f}_b\right) = \frac{\sigma^2}{B}$$

For correlated predictors with pairwise correlation $\rho$:

$$\text{Var}\left(\frac{1}{B}\sum_{b=1}^{B}\hat{f}_b\right) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

As $B \to \infty$, variance approaches $\rho\sigma^2$. **Key takeaway**: Decorrelating predictors (low $\rho$) is crucial for variance reduction, this motivates Random Forests' feature randomization.

---

## II. Voting Classifiers

### 2.1 Theoretical Framework

**Voting Classifiers** aggregate predictions from diverse base classifiers. Maximum effectiveness occurs when base learners make **uncorrelated errors**.

**Diversity Principle:**  
If classifiers have error rate $\epsilon < 0.5$ and make independent errors, ensemble error decreases exponentially with ensemble size (Condorcet's Jury Theorem).

### 2.2 Hard Voting (Majority Vote)

Prediction is the mode of individual predictions:

$$\hat{y}(\mathbf{x}) = \underset{k \in \mathcal{Y}}{\operatorname{argmax}} \sum_{j=1}^{N} \mathbb{1}(h_j(\mathbf{x}) = k)$$

where:

- $N$ = number of classifiers
- $h_j(\mathbf{x})$ = prediction of $j$-th classifier
- $\mathbb{1}(\cdot)$ = indicator function
- $\mathcal{Y}$ = set of class labels

**Example**
Classifier 1 -> A
Classifier 2 -> B
classifier 3 -> A

mode -> A
Hard Voting result -> A
**Python Implementation:**

```python
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

# Generate synthetic dataset
X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define diverse base classifiers
log_clf = LogisticRegression(random_state=42, max_iter=1000)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(kernel='rbf', probability=True, random_state=42)

# Hard Voting Classifier
voting_hard = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rf_clf), ('svm', svm_clf)],
    voting='hard'
)

# Evaluate individual classifiers and ensemble
for clf, name in zip([log_clf, rf_clf, svm_clf, voting_hard], 
                      ['Logistic Reg', 'Random Forest', 'SVM', 'Hard Voting']):
    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name:15s}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

**Expected Output Pattern:**  
The voting classifier typically outperforms individual classifiers, especially when base learners have comparable but distinct error patterns.

### 2.3 Soft Voting (Probability Averaging)

When classifiers provide class probabilities $p_j(k|\mathbf{x})$, the ensemble predicts:

$$\hat{y}(\mathbf{x}) = \underset{k \in \mathcal{Y}}{\operatorname{argmax}} \frac{1}{N}\sum_{j=1}^{N} p_j(k|\mathbf{x})$$

**Advantage:** Weights confident predictions more heavily. A classifier 99% certain contributes more than one that's 51% certain.

**Example**
Classifier 1 -> [A: 0.6, B: 0.4]
Classifier 2 -> [A: 0.4, B: 0.6]
Classifier 3 -> [A: 0.7, B: 0.3]
Average -> [A: 0.5666, B: 0.433]

Soft Voting Result -> A

**Python Implementation:**

```python
# Soft Voting Classifier (requires probability estimates)
voting_soft = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rf_clf), ('svm', svm_clf)],
    voting='soft'
)

voting_soft.fit(X_train, y_train)
soft_scores = cross_val_score(voting_soft, X_train, y_train, cv=5, scoring='accuracy')
print(f"Soft Voting: {soft_scores.mean():.4f} (+/- {soft_scores.std():.4f})")

# Compare probabilities for a single instance
sample_idx = 0
X_sample = X_test[sample_idx:sample_idx+1]

print("\nPrediction probabilities for sample instance:")
for name, clf in [('Logistic Reg', log_clf), ('Random Forest', rf_clf), 
                   ('SVM', svm_clf), ('Soft Voting', voting_soft)]:
    probs = clf.predict_proba(X_sample)[0]
    print(f"{name:15s}: Class 0: {probs[0]:.3f}, Class 1: {probs[1]:.3f}")
```

**Key Insight:** Soft voting typically outperforms hard voting because it leverages prediction confidence, not just class labels.

---

## III. Bagging and Pasting

### 3.1 Sampling Strategies

**Bagging** (Bootstrap AGGregatING) and **Pasting** train identical model architectures on different random subsets of training data.

**Definitions:**

- **Bagging**: Sampling **with replacement** (bootstrap sampling)
  - Each subset contains ~63.2% unique instances ($1 - e^{-1} \approx 0.632$)
  - Introduces maximum diversity through bootstrap variance

- **Pasting**: Sampling **without replacement**
  - Each subset is a random subsample
  - Lower diversity than bagging

**Bootstrap Sampling Mathematics:**

It basically is SRS but with replacement, now, if m tends to infinity, the SRSWR becomes $e^{-1}$

Probability an instance is **not** selected in one draw: $(1 - \frac{1}{m})$  
Probability **not** selected in $m$ draws: $(1 - \frac{1}{m})^m \xrightarrow{m \to \infty} e^{-1} \approx 0.368$

Thus ~36.8% instances remain **Out-of-Bag (OOB)** per bootstrap sample.

### 3.2 Aggregation Functions

**For Classification:**
$$\hat{y}(\mathbf{x}) = \text{mode}\{h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_B(\mathbf{x})\}$$

**For Regression:**
$$\hat{f}_{\text{bag}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}^{*b}(\mathbf{x})$$

where $\hat{f}^{*b}$ is trained on the $b$-th bootstrap sample.

### 3.3 Variance Reduction Mechanism

Bagging is most effective for **high-variance, low-bias** base learners (e.g., deep unpruned trees) or simply, when the model overfits.

**Why it works:**

1. Bootstrap sampling creates diverse training sets
2. Individual trees have high variance but (approximately) unbiased
3. Averaging reduces variance: $\text{Var}(\bar{X}) = \frac{\sigma^2}{n}$ for i.i.d. variables
4. Bias remains approximately constant: $\text{Bias}(\bar{X}) = \text{Bias}(X)$

### 3.4 Python Implementation: BaggingClassifier

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Single deep decision tree (high variance baseline)
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
tree_pred = tree_clf.predict(X_test)
print(f"Single Tree Accuracy: {accuracy_score(y_test, tree_pred):.4f}")

# Bagging with 500 trees
bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,      # Use 100 instances per tree
    bootstrap=True,       # Bagging (with replacement)
    n_jobs=-1,            # Parallel training
    random_state=42,
    oob_score=True        # Enable OOB evaluation
)

bag_clf.fit(X_train, y_train)
bag_pred = bag_clf.predict(X_test)
print(f"Bagging Accuracy (Test): {accuracy_score(y_test, bag_pred):.4f}")
print(f"Bagging OOB Score: {bag_clf.oob_score_:.4f}")

# Compare with Pasting (without replacement)
paste_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,
    bootstrap=False,      # Pasting (without replacement)
    n_jobs=-1,
    random_state=42
)

paste_clf.fit(X_train, y_train)
paste_pred = paste_clf.predict(X_test)
print(f"Pasting Accuracy: {accuracy_score(y_test, paste_pred):.4f}")
```

### 3.5 Out-of-Bag (OOB) Evaluation

**Concept:** For each training instance, evaluate using only the trees that didn't see it during training (the ~37% OOB instances for that tree).

**OOB Score Calculation:**

1. For instance $i$, identify all trees $\mathcal{T}_i$ for which $i$ was OOB
2. Aggregate predictions from $\mathcal{T}_i$ only
3. Compute accuracy/error across all instances

**Advantage:** Provides unbiased test error estimate without needing a validation set, essentially free cross-validation.

**Python Implementation:**

```python
# OOB predictions for individual instances
oob_decision = bag_clf.oob_decision_function_
print(f"\nOOB Decision Function shape: {oob_decision.shape}")
print(f"First 5 instances OOB probabilities:\n{oob_decision[:5]}")

# Visualize OOB vs Test accuracy as function of ensemble size
from sklearn.ensemble import BaggingClassifier
import numpy as np

oob_scores = []
test_scores = []
n_estimators_range = range(1, 201, 10)

for n_est in n_estimators_range:
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=n_est,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    bag.fit(X_train, y_train)
    oob_scores.append(bag.oob_score_)
    test_scores.append(bag.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, oob_scores, label='OOB Score', marker='o')
plt.plot(n_estimators_range, test_scores, label='Test Score', marker='s')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('OOB vs Test Accuracy: Convergence Analysis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
```

**Interpretation:** OOB score closely tracks test score, validating its use as a proxy for generalization performance.

### 3.6 Random Patches and Random Subspaces

These are advanced extensions of **Bagging** that deliberately introduce more randomness, aiming to make individual base models in the ensemble **less correlated**.
The purpose is to **reduce variance** by ensuring no single feature or subset of training instances dominates all the base learners.

---

#### **1. Random Subspaces - The Feature-Focused Method**

**Intuition:**
Imagine a dataset with 100 features, but only 5 are truly informative. In regular bagging, every tree may focus on those same 5 features.
**Random Subspaces** forces each tree to ignore most features, solving the problem using only a random subset.

**Mechanism:**

- Trains each base learner on **all rows** (instances).
- Each learner only sees a **random subset of the features (columns)**.

**Trade-off:**

- Slightly increases bias (since each model sees fewer features).
- Greatly reduces variance (models become less correlated).

**Primary Use:**
Best for **high-dimensional datasets** where many features are redundant or correlated, and training speed is a concern.

| **Scikit-Learn Parameter**  | **Interpretation**                              |
| --------------------------- | ----------------------------------------------- |
| `bootstrap=False`           | No instance sampling (use all rows).            |
| `max_samples=1.0`           | Use all training instances (100% of rows).      |
| `max_features < n_features` | Sample features (use only a subset of columns). |

---

#### **2. Random Patches - The Double Randomization Method**

**Intuition:**
This method goes further - forcing the model to solve the problem based only on a random *patch* (a sub-rectangle) of the dataset matrix, ensuring maximum diversity.

**Mechanism:**

- Trains each base learner on a **random subset of instances (rows)** *and* a **random subset of features (columns)**.

**Trade-off:**

- Produces the highest model diversity (variance reduction).
- Introduces higher bias, as each learner is trained on less data.
- The aggregation (ensemble averaging) compensates for the individual model weakness.

**Primary Use:**
Ideal for **very large datasets** or when both instance and feature correlations are high, and computational efficiency is critical.

| **Scikit-Learn Parameter**  | **Interpretation**                                  |
| --------------------------- | --------------------------------------------------- |
| `bootstrap=True`            | Sample instances (subset of rows with replacement). |
| `max_features < n_features` | Sample features (use a subset of columns).          |

**Python Implementation**:

```python
# Random Patches Method
random_patches = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,       # Sample instances
    max_features=0.5,      # Sample 50% of features
    bootstrap=True,
    bootstrap_features=True,  # Sample features with replacement
    random_state=42,
    n_jobs=-1
)

random_patches.fit(X_train, y_train)
print(f"Random Patches Accuracy: {random_patches.score(X_test, y_test):.4f}")

# Random Subspaces Method
random_subspaces = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=len(X_train),  # All instances
    max_features=0.5,           # Sample 50% of features
    bootstrap=False,
    bootstrap_features=True,
    random_state=42,
    n_jobs=-1
)

random_subspaces.fit(X_train, y_train)
print(f"Random Subspaces Accuracy: {random_subspaces.score(X_test, y_test):.4f}")
```

---

## IV. Random Forests

### 4.1 Architecture and Motivation

**Random Forest** = Bagging + Feature Randomization at each split

**Key Innovation:** When growing each tree, at each node, consider only a **random subset** of features of size $m$ for splitting.

**Why this matters:**

- Standard bagging: If one feature is very strong, most trees split on it early → highly correlated trees → limited variance reduction
- Random Forests: Feature randomization decorrelates trees → lower $\rho$ → greater variance reduction

### 4.2 Mathematical Formulation

**Variance with Correlation:**
$$\text{Var}(\bar{X}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

As $B \to \infty$:
$$\text{Var}(\bar{X}) \to \rho\sigma^2$$

**Goal:** Minimize $\rho$ (correlation between trees) through feature randomization.

**Hyperparameter $m$ (max_features):**

- **Classification**: Recommended $m = \sqrt{p}$
- **Regression**: Recommended $m = \frac{p}{3}$
- **Special cases**:
  - $m = p$: Equivalent to bagging (no randomization)
  - $m = 1$: Maximum randomization (each split sees one random feature)

### 4.3 Algorithm Workflow

**Training Procedure:**

```
For b = 1 to B:
    1. Draw bootstrap sample Z* of size N from training data
    2. Grow tree T_b on Z*:
       For each node:
           a. Randomly select m features from p total features
           b. Find best split among these m features only
           c. Split node using best feature/threshold
    3. Grow tree to maximum depth (no pruning)
```

**Prediction:**

- **Regression**: $\hat{f}_{rf}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^{B} T_b(\mathbf{x})$
- **Classification**: $\hat{y}_{rf}(\mathbf{x}) = \text{mode}\{T_1(\mathbf{x}), \ldots, T_B(\mathbf{x})\}$

### 4.4 Python Implementation

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import GridSearchCV

# === CLASSIFICATION EXAMPLE ===
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42
)

# Default Random Forest
rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_features='sqrt',     # sqrt(n_features) for classification
    max_depth=None,          # Grow trees fully
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf_clf.fit(X_train_i, y_train_i)
print(f"Random Forest Classification Accuracy: {rf_clf.score(X_test_i, y_test_i):.4f}")
print(f"OOB Score: {rf_clf.oob_score_:.4f}")

# === REGRESSION EXAMPLE ===
diabetes = load_diabetes()
X_diab, y_diab = diabetes.data, diabetes.target
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_diab, y_diab, test_size=0.2, random_state=42
)

rf_reg = RandomForestRegressor(
    n_estimators=500,
    max_features='sqrt',      # Can also use 1/3 for regression
    random_state=42,
    n_jobs=-1
)

rf_reg.fit(X_train_d, y_train_d)
from sklearn.metrics import mean_squared_error, r2_score

y_pred_d = rf_reg.predict(X_test_d)
mse = mean_squared_error(y_test_d, y_pred_d)
r2 = r2_score(y_test_d, y_pred_d)
print(f"\nRandom Forest Regression - MSE: {mse:.2f}, R^2: {r2:.4f}")
```

### 4.5 Hyperparameter Tuning

```python
# Grid search for optimal hyperparameters
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_features': ['sqrt', 'log2', 0.3],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_i, y_train_i)
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test_i, y_test_i):.4f}")
```

### 4.6 Feature Importance

Random Forests compute feature importance by measuring the **total reduction** in the splitting criterion (Gini impurity or MSE) attributed to each feature, averaged across all trees.

**Gini Importance (Classification):**
$$\text{Importance}(X_j) = \frac{1}{B}\sum_{b=1}^{B} \sum_{t \in T_b: v(t)=j} p(t) \Delta i(t)$$

where:

- $v(t)$ = feature used at node $t$
- $p(t)$ = proportion of samples reaching node $t$
- $\Delta i(t)$ = decrease in impurity from splitting node $t$

**Python Implementation:**

```python
# Feature importance analysis
importances = rf_clf.feature_importances_
feature_names = iris.feature_names
indices = np.argsort(importances)[::-1]

print("\nFeature Ranking:")
for i, idx in enumerate(indices):
    print(f"{i+1}. {feature_names[idx]:20s}: {importances[idx]:.4f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.title("Feature Importances - Iris Dataset")
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.ylabel('Importance Score')
plt.tight_layout()

# Permutation importance (more robust, model-agnostic)
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(
    rf_clf, X_test_i, y_test_i, 
    n_repeats=30, 
    random_state=42,
    n_jobs=-1
)

print("\nPermutation Importance (Mean ± Std):")
for i, idx in enumerate(indices):
    print(f"{feature_names[idx]:20s}: {perm_importance.importances_mean[idx]:.4f} "
          f"± {perm_importance.importances_std[idx]:.4f}")
```

**Interpretation:**

- **High importance**: Feature frequently selected and leads to significant impurity reduction
- **Low importance**: Feature rarely used or minimally reduces impurity

**Caveats:**

- Biased toward high-cardinality features
- Correlated features share importance
- Permutation importance addresses some biases

### 4.7 Extra-Trees (Extremely Randomized Trees)

**Modification:** Use **random thresholds** for splits instead of searching for optimal thresholds.

**Algorithm Change:**

```
Standard Random Forest split:
    - Consider m random features
    - For each feature, find OPTIMAL threshold (minimize Gini/MSE)
    
Extra-Trees split:
    - Consider m random features
    - For each feature, select RANDOM threshold
    - Choose best among random feature-threshold pairs
```

**Trade-offs:**

- **Faster training**: No threshold optimization
- **Lower variance**: More randomization → less correlation
- **Slightly higher bias**: Suboptimal splits

**Python Implementation:**

```python
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
import time

# Compare training time and performance
start_time = time.time()
et_clf = ExtraTreesClassifier(
    n_estimators=500,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
et_clf.fit(X_train_i, y_train_i)
et_time = time.time() - start_time
et_score = et_clf.score(X_test_i, y_test_i)

start_time = time.time()
rf_clf_compare = RandomForestClassifier(
    n_estimators=500,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_clf_compare.fit(X_train_i, y_train_i)
rf_time = time.time() - start_time
rf_score = rf_clf_compare.score(X_test_i, y_test_i)

print(f"\nExtra-Trees: Accuracy={et_score:.4f}, Time={et_time:.2f}s")
print(f"Random Forest: Accuracy={rf_score:.4f}, Time={rf_time:.2f}s")
print(f"Speedup: {rf_time/et_time:.2f}x")
```

**When to use Extra-Trees:**

- Large datasets where training time is critical
- High-dimensional data
- When variance reduction is more important than maintaining low bias

---

## V. Boosting

### 5.1 Conceptual Framework

**Boosting** combines **weak learners** (slightly better than random guessing) sequentially to create a **strong learner**.

**Weak Learner Definition:**  
A classifier with error rate $\epsilon < 0.5$ (better than random).

**Boosting Guarantee (Probability Approximately Correct Learning):**  
Given sufficient weak PAC Learnable learners, boosting provably achieves arbitrarily low training error.

**Key Difference from Bagging:**

- **Bagging**: Parallel, independent training → variance reduction
- **Boosting**: Sequential, adaptive training → bias reduction

### 5.2 AdaBoost (Adaptive Boosting)

#### 5.2.1 Algorithm Overview

**Core Idea:** Each iteration focuses on instances misclassified by previous predictors by **increasing their weights**.

**Mathematical Formulation (SAMME):**

**Initialization:**
$$w_1^{(i)} = \frac{1}{m} \quad \text{for } i = 1, \ldots, m$$

**For** $j = 1$ **to** $M$ **(number of iterations):**

1. **Train** weak learner $h_j$ on weighted dataset $(X, y, w_j)$

2. **Compute weighted error rate:**
$$r_j = \frac{\sum_{i=1}^{m} w_j^{(i)} \cdot \mathbb{1}(h_j(\mathbf{x}^{(i)}) \neq y^{(i)})}{\sum_{i=1}^{m} w_j^{(i)}}$$

3. **Compute predictor weight:**
$$\alpha_j = \eta \log\left(\frac{1 - r_j}{r_j}\right) + \eta \log(K - 1)$$
   where $K$ is number of classes, $\eta$ is learning rate

4. **Update instance weights:**
$$w_{j+1}^{(i)} = w_j^{(i)} \cdot \exp(\alpha_j \cdot \mathbb{1}(h_j(\mathbf{x}^{(i)}) \neq y^{(i)}))$$

5. **Normalize weights:**
$$w_{j+1}^{(i)} \leftarrow \frac{w_{j+1}^{(i)}}{\sum_{k=1}^{m} w_{j+1}^{(k)}}$$

**Final Prediction:**
$$\hat{y}(\mathbf{x}) = \underset{k}{\operatorname{argmax}} \sum_{j=1}^{M} \alpha_j \cdot \mathbb{1}(h_j(\mathbf{x}) = k)$$

#### 5.2.2 Intuition Behind Formulas

**Predictor Weight** $\alpha_j$:

- If $r_j = 0$ (perfect): $\alpha_j \to \infty$ (maximum influence)
- If $r_j = 0.5$ (random): $\alpha_j = 0$ (no influence)
- If $r_j \to 1$ (anti-correlated): $\alpha_j < 0$ (reverse prediction)

**Weight Update:**

- Misclassified: $w^{(i)} \gets w^{(i)} \cdot e^{\alpha_j}$ (increase exponentially)
- Correctly classified: $w^{(i)}$ unchanged (before normalization)

**Effect:** Next iteration focuses on "hard" examples.

#### 5.2.3 Python Implementation

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Weak learner: shallow decision tree (stump)
weak_learner = DecisionTreeClassifier(max_depth=1, random_state=42)

# AdaBoost with SAMME algorithm
ada_clf = AdaBoostClassifier(
    estimator=weak_learner,
    n_estimators=200,
    learning_rate=1.0,      # η (eta)
    # algorithm='SAMME',
    random_state=42
)

ada_clf.fit(X_train, y_train)
ada_pred = ada_clf.predict(X_test)
ada_score = accuracy_score(y_test, ada_pred)
print(f"AdaBoost Accuracy: {ada_score:.4f}")

# Access predictor weights and errors
print(f"\nFirst 10 estimator weights (α): {ada_clf.estimator_weights_[:10]}")
print(f"First 10 estimator errors: {ada_clf.estimator_errors_[:10]}")

# Staged predictions (predictions after each iteration)
staged_scores = []
for i, y_pred_staged in enumerate(ada_clf.staged_predict(X_test)):
    staged_scores.append(accuracy_score(y_test, y_pred_staged))

# Visualize learning curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(staged_scores) + 1), staged_scores, marker='o', markersize=3)
plt.xlabel('Number of Estimators')
plt.ylabel('Test Accuracy')
plt.title('AdaBoost: Accuracy vs. Number of Estimators')
plt.grid(True, alpha=0.3)
plt.tight_layout()
```

#### 5.2.4 Learning Rate and Regularization

**Learning Rate** $\eta$ scales predictor contributions:
$$\alpha_j \leftarrow \eta \cdot \alpha_j$$

#### 5.2.4 Learning Rate and Regularization

**Learning Rate** $\eta$ (shrinkage parameter) scales predictor contributions to control the ensemble's learning speed:

$$\alpha_j \leftarrow \eta \cdot \alpha_j$$

**Effect on Bias-Variance Trade-off:**

- **High $\eta$ (e.g., 1.0)**:
  - Faster convergence
  - Each weak learner has maximum influence
  - Higher risk of overfitting
  - Fewer iterations needed

- **Low $\eta$ (e.g., 0.1, 0.01)**:
  - Slower, more careful learning
  - Requires more estimators ($M$) to achieve similar performance
  - Better generalization (lower variance)
  - More robust to noise in training data

**Optimal Strategy:**  
Use smaller $\eta$ with larger $M$. Common practice: $\eta \in [0.01, 0.1]$ with $M \in [1000, 5000]$.

**Statistical Insight:**  
The learning rate implements **shrinkage**, analogous to ridge regression. By constraining the contribution of each weak learner, we prevent any single predictor from dominating the ensemble, which reduces overfitting to training data peculiarities.

**Python Implementation:**

```python
# Compare different learning rates
learning_rates = [1.0, 0.5, 0.1, 0.01]
n_estimators = 200

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, lr in enumerate(learning_rates):
    ada_clf_lr = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n_estimators,
        learning_rate=lr,
        # algorithm='SAMME',
        random_state=42
    )
    
    ada_clf_lr.fit(X_train, y_train)
    
    # Track training and test accuracy over iterations
    train_scores = []
    test_scores = []
    
    for train_pred, test_pred in zip(
        ada_clf_lr.staged_predict(X_train),
        ada_clf_lr.staged_predict(X_test)
    ):
        train_scores.append(accuracy_score(y_train, train_pred))
        test_scores.append(accuracy_score(y_test, test_pred))
    
    axes[idx].plot(range(1, n_estimators + 1), train_scores, 
                   label='Train', linewidth=2)
    axes[idx].plot(range(1, n_estimators + 1), test_scores, 
                   label='Test', linewidth=2)
    axes[idx].set_xlabel('Number of Estimators')
    axes[idx].set_ylabel('Accuracy')
    axes[idx].set_title(f'Learning Rate: {lr}')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)
    
    # Detect overfitting
    gap = train_scores[-1] - test_scores[-1]
    axes[idx].text(0.02, 0.02, f'Train-Test Gap: {gap:.4f}', 
                   transform=axes[idx].transAxes, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.suptitle('AdaBoost: Learning Rate Impact on Overfitting', 
             fontsize=14, y=1.02)
```

**Interpretation Guidelines:**

- **Converging curves**: Good generalization, model is learning robust patterns
- **Diverging curves**: Overfitting-reduce $\eta$ or increase regularization
- **Train accuracy → 1.0 rapidly**: Likely overfitting-use smaller $\eta$

#### 5.2.5 Early Stopping

**Monitoring validation performance** during training enables early stopping to prevent overfitting:

```python
from sklearn.model_selection import train_test_split

# Split training data into train/validation
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Train with validation monitoring
n_estimators = 500
ada_clf_early = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=n_estimators,
    learning_rate=0.1,
    algorithm='SAMME',
    random_state=42
)

ada_clf_early.fit(X_train_sub, y_train_sub)

# Find optimal number of estimators
val_scores = []
for val_pred in ada_clf_early.staged_predict(X_val):
    val_scores.append(accuracy_score(y_val, val_pred))

optimal_n_estimators = np.argmax(val_scores) + 1
best_val_score = val_scores[optimal_n_estimators - 1]

print(f"Optimal number of estimators: {optimal_n_estimators}")
print(f"Best validation accuracy: {best_val_score:.4f}")

# Retrain with optimal number on full training set
ada_clf_final = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=optimal_n_estimators,
    learning_rate=0.1,
    algorithm='SAMME',
    random_state=42
)

ada_clf_final.fit(X_train, y_train)
final_test_score = ada_clf_final.score(X_test, y_test)
print(f"Final test accuracy: {final_test_score:.4f}")

# Visualize early stopping
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_estimators + 1), val_scores, linewidth=2)
plt.axvline(x=optimal_n_estimators, color='r', linestyle='--', 
            label=f'Optimal: {optimal_n_estimators} estimators')
plt.xlabel('Number of Estimators')
plt.ylabel('Validation Accuracy')
plt.title('AdaBoost: Early Stopping via Validation Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
```

---

### 5.3 Gradient Boosting (GBM/GBRT)

#### 5.3.1 Theoretical Foundation

**Gradient Boosting** frames ensemble learning as **functional gradient descent** in hypothesis space. Instead of adjusting instance weights (AdaBoost), it trains each new predictor to fit the **residual errors** (negative gradients) of the current ensemble.

**Optimization Perspective:**  
Minimize a differentiable loss function $L(y, F(\mathbf{x}))$ by iteratively adding functions:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

where $h_m$ is trained to approximate the negative gradient:

$$h_m \approx -\nabla_{F_{m-1}} L(y, F_{m-1}(\mathbf{x}))$$

**For Squared Loss** (regression):

$$L(y, F(\mathbf{x})) = \frac{1}{2}(y - F(\mathbf{x}))^2$$

$$-\frac{\partial L}{\partial F(\mathbf{x})} = y - F(\mathbf{x}) = \text{residual}$$

Thus, each tree fits the residuals of the previous ensemble.

**For Deviance Loss** (classification):

$$L(y, F(\mathbf{x})) = -\log P(y|\mathbf{x})$$

The negative gradient becomes the deviance residuals, which are more complex but follow the same principle.

#### 5.3.2 Algorithm: Gradient Boosting for Regression

**Initialization:**
$$F_0(\mathbf{x}) = \underset{\gamma}{\operatorname{argmin}} \sum_{i=1}^{m} L(y^{(i)}, \gamma)$$

For squared loss: $F_0(\mathbf{x}) = \bar{y}$ (mean of training targets)

**For** $m = 1$ **to** $M$:

1. **Compute pseudo-residuals** (negative gradients):
$$r_m^{(i)} = -\left[\frac{\partial L(y^{(i)}, F(\mathbf{x}^{(i)}))}{\partial F(\mathbf{x}^{(i)})}\right]_{F = F_{m-1}} \quad \text{for } i = 1, \ldots, n$$

   For squared loss: $r_m^{(i)} = y^{(i)} - F_{m-1}(\mathbf{x}^{(i)})$

2. **Fit base learner** $h_m(\mathbf{x})$ to pseudo-residuals:
$$h_m = \underset{h}{\operatorname{argmin}} \sum_{i=1}^{n} (r_m^{(i)} - h(\mathbf{x}^{(i)}))^2$$

3. **Update ensemble:**
$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

**Final Model:**
$$F_M(\mathbf{x}) = F_0(\mathbf{x}) + \eta \sum_{m=1}^{M} h_m(\mathbf{x})$$

#### 5.3.3 Python Implementation: Basic Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.datasets import load_diabetes, make_classification
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === REGRESSION EXAMPLE ===
X_diab, y_diab = load_diabetes(return_X_y=True)
X_train_gb, X_test_gb, y_train_gb, y_test_gb = train_test_split(
    X_diab, y_diab, test_size=0.2, random_state=42
)

# Gradient Boosting Regressor
gbm_reg = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,              # Shallow trees (weak learners)
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=1.0,            # Use 100% of samples (no stochastic)
    random_state=42,
    verbose=0
)

gbm_reg.fit(X_train_gb, y_train_gb)
y_pred_gb = gbm_reg.predict(X_test_gb)

print("Gradient Boosting Regression Results:")
print(f"  MSE: {mean_squared_error(y_test_gb, y_pred_gb):.2f}")
print(f"  MAE: {mean_absolute_error(y_test_gb, y_pred_gb):.2f}")
print(f"  R^2: {r2_score(y_test_gb, y_pred_gb):.4f}")

# Visualize staged predictions (learning trajectory)
train_scores = []
test_scores = []

for i, (train_pred, test_pred) in enumerate(zip(
    gbm_reg.staged_predict(X_train_gb),
    gbm_reg.staged_predict(X_test_gb)
)):
    train_scores.append(mean_squared_error(y_train_gb, train_pred))
    test_scores.append(mean_squared_error(y_test_gb, test_pred))

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_scores) + 1), train_scores, 
         label='Train MSE', linewidth=2)
plt.plot(range(1, len(test_scores) + 1), test_scores, 
         label='Test MSE', linewidth=2)
plt.xlabel('Number of Trees')
plt.ylabel('Mean Squared Error')
plt.title('Gradient Boosting: Training Trajectory')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Find optimal number of trees (early stopping point)
optimal_trees = np.argmin(test_scores) + 1
print(f"\nOptimal number of trees: {optimal_trees}")
print(f"Test MSE at optimal point: {test_scores[optimal_trees - 1]:.2f}")
```

#### 5.3.4 Regularization Techniques

Gradient Boosting is prone to overfitting. Multiple regularization strategies are available:

**1. Shrinkage (Learning Rate $\eta$)**

As discussed earlier, smaller $\eta$ requires more trees but improves generalization:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x}), \quad 0 < \eta \leq 1$$

**Typical values**: $\eta \in [0.01, 0.1]$

**2. Tree Constraints**

Limit complexity of individual trees:

- `max_depth`: Typical range [3, 8]-shallow trees as weak learners
- `min_samples_split`: Minimum samples to split a node [2, 20]
- `min_samples_leaf`: Minimum samples per leaf [1, 10]
- `max_leaf_nodes`: Direct constraint on tree size

**3. Stochastic Gradient Boosting (Subsampling)**

Introduce randomness by training each tree on a random subsample:

$$\text{subsample} \in (0, 1]$$

- `subsample=0.5`: Each tree sees 50% of training data
- Adds variance, reduces overfitting
- Speeds up training

**4. Feature Subsampling**

Similar to Random Forests, consider random feature subsets:

- `max_features`: Proportion or number of features per split

```python
# Comprehensive regularization example
gbm_reg_tuned = GradientBoostingRegressor(
    # Ensemble parameters
    n_estimators=1000,
    learning_rate=0.01,        # Slow learning
    
    # Tree structure constraints
    max_depth=4,               # Shallow trees
    min_samples_split=10,      # Require more samples to split
    min_samples_leaf=4,        # Larger leaf size
    max_leaf_nodes=15,         # Explicit tree size limit
    
    # Stochastic components
    subsample=0.8,             # 80% of samples per tree
    max_features='sqrt',       # Random feature subset
    
    # Other
    random_state=42,
    validation_fraction=0.1,   # 10% for early stopping
    n_iter_no_change=50,       # Stop if no improvement for 50 iterations
    tol=1e-4
)

gbm_reg_tuned.fit(X_train_gb, y_train_gb)
y_pred_tuned = gbm_reg_tuned.predict(X_test_gb)

print("\nTuned Gradient Boosting Results:")
print(f"  MSE: {mean_squared_error(y_test_gb, y_pred_tuned):.2f}")
print(f"  R^2: {r2_score(y_test_gb, y_pred_tuned):.4f}")
print(f"  Number of trees used: {gbm_reg_tuned.n_estimators_}")
```

#### 5.3.5 Loss Functions

Gradient Boosting supports various loss functions for different tasks:

**Regression:**

1. **Squared Loss** (`loss='squared_error'`):
   $$L(y, F) = \frac{1}{2}(y - F)^2$$
   - Default choice
   - Sensitive to outliers

2. **Absolute Loss** (`loss='absolute_error'`):
   $$L(y, F) = |y - F|$$
   - Robust to outliers (minimizes median)
   - Slower convergence

3. **Huber Loss** (`loss='huber'`):
   $$L(y, F) = \begin{cases}
   \frac{1}{2}(y - F)^2 & \text{if } |y - F| \leq \delta \\
   \delta(|y - F| - \frac{\delta}{2}) & \text{otherwise}
   \end{cases}$$
   - Combines squared and absolute loss
   - Robust to outliers while maintaining efficiency

4. **Quantile Loss** (`loss='quantile'`):
   - Predicts specific quantiles (e.g., 0.9 for 90th percentile)
   - Useful for uncertainty estimation

**Classification:**

1. **Deviance** (`loss='log_loss'`):
   - Multi-class log-likelihood
   - Produces probability estimates

2. **Exponential Loss** (`loss='exponential'`):
   - Binary classification only
   - Equivalent to AdaBoost
   - More sensitive to outliers than deviance

```python
# Compare loss functions for regression
losses = ['squared_error', 'absolute_error', 'huber']
results = {}

for loss_fn in losses:
    gbm = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        loss=loss_fn,
        random_state=42
    )
    gbm.fit(X_train_gb, y_train_gb)
    y_pred = gbm.predict(X_test_gb)
    
    mse = mean_squared_error(y_test_gb, y_pred)
    mae = mean_absolute_error(y_test_gb, y_pred)
    results[loss_fn] = {'MSE': mse, 'MAE': mae}
    
    print(f"\nLoss: {loss_fn}")
    print(f"  MSE: {mse:.2f}")
    print(f"  MAE: {mae:.2f}")
```

#### 5.3.6 Histogram-Based Gradient Boosting (HistGradientBoosting)

**Motivation:** Standard GBRT has complexity $O(n \times d \times \log(n))$ for sorting features at each split. For large datasets, this becomes prohibitive.

**Solution:** Bin continuous features into discrete bins (typically 255), reducing complexity to $O(b \times d \times n)$ where $b \ll n$.

**Key Advantages:**

1. **Speed**: 10-100x faster on large datasets
2. **Memory efficiency**: Reduced memory footprint
3. **Native handling** of missing values
4. **Built-in categorical support**
5. **Monotonic constraints** available

**Algorithm Modifications:**

- Pre-bin all features into integer bins (default: 255 bins)
- Split search operates on bin indices, not raw values
- Enables GPU acceleration (via CUDA)

```python
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
import time

# Compare standard vs histogram-based on larger dataset
from sklearn.datasets import make_regression

X_large, y_large = make_regression(
    n_samples=100000, 
    n_features=20, 
    noise=0.1, 
    random_state=42
)

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_large, y_large, test_size=0.2, random_state=42
)

# Standard Gradient Boosting
start = time.time()
gbm_standard = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
gbm_standard.fit(X_train_l, y_train_l)
time_standard = time.time() - start
score_standard = r2_score(y_test_l, gbm_standard.predict(X_test_l))

# Histogram-based Gradient Boosting
start = time.time()
hgb = HistGradientBoostingRegressor(
    max_iter=100,          # Equivalent to n_estimators
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
hgb.fit(X_train_l, y_train_l)
time_hgb = time.time() - start
score_hgb = r2_score(y_test_l, hgb.predict(X_test_l))

print(f"\nPerformance Comparison (100k samples, 20 features):")
print(f"Standard GBM:      R^2={score_standard:.4f}, Time={time_standard:.2f}s")
print(f"Histogram GBM:     R^2={score_hgb:.4f}, Time={time_hgb:.2f}s")
print(f"Speedup:           {time_standard/time_hgb:.1f}x")
```

**Advanced Features:**

```python
# Missing value handling and categorical features
# Create data with missing values and categorical features
X_missing = X_train_l.copy()
mask = np.random.random(X_missing.shape) < 0.1
X_missing[mask] = np.nan

# Add categorical feature
cat_feature = np.random.choice(['A', 'B', 'C'], size=len(X_missing))
X_with_cat = np.column_stack([X_missing, cat_feature])

# HistGradientBoosting handles this natively
hgb_advanced = HistGradientBoostingRegressor(
    max_iter=100,
    categorical_features=[20],    # Index of categorical feature
    random_state=42
)

# Note: For categorical features, encode as integers first
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_with_cat[:, -1] = le.fit_transform(cat_feature)
X_with_cat = X_with_cat.astype(float)

hgb_advanced.fit(X_with_cat, y_train_l)
print(f"\nHistGBM with missing values + categorical: R^2={hgb_advanced.score(X_with_cat[:len(y_train_l)], y_train_l):.4f}")
```

**When to Use HistGradientBoosting:**

- Dataset size > 10,000 samples
- Many features (> 50)
- Presence of missing values
- Categorical features
- Need for speed without sacrificing accuracy

---

## VI. Stacking (Stacked Generalization)

### 6.1 Conceptual Framework

**Stacking** learns to optimally combine predictions from diverse base models using a **meta-learner** (blender). Unlike voting, which uses fixed combination rules, stacking **learns** the optimal weighting.

**Architecture:**

```
Training Data → Base Model 1 → Predictions 1 ↘
              → Base Model 2 → Predictions 2 → Meta-Learner → Final Prediction
              → Base Model 3 → Predictions 3 ↗
```

**Key Insight:** Base models may have complementary strengths/weaknesses. The meta-learner discovers optimal combination weights for different regions of the input space.

### 6.2 Training Procedure

**Critical Requirement:** Meta-learner must train on **out-of-sample predictions** to avoid overfitting.

**Standard Protocol (Cross-Validation Based):**

1. **Split training data** using K-fold cross-validation (e.g., K=5)

2. **For each fold** $k$:
   - Train each base model on other $K-1$ folds
   - Generate predictions on held-out fold $k$

3. **Concatenate** all out-of-fold predictions → Meta-training set

4. **Train meta-learner** on meta-training set

5. **Retrain base models** on full training data for final deployment

**Mathematical Formulation:**

Let $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{m}$ be the training set.

**Step 1:** Partition $\mathcal{D}$ into $K$ folds: $\mathcal{D} = \mathcal{D}_1 \cup \mathcal{D}_2 \cup \cdots \cup \mathcal{D}_K$

**Step 2:** For each base learner $h_j$ and fold $k$:

$$h_j^{(-k)} = \text{train}(h_j, \mathcal{D} \setminus \mathcal{D}_k)$$

$$\hat{y}_{j,k}^{(i)} = h_j^{(-k)}(\mathbf{x}^{(i)}) \quad \text{for } (\mathbf{x}^{(i)}, y^{(i)}) \in \mathcal{D}_k$$

**Step 3:** Construct meta-training set:

$$\mathcal{D}_{\text{meta}} = \left\{([\hat{y}_{1}^{(i)}, \hat{y}_{2}^{(i)}, \ldots, \hat{y}_{L}^{(i)}], y^{(i)})\right\}_{i=1}^{m}$$

where $\hat{y}_{j}^{(i)}$ are the out-of-fold predictions from base learner $j$.

**Step 4:** Train meta-learner:

$$g = \text{train}(\text{meta-learner}, \mathcal{D}_{\text{meta}})$$

**Final prediction:**

$$\hat{y}(\mathbf{x}) = g(h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_L(\mathbf{x}))$$

### 6.3 Python Implementation

```python
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

# === CLASSIFICATION EXAMPLE ===

# Define diverse base learners
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('nb', GaussianNB())
]

# Define meta-learner (typically simple model)
meta_learner = LogisticRegression(max_iter=1000, random_state=42)

# Create stacking classifier
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,                        # 5-fold CV for generating meta-features
    stack_method='auto',         # Use predict_proba if available
    n_jobs=-1
)

# Train and evaluate
stacking_clf.fit(X_train, y_train)
stacking_score = stacking_clf.score(X_test, y_test)

print("Stacking Classifier Results:")
print(f"  Test Accuracy: {stacking_score:.4f}")

# Compare with individual base learners
print("\nBase Learner Performance:")
for name, model in base_learners:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"  {name:10s}: {score:.4f}")

# Compare with simple voting
voting_clf = VotingClassifier(
    estimators=base_learners,
    voting='soft',
    n_jobs=-1
)
voting_clf.fit(X_train, y_train)
voting_score = voting_clf.score(X_test, y_test)
print(f"\nSoft Voting:     {voting_score:.4f}")
print(f"Stacking:        {stacking_score:.4f}")
print(f"Improvement:     {(stacking_score - voting_score)*100:.2f}%")
```

### 6.4 Multi-Layer Stacking

Stack multiple layers of meta-learners for marginal gains:

```
Layer 0 (Base): [RF, SVM, KNN, NB]
                        ↓
Layer 1 (Meta-1): [GBM, Ridge]
                        ↓
Layer 2 (Meta-2): [Logistic Regression]
```

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# === Step 1: Define Layer 0 base learners ===
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=80, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('nb', GaussianNB())
]

# === Step 2: Define Layer 1 meta-learners ===
layer1_learners = [
    ('gbm', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('ridge', RidgeClassifier(random_state=42))
]

# === Step 3: Build Layer 1 block (takes outputs from Layer 0) ===
layer1_block = StackingClassifier(
    estimators=base_learners,
    final_estimator=GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
    stack_method='predict_proba',
    cv=3,
    n_jobs=-1
)

# === Step 4: Build the final stack (Layer 0 → Layer 1 → Layer 2) ===
# Here, the entire layer1_block becomes a single base estimator.
multi_layer_stack = StackingClassifier(
    estimators=[('L1_block', layer1_block), *layer1_learners],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    stack_method='auto',
    cv=3,
    n_jobs=-1
)

# === Step 5: Train and evaluate ===
multi_layer_stack.fit(X_train, y_train)
y_pred = multi_layer_stack.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Multi-layer Stacking Accuracy: {acc:.4f}")

```
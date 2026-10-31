# AirSense — Model Theory Reference

**Project:** AirSense — Air Quality Classification Using Machine Learning  
**Models:** Decision Tree · Random Forest · SVM · XGBoost  
**Target:** `AQI_Bucket` (6-class: Good → Satisfactory → Moderate → Poor → Very Poor → Severe)

---

## Part I — Decision Tree

### 1.1 What is a Decision Tree?

A Decision Tree is a supervised learning algorithm that predicts an output class by asking a sequence of binary questions about the input features, one feature at a time. The model learns which questions to ask — and in which order — from the training data.

**Example in plain language:**

Imagine a doctor diagnosing air quality from a city's pollution readings:

> "Is PM2.5 above 120 µg/m³?"  
> → YES → "Is CO above 2 mg/m³?"  
>         → YES → **Predict: Very Poor**  
>         → NO  → **Predict: Poor**  
> → NO  → "Is Month in [11, 12, 1, 2]?" (Winter)  
>         → YES → "Is PM10 above 80 µg/m³?"  
>                 → YES → **Predict: Moderate**  
>                 → NO  → **Predict: Satisfactory**  
>         → NO  → **Predict: Good**

Each path from the root to a leaf is a human-readable IF-THEN rule.

---

### 1.2 Tree Anatomy

```
                         Root Node
                    [PM2.5 > 120 µg/m³?]
                       /          \
                     YES           NO
                    /               \
          Internal Node         Internal Node
         [CO > 2 mg/m³?]       [Month ∈ Winter?]
           /       \              /           \
         YES        NO           YES            NO
          |          |            |              |
      Leaf Node  Leaf Node   Internal Node   Leaf Node
      "Very Poor" "Poor"   [PM10 > 80 µg/m³?]  "Good"
                              /         \
                            YES          NO
                          Leaf Node   Leaf Node
                          "Moderate"  "Satisfactory"
```

- **Root Node**: The very first split. Chosen because it provides the most discriminative information across all training examples.
- **Internal Nodes**: Each internal node contains one feature and one threshold. All examples satisfying the condition go left; the rest go right.
- **Leaf Nodes**: Terminal nodes that output the majority class among all training examples that ended up in that leaf during training.

---

### 1.3 Splitting Criteria

At each node, the algorithm evaluates every possible (feature, threshold) combination and selects the one that maximises the reduction in **impurity**:

$$\text{Information Gain} = \text{Impurity}(\text{parent}) - \sum_{\text{child}} \frac{N_{\text{child}}}{N_{\text{parent}}} \times \text{Impurity}(\text{child})$$

#### Gini Impurity (Default in AirSense)

$$\text{Gini}(t) = 1 - \sum_{k=1}^{K} p_k^2$$

where $p_k$ = fraction of class $k$ at node $t$.

| Node contents | Gini |
|:-------------|:-----|
| All "Moderate" (pure) | $1 - 1^2 = 0$ |
| Equal mix of all 6 classes | $1 - 6 \times (1/6)^2 = 0.833$ |
| 90% "Poor", 10% "Very Poor" | $1 - (0.9^2 + 0.1^2) = 0.18$ |

A perfect split creates two child nodes with Gini = 0.

#### Entropy (Information Gain)

$$H(t) = -\sum_{k=1}^{K} p_k \log_2 p_k$$

Entropy has the same purpose as Gini — measuring node purity — but uses a logarithmic rather than quadratic formula. Both criteria generally produce similar trees.

---

### 1.4 How Splitting Works — Step by Step

1. At the current node, consider all features $f_1, f_2, \ldots, f_P$.
2. For each numerical feature $f_j$, sort all training sample values.
3. Evaluate each midpoint between consecutive distinct values as a candidate threshold $\tau$.
4. Compute the weighted Gini impurity of the two resulting child nodes.
5. Select $(f^*, \tau^*)$ = argmin weighted Gini.
6. Create two child nodes; recurse until a stopping criterion is met.

**Air quality example**: If PM2.5 thresholded at 60 µg/m³ separates "Good"/"Satisfactory" from "Poor"/"Severe" nearly perfectly, it gets a Gini gain of almost 0.83 — and becomes the first split.

---

### 1.5 Overfitting and Pruning

Without constraints, a Decision Tree will grow until every leaf contains exactly one training example — achieving 100% training accuracy but generalising poorly (high variance).

**AirSense hyperparameters used to control overfitting:**

| Parameter | Value | Effect |
|:----------|:------|:-------|
| `max_depth` | 10 | Limits the tree to at most 10 levels of splits |
| `min_samples_split` | 20 | A node must have ≥20 samples to be eligible for splitting |
| `min_samples_leaf` | 10 | Each leaf must represent ≥10 training examples |
| `class_weight` | `'balanced'` | Upweights minority classes (Good/Severe at 5.4%) |

**Cost-complexity pruning** (`ccp_alpha > 0`): After growing a full tree, sub-trees that add minimal predictive value relative to their complexity are removed. Not used in the default AirSense configuration but configurable.

---

### 1.6 Advantages and Limitations

| Advantages | Limitations |
|:-----------|:-----------|
| Fully interpretable — visualisable as rules | High variance — small data changes produce very different trees |
| Handles mixed feature types naturally | Cannot model diagonal/curved class boundaries |
| No feature scaling required | Tends to overfit without careful hyperparameter control |
| Naturally handles multiclass | Single tree can be less accurate than ensemble methods |
| Fast training and inference | Splits are axis-aligned — limited representational power |

---

## Part II — Random Forest

### 2.1 Ensemble Learning

A single model trained on a finite dataset reflects the noise in that particular sample. An **ensemble** trains many different models and aggregates their predictions, reducing the impact of any individual model's noise.

**Bias-Variance Tradeoff**:
- **Bias**: Error from systematic incorrect assumptions. An overly simple model has high bias.
- **Variance**: Error from sensitivity to fluctuations in training data. An overly complex model has high variance.
- A single deep Decision Tree has **low bias + high variance**.
- Random Forest combines many high-variance trees to produce **low bias + low variance**.

Ensemble error ≈ $\rho \sigma^2 + (1-\rho) \sigma^2/K$

where $K$ = number of trees, $\rho$ = correlation between trees, $\sigma^2$ = individual tree variance. Reducing $\rho$ (by randomising features) dramatically reduces ensemble variance.

---

### 2.2 Bootstrap Sampling

For each of the $K$ trees:
1. Draw a random bootstrap sample of $N$ rows **with replacement** from the training set.
2. On average, $\approx 63.2\%$ of rows appear in the bootstrap sample (the rest are "out-of-bag" rows).
3. Train one complete Decision Tree on this bootstrap sample.

The key outcome: each tree sees a **slightly different training set**, so they make **different errors**. When their predictions are averaged, independent errors cancel out.

**Out-of-Bag (OOB) Score**: Each training example is "left out" of approximately 37% of trees. The OOB score is computed by predicting each training example using only the trees it was NOT trained on — providing a free, unbiased validation estimate without a held-out set.

---

### 2.3 Random Feature Selection

At every node split in every tree, instead of searching all $P$ features, the algorithm randomly selects a subset of $\sqrt{P}$ features (for classification) and finds the best split among only those.

**Why not use all features?**  
If one strong predictor (e.g., PM2.5) exists, every tree would use it for the first split, making all trees nearly identical — and their correlated errors would NOT cancel when averaged.

Random feature selection **de-correlates** the trees. With $P = 17$ features, $\sqrt{17} \approx 4$ features are sampled at each node split.

---

### 2.4 Majority Voting

For classification, each tree produces a class prediction:
$$\hat{y}_k = \text{argmax}_c \; [\text{tree}_k \text{ score for class } c]$$

The ensemble's final prediction is the class chosen by the most trees (majority vote):
$$\hat{y} = \text{argmax}_c \sum_{k=1}^{K} \mathbf{1}[\hat{y}_k = c]$$

For 6-class AQI prediction with 200 trees: if 95 trees predict "Moderate", 67 predict "Poor", 20 predict "Very Poor", and 18 predict other classes → **Moderate** wins with 95 votes.

---

### 2.5 Single Tree vs. Random Forest

| Property | Decision Tree | Random Forest |
|:---------|:-------------|:--------------|
| Training data | Full dataset | K bootstrap samples |
| Feature selection | All P features per split | √P random features per split |
| Number of models | 1 | K (200 in AirSense) |
| Variance | High | Low |
| Bias | Low | Slightly higher |
| Interpretability | Full tree visualisable | "Black-box" ensemble |
| Overfitting resistance | Low (needs pruning) | High (ensemble averaging) |
| Training time | Fast | K × single-tree time |
| Feature importance | Single tree's MDI | Mean MDI across all K trees |

---

### 2.6 Advantages and Limitations

| Advantages | Limitations |
|:-----------|:-----------|
| Robust to outliers (averaging) | Less interpretable than a single tree |
| Built-in feature importance | Higher memory (K complete trees stored) |
| Excellent out-of-the-box performance | Slower than a single tree (K× training time) |
| Free OOB validation estimate | Cannot extrapolate beyond training value range |
| Handles class imbalance with `class_weight` | Still axis-aligned boundaries (rectangular regions) |

---

## Part III — Support Vector Machine (SVM)

### 3.1 Support Vectors

Given a training dataset $\{(\mathbf{x}_i, y_i)\}$ where $y_i \in \{-1, +1\}$ (binary case), the SVM finds a hyperplane that separates the two classes with **maximum margin**.

The training examples that lie exactly on the margin boundaries are the **support vectors** — they literally "support" (define) the hyperplane. Move any support vector and the hyperplane changes. Move any non-support-vector training example and the hyperplane stays identical.

For air quality: if "Moderate" and "Poor" days are close in PM2.5-PM10 space, only the borderline Moderate days closest to the Poor cluster, and vice versa, are support vectors.

---

### 3.2 Hyperplanes and Margins

In $P$-dimensional feature space, a hyperplane is defined as:
$$\mathbf{w} \cdot \mathbf{x} + b = 0$$

where $\mathbf{w} \in \mathbb{R}^P$ is the normal vector and $b$ is the bias.

The two **margin hyperplanes** are:
$$\mathbf{w} \cdot \mathbf{x} + b = +1 \quad \text{(positive margin boundary)}$$
$$\mathbf{w} \cdot \mathbf{x} + b = -1 \quad \text{(negative margin boundary)}$$

The **margin width** = $\frac{2}{\|\mathbf{w}\|}$.

---

### 3.3 Maximum-Margin Classification

The SVM training objective is a convex quadratic program:

$$\underset{\mathbf{w}, b}{\text{minimise}} \quad \frac{1}{2}\|\mathbf{w}\|^2$$
$$\text{subject to} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 \quad \text{for all } i$$

This is equivalent to maximising the margin $2/\|\mathbf{w}\|$. Since the problem is convex, the solution is **globally unique** — there is only one optimal hyperplane.

**Soft-margin SVM** (for non-separable data):
$$\underset{\mathbf{w}, b, \xi}{\text{minimise}} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i$$
$$\text{subject to} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

The **regularization parameter C** controls the trade-off:
- Large $C$: Strongly penalise misclassification → narrow margin → low bias, high variance.
- Small $C$: Accept more misclassification → wide margin → higher bias, lower variance.

---

### 3.4 The Kernel Trick

Real pollution data is not linearly separable in 12-dimensional feature space. The **kernel trick** implicitly maps $\mathbf{x}$ to a higher-dimensional space $\phi(\mathbf{x})$ without computing $\phi(\mathbf{x})$ explicitly:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)$$

The SVM dual formulation requires only inner products — replacing $\mathbf{x}_i \cdot \mathbf{x}_j$ with $K(\mathbf{x}_i, \mathbf{x}_j)$ allows infinite-dimensional mappings at finite cost.

---

### 3.5 Linear vs. RBF Kernels

| Property | Linear Kernel | RBF Kernel |
|:---------|:-------------|:-----------|
| Formula | $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i \cdot \mathbf{x}_j$ | $K(\mathbf{x}_i, \mathbf{x}_j) = e^{-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2}$ |
| Decision boundary | Linear hyperplane | Non-linear (circular/ellipsoidal regions) |
| Parameters | C only | C and γ |
| Training speed | Fast: O(N) to O(N²) | Slower: O(N²) to O(N³) |
| Memory | Low | High (N support vectors max) |
| Best for | High-dimensional sparse data | Dense, moderate-dimensional tabular data |

**AirSense choice**: RBF kernel with `C=10`, `gamma='scale'`. The boundary between "Moderate" and "Poor" in 17-dimensional pollutant space is not a hyperplane — it depends on complex interactions between PM2.5, PM10, and temporal variables. RBF can model these curved boundaries.

**gamma='scale'**: Sets $\gamma = 1 / (n\_features \times \text{Var}(\text{training } X))$. This adapts the kernel width to the feature count and scale automatically, and is the recommended default for scaled features.

---

### 3.6 Importance of Feature Scaling

Without scaling, SVM computes $\|\mathbf{x}_i - \mathbf{x}_j\|^2$:
$$\|(\text{PM2.5}_i - \text{PM2.5}_j, \ldots, \text{Benzene}_i - \text{Benzene}_j)\|^2$$

PM2.5 ranges 0–500; Benzene ranges 0–10. The PM2.5 term contributes 2,500× more to the distance than the Benzene term — effectively making Benzene invisible to the RBF kernel.

After `StandardScaler`:
- PM2.5: mean ≈ 0, std ≈ 1
- Benzene: mean ≈ 0, std ≈ 1
- All 17 features contribute equally to the kernel distance.

**This is why the AirSense pipeline always applies StandardScaler**, regardless of model.

---

### 3.7 Multiclass: One-vs-One (OvO) Strategy

sklearn's `SVC` handles 6-class AQI prediction via OvO:
- Train $\binom{6}{2} = 15$ binary SVMs, one for each pair of classes.
- At inference, each of the 15 binary SVMs votes for one class.
- The class with the most votes wins.

**class_weight='balanced'**: The class weight for class $k$ is $\frac{N}{n\_classes \times N_k}$, where $N_k$ is the count of class $k$. Minority classes (Good, Severe at 5.4%) receive higher weights, so their misclassification is penalised more heavily.

---

### 3.8 Computational Limitations

| Dataset size $N$ | Approximate training time | Memory |
|:----------------|:--------------------------|:-------|
| 1,000 | < 1 second | Trivial |
| 5,000 | 1–10 seconds | Low |
| 17,394 (full train) | 2–15 minutes (RBF) | Moderate |
| 100,000+ | Hours to days | Impractical |

For $N \approx 17K$, full RBF SVM training is feasible in minutes. The AirSense `run_svm()` function supports `max_train_samples` for a disclosed, stratified subsample when training time is impractical. Any subsampling is explicitly logged and reported — no silent downsampling.

---

## Part IV — XGBoost

### 4.1 Gradient Boosting

Gradient boosting builds a model $F_M(\mathbf{x}) = \sum_{m=0}^{M} \eta h_m(\mathbf{x})$ as an additive sequence, where each $h_m$ is a Decision Tree fitted to correct the **residual errors** of the current ensemble.

The key insight: instead of fitting $h_m$ to the original targets, fit it to the **negative gradient of the loss function** with respect to the current predictions. This generalises the concept of residuals to arbitrary differentiable loss functions.

---

### 4.2 Sequential Construction of Trees

**Algorithm (Gradient Boosting)**:

**Step 0**: Initialize with a constant prediction:
$$F_0(\mathbf{x}) = \underset{\gamma}{\text{argmin}} \sum_{i=1}^N L(y_i, \gamma)$$

**For $m = 1, 2, \ldots, M$:**

1. Compute pseudo-residuals (negative gradient):
$$r_{im} = -\left[\frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F = F_{m-1}}$$

2. Fit a Decision Tree $h_m(\mathbf{x})$ to the pairs $\{(\mathbf{x}_i, r_{im})\}_{i=1}^N$.

3. Update the model:
$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

where $\eta$ is the learning rate (shrinkage).

For 6-class AQI: one separate tree is trained per class per round (via one-vs-all decomposition), producing $6 \times M$ trees total.

---

### 4.3 Loss Functions and Gradient Information

For multiclass classification, XGBoost uses the **softmax cross-entropy** loss:

$$L(y, \hat{y}) = -\sum_{k=1}^{K} \mathbf{1}[y = k] \log p_k, \quad p_k = \frac{e^{\hat{y}_k}}{\sum_j e^{\hat{y}_j}}$$

The gradient (first derivative) tells each tree which direction and magnitude to move the prediction:
$$\frac{\partial L}{\partial \hat{y}_k} = p_k - \mathbf{1}[y = k]$$

The Hessian (second derivative) weights each gradient step:
$$\frac{\partial^2 L}{\partial \hat{y}_k^2} = p_k(1 - p_k)$$

XGBoost uses **both** gradient and Hessian information (Newton step) — this is what distinguishes it from classic gradient boosting, providing faster convergence and better leaf weight estimation.

---

### 4.4 Regularization

XGBoost's objective at each boosting round includes explicit regularization:

$$\text{Obj}^{(m)} = \sum_{i=1}^{N} L(y_i, \hat{y}_i^{(m-1)} + h_m(\mathbf{x}_i)) + \Omega(h_m)$$

where:
$$\Omega(h) = \gamma T + \frac{\lambda}{2} \sum_{j=1}^{T} w_j^2 + \alpha \sum_{j=1}^{T} |w_j|$$

- $T$: Number of leaf nodes in the tree.
- $w_j$: Weight (output value) of leaf $j$.
- $\gamma$: Minimum gain required to create a new split (tree pruning).
- $\lambda$: L2 regularization on leaf weights (Ridge penalty).
- $\alpha$: L1 regularization on leaf weights (Lasso penalty).

In AirSense: `reg_alpha=0.1`, `reg_lambda=1.0` — mild L1+L2 to prevent overfit.

---

### 4.5 Learning Rate and Boosting Rounds

The **learning rate** $\eta$ (called `learning_rate` in XGBoost, also called "shrinkage"):

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

- $\eta = 1.0$: Each tree makes a full Newton step. Fast but overfits.
- $\eta = 0.05$ (AirSense default): Each tree makes a 5% step. Many rounds needed for convergence but generalises better.

**Early stopping**: Instead of training exactly $M = 300$ rounds, AirSense uses the validation set loss as a stopping criterion: if the validation log-loss does not improve for 30 consecutive rounds, training stops. The model at the best validation round is kept. This avoids overfitting while finding the optimal $M$ automatically.

---

### 4.6 Stochastic Gradient Boosting

XGBoost extends gradient boosting with stochastic sampling at each round:

| Parameter | AirSense value | Effect |
|:----------|:-------------|:-------|
| `subsample` | 0.8 | Each tree trained on 80% of training rows (randomly sampled without replacement) |
| `colsample_bytree` | 0.8 | Each tree uses 80% of features (randomly sampled) |

Similar to Random Forest's bootstrap sampling and feature subsetting, stochastic gradient boosting:
- Reduces correlation between consecutive trees.
- Prevents overfitting on the full training set.
- Speeds training by reducing data processed per tree.

---

### 4.7 XGBoost vs. Random Forest

| Property | Random Forest | XGBoost |
|:---------|:-------------|:--------|
| Learning strategy | Bagging (parallel) | Boosting (sequential) |
| Tree correlation | De-correlated (by design) | Intentionally correlated (sequential correction) |
| Fitting target | Original class labels | Negative gradient (pseudo-residuals) |
| Regularization | Implicit (bootstrap + max_features) | Explicit (γ, λ, α) |
| Training speed | Fully parallelizable | Sequential — partial parallelism within each tree |
| Overfitting risk | Low | Higher if not tuned |
| Hyperparameter sensitivity | Low | Moderate (learning_rate, n_estimators interact) |
| Performance ceiling | High | Often state-of-the-art on tabular data |
| Interpretability | Feature importances | Feature importances + SHAP values |

**Both models are strong baselines. Actual performance on AirSense is determined empirically — neither is assumed superior without evidence.**

---

## Part V — Model Comparison Framework

### 5.1 Shared Evaluation Protocol

All four models are evaluated using **identical** conditions:
- **Same dataset**: `city_day.csv`, N=24,850 labeled rows.
- **Same splits**: Stratified 70/15/15 with `random_state=42`. Train: ~17,394 rows, Val: ~3,728, Test: ~3,728.
- **Same preprocessing pipeline**: WinsorizeTransformer → SimpleImputer → StandardScaler (numerical) + OrdinalEncoder (categorical).
- **Same metrics**: Accuracy, Macro F1, Weighted F1, Per-class Precision/Recall/F1, Confusion Matrix.

### 5.2 Evaluation Metrics Glossary

| Metric | Formula | Interpretation |
|:-------|:--------|:--------------|
| Accuracy | $\frac{TP}{N}$ | Fraction of all predictions that are correct |
| Macro Precision | $\frac{1}{K}\sum_k \frac{TP_k}{TP_k + FP_k}$ | Average precision across all 6 classes (equal weight) |
| Macro Recall | $\frac{1}{K}\sum_k \frac{TP_k}{TP_k + FN_k}$ | Average recall across all 6 classes (equal weight) |
| Macro F1 | $\frac{2 \times \text{Macro Precision} \times \text{Macro Recall}}{\text{Macro Precision} + \text{Macro Recall}}$ | Balanced harmonic mean; recommended for imbalanced multiclass |
| Weighted F1 | $\sum_k \frac{N_k}{N} F1_k$ | F1 weighted by class support; reflects overall model usefulness |
| Per-class F1 | $\frac{2 \times P_k \times R_k}{P_k + R_k}$ | Class-specific performance; critical for minority classes |

**Class imbalance note**: The AirSense dataset has severe imbalance (Moderate: 35.5% vs. Good: 5.4%). Accuracy can be misleadingly high (a model predicting only "Moderate" achieves 35.5% accuracy). **Macro F1 is the primary benchmark metric** as it treats all 6 classes equally.

### 5.3 `zero_division` Handling

When a model fails to predict any examples for a class (e.g., never predicts "Severe"), precision for that class is mathematically undefined ($0/0$). sklearn's `classification_report` with `zero_division=0` sets such metrics to 0.0 and issues a warning. This is honest: a model that never predicts a class has zero precision for it.

---

## Part VI — Actual Model Results

> Evaluated on the full 24,850 labeled records of `city_day.csv` (17,394 train, 3,728 val, 3,728 test).
> See `reports/model_results.md` for comprehensive per-class breakdowns, confusion matrices, and feature importances.

| Model | Train Accuracy | Val Accuracy | Test Accuracy | Test Macro F1 | Weighted F1 | Train Time |
|:------|:--------------:|:------------:|:-------------:|:-------------:|:-----------:|:----------:|
| **Decision Tree** | 0.7892 | 0.7465 | 0.7454 | 0.7301 | 0.7490 | 0.26s |
| **Random Forest** | 0.8873 | 0.7881 | 0.7881 | 0.7654 | 0.7903 | 2.95s |
| **SVM (RBF Kernel)** | 0.8270 | 0.7438 | 0.7361 | 0.7135 | 0.7410 | 8.10s |
| **XGBoost** | 0.9175 | **0.8090** | **0.8082** | **0.7816** | **0.8074** | 4.69s |

*All results recorded from reproducible runs with `random_state=42` under strict train/test isolation.*

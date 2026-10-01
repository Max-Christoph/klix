# Mathematical Specification of Klix Decision Heads

This document contains the formal mathematical definitions and linear algebra formulations for all decision heads evaluated within `klix-engine`.

---

## 1. Shared Feature Extraction

Let $x \in \mathcal{V}^*$ denote an input text sequence. The shared backbone projects $x$ into a dense vector space:

$$\mathbf{q}_{\text{raw}} = \text{Encoder}(x) \in \mathbb{R}^D$$

where $D = 384$ for `paraphrase-multilingual-MiniLM-L12-v2`. The vector is strictly $L_2$-normalized:

$$\mathbf{q} = \frac{\mathbf{q}_{\text{raw}}}{\|\mathbf{q}_{\text{raw}}\|_2}, \quad \|\mathbf{q}\|_2 = 1$$

All downstream decision heads evaluate concurrently over this single normalized vector $\mathbf{q}$.

---

## 2. Categorical Choice (`Choice`)

Let $\mathcal{C} = \{1, \dots, K\}$ denote the set of candidate classes. For each class $k$, the user provides a set of anchor vectors $\{\mathbf{x}_{k,1}, \dots, \mathbf{x}_{k,n_k}\} \subset \mathbb{R}^D$, with $\|\mathbf{x}_{k,i}\|_2 = 1$.

### Centroid Classifier (`classifier="centroid"`, Default)

The class representative $\mathbf{c}_k$ is the normalized geometric mean of its constituent anchor vectors:

$$\mathbf{c}_k = \frac{\sum_{i=1}^{n_k} \mathbf{x}_{k,i}}{\|\sum_{i=1}^{n_k} \mathbf{x}_{k,i}\|_2} \in \mathbb{R}^D$$

Inference computes similarities across all $K$ classes simultaneously via a matrix-vector product:

$$\mathbf{s} = \mathbf{C} \mathbf{q}, \quad \text{where } \mathbf{C} = \begin{bmatrix} \mathbf{c}_1^T \\ \vdots \\ \mathbf{c}_K^T \end{bmatrix} \in \mathbb{R}^{K \times D}$$

The predicted class is:

$$k^* = \arg\max_{k \in \{1,\dots,K\}} (\mathbf{q} \cdot \mathbf{c}_k)$$

Computational complexity is $O(K \cdot D)$ flops post-embedding.

### Out-of-Domain Rejection & Negative Poles

Given a set of contrastive reject anchors $\mathbf{R} = \{\mathbf{r}_1, \dots, \mathbf{r}_M\}$, the reject score is:

$$s_{\text{reject}}(\mathbf{q}) = \max_{m \in \{1,\dots,M\}} (\mathbf{q} \cdot \mathbf{r}_m)$$

The final decision value is nullified (`value=None`) if the reject pole dominates or if the winning score fails a minimum threshold $\tau_{\min}$:

$$\text{output}(\mathbf{q}) = \begin{cases} \text{None}, & \text{if } s_{\text{reject}}(\mathbf{q}) > s_{k^*}(\mathbf{q}) \;\lor\; s_{k^*}(\mathbf{q}) < \tau_{\min} \\ k^*, & \text{otherwise} \end{cases}$$

### Linear Probe (`classifier="linear"`)

Fits an $L_2$-regularized multinomial logistic regression boundary over the anchor pool:

$$k^* = \arg\max_{k \in \{1,\dots,K\}} (\mathbf{w}_k^T \mathbf{q} + b_k)$$

subject to:

$$\min_{\mathbf{W}, \mathbf{b}} \mathcal{L}_{\text{CE}}(\mathbf{W}, \mathbf{b}) + \frac{\lambda}{2} \|\mathbf{W}\|_F^2$$

---

## 3. Calibrated Multi-Label (`MultiLabel`)

Unlike categorical routing, each category is evaluated as an independent decision boundary. The raw cosine alignment $a_k = \mathbf{q} \cdot \mathbf{c}_k$ is mapped to a calibrated probability $s_k \in [0, 1]$ via a parameterized sigmoid transformation:

$$s_k(\mathbf{q}) = \sigma(\gamma \cdot (a_k - c_0)) = \frac{1}{1 + \exp(-\gamma \cdot (a_k - c_0))}$$

where:
* $\gamma > 0$ controls the logistic steepness (`sharpness`, default: `12.0`).
* $c_0 \in [-1, 1]$ defines the cosine similarity inflection center (`center`, default: `0.40`).

The set of active labels is obtained by threshold truncation:

$$\mathcal{Y}(\mathbf{q}) = \{k \in \mathcal{C} \mid s_k(\mathbf{q}) \ge \tau\}$$

---

## 4. Continuous Metric Projection (`Score`)

Maps query $\mathbf{q}$ onto a bounded interval $[V_{\min}, V_{\max}]$ based on relative proximity to positive and negative anchor distributions $\mathcal{A}_{\text{high}}$ and $\mathcal{A}_{\text{low}}$:

$$v(\mathbf{q}) = V_{\min} + (V_{\max} - V_{\min}) \cdot \frac{\text{top}_k(\mathbf{q}, \mathcal{A}_{\text{high}}) - \text{top}_k(\mathbf{q}, \mathcal{A}_{\text{low}}) + 1}{2}$$

---

## 5. Ternary Polarity Flag (`Flag`)

Evaluates binary classification against an explicit neutral rejection pole over a 3-component probability simplex:

$$P(y \mid \mathbf{q}) = \frac{\exp((\mathbf{q} \cdot \mathbf{c}_y) / T)}{\sum_{j \in \{\text{true}, \text{false}, \text{neutral}\}} \exp((\mathbf{q} \cdot \mathbf{c}_j) / T)}$$

where $T > 0$ is the softmax temperature. The head outputs:
* `True` if $P(\text{true}) \ge \tau$ and $P(\text{true}) > P(\text{neutral})$.
* `False` if $P(\text{false}) \ge \tau$ and $P(\text{false}) > P(\text{neutral})$.
* `None` if the neutral pole dominates ($P(\text{neutral}) \ge \max(P(\text{true}), P(\text{false}))$) or confidence is below threshold.

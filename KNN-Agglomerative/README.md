# 🍷 Wine Clustering Analysis with K-Means

This project performs **Exploratory Data Analysis (EDA)** and **unsupervised clustering** on a wine dataset using the **K-Means algorithm**. It also applies the **Elbow Method** to determine the optimal number of clusters, with support for visualization using `matplotlib` and `seaborn`.

---

## 🛠️ Technologies Used

- Python 3
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- [kneed](https://github.com/arvkevi/kneed) (optional, for elbow detection)

---

## 📈 Workflow

1. **Load & Inspect Data**  
   `pandas`, `.info()`, `.describe()`, null check, and class distribution

2. **Visualize**

   - Pairplot of feature distributions
   - Histogram of wine classes

3. **Preprocessing**

   - Feature matrix `X` is prepared by dropping the `Wine` column
   - Optional: scaling with `StandardScaler`

4. **K-Means Clustering**

   - Fit K-Means for values of `k` in range [2, 8]
   - Compute **SSE (Sum of Squared Errors)** for each `k`
   - Plot SSE vs. `k` to use the **Elbow Method**

5. **Optimal k Selection**

   - Visualized manually using the elbow plot
   - Optionally use the `kneed` package for automated knee detection

6. **Clustering and Output**

   - Fit K-Means with chosen `k` (e.g., `k=4`)
   - Predict cluster labels
   - Optional visualization using PCA

7. **#Agglomerative Hierachial Clustering**
   - Perform single-link and complete-link clustering
   - Visualize dendograms

---

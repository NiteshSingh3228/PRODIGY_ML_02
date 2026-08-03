# PRODIGY_ML_02: Customer Segmentation using K-Means Clustering
**Prodigy InfoTech — Machine Learning Internship**

## Objective
Group mall customers into segments based on **Annual Income** and **Spending Score** using **K-Means clustering**.

## Dataset
[Mall Customer Segmentation Data (Kaggle)](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) — 200 customers, columns: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100).

## Tools
Python, pandas, numpy, matplotlib, seaborn, scikit-learn (KMeans, StandardScaler, silhouette_score)

## Approach
1. Scale Income & Spending Score
2. Find optimal K via Elbow Method + Silhouette Score → **K = 5** (score ≈ 0.555)
3. Fit K-Means, visualize clusters (2D & 3D)
4. Profile & label segments; heatmap, gender/age breakdowns, cluster size pie chart

*Note: K-Means is unsupervised — no ground-truth labels, so no accuracy/confusion matrix. Silhouette score is used instead.*

## Results

| Cluster | Avg Income | Avg Spending | Segment |
|---|---|---|---|
| 0 | $55.3k | 49.5 | Standard |
| 1 | $86.5k | 82.1 | **Target** (high income, high spend) |
| 2 | $25.7k | 79.4 | Careless (low income, high spend) |
| 3 | $88.2k | 17.1 | Careful (high income, low spend) |
| 4 | $26.3k | 20.9 | Sensible (low income, low spend) |

**Target** segment = best marketing focus.

## Files
- `PRODIGY_ML_02.ipynb` — full notebook
- `Mall_Customers.csv` — original dataset
- `Mall_Customers_Clustered.csv` — data with cluster labels
- `output_images/` — saved chart images

## Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook PRODIGY_ML_02.ipynb
```
Place `Mall_Customers.csv` in the same folder before running.

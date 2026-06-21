# Customer Segmentation using K-Means Clustering

## 📌 Overview
This project segments customers into distinct groups based on their
**Age**, **Annual Income**, and **Spending Score** using the
**K-Means Clustering** algorithm. Customer segmentation helps
businesses understand different customer types so they can target
marketing, offers, and services more effectively.

---
## ⚙️ Requirements

- Python 3.8+
- Libraries:
  - numpy
  - pandas
  - matplotlib
  - scikit-learn

Install all dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn
```

(If using a restricted environment that requires it)
```bash
pip install numpy pandas matplotlib scikit-learn --break-system-packages
```

---

## ▶️ How to Run

```bash
python customer_segmentation.py
```

By default, the script **auto-generates a realistic sample dataset**
so it runs out of the box with no setup.

### Using your own dataset
1. Place your CSV file (e.g. `Mall_Customers.csv`) in the project folder.
   Expected columns:
   - `CustomerID`
   - `Gender`
   - `Age`
   - `Annual Income (k$)`
   - `Spending Score (1-100)`
2. Open `customer_segmentation.py` and update the config section:
   ```python
   USE_SAMPLE_DATA = False
   DATA_PATH = "Mall_Customers.csv"
   ```
3. If your column names differ, update the `features` list inside `main()`.

---

## 🧠 How It Works

| Step | Description |
|------|-------------|
| 1. Load Data | Reads CSV or generates synthetic customer data |
| 2. Preprocess | Scales features using `StandardScaler` |
| 3. Find Optimal k | Uses Elbow Method + Silhouette Score (k = 2–10) |
| 4. Fit K-Means | Trains final model with the best `k` |
| 5. Visualize | Projects clusters to 2D using PCA |
| 6. Profile Segments | Shows average Age/Income/Spending per cluster |
| 7. Save Output | Exports labeled CSV and plots |

---

## 📊 Output Files

| File | Description |
|------|--------------|
| `customer_segments_output.csv` | Original data + assigned `Cluster` column |
| `elbow_silhouette.png` | Elbow curve and silhouette score chart used to pick `k` |
| `customer_segments.png` | 2D PCA scatter plot showing customer clusters |

Console output also prints a **segment profile table** showing the
average Age, Income, and Spending Score per cluster, along with
customer counts.

---

## 🔧 Customization

- **Manually set number of clusters:** override inside `main()`:
  ```python
  best_k = 5   # instead of using auto-suggested k
  ```
- **Change clustering features:** edit the `features` list in `main()`:
  ```python
  features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
  ```
- **Use a different algorithm:** swap `KMeans` for `AgglomerativeClustering`
  or `DBSCAN` from `sklearn.cluster` if needed.

---

## 📈 Example Segment Profile

| Cluster | Age | Income (k$) | Spending Score | Count |
|---------|-----|--------------|------------------|-------|
| 0 | 56.4 | 61.9 | 63.1 | 95 |
| 1 | 32.7 | 33.5 | 88.1 | 105 |

*(Values will vary depending on dataset used)*

---

## 📝 Notes
- The silhouette score is used to automatically recommend the best
  number of clusters, but you're encouraged to inspect the elbow plot
  and adjust `k` based on domain knowledge.
- PCA is used **only for visualization** — clustering itself is done
  on the original scaled features, not the PCA-reduced ones.

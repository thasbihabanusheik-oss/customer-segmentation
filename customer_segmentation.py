
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
USE_SAMPLE_DATA = True          # set False to load your own CSV
DATA_PATH = "Mall_Customers.csv"  # used only if USE_SAMPLE_DATA = False
RANDOM_STATE = 42


# -------------------------------------------------------------------
# STEP 1: LOAD DATA
# -------------------------------------------------------------------
def load_data():
    if not USE_SAMPLE_DATA:
        df = pd.read_csv(DATA_PATH)
        return df

    # Generate a realistic synthetic customer dataset
    rng = np.random.default_rng(RANDOM_STATE)
    n = 200

    age = rng.integers(18, 70, n)
    gender = rng.choice(["Male", "Female"], n)

    # Income roughly correlated with age, plus noise
    income = (15 + (age - 18) * 1.2 + rng.normal(0, 12, n)).clip(15, 140)

    # Spending score: younger / mid-income customers tend to spend more
    spending = (100 - np.abs(age - 35) * 1.5 + rng.normal(0, 15, n)).clip(1, 100)

    df = pd.DataFrame({
        "CustomerID": np.arange(1, n + 1),
        "Gender": gender,
        "Age": age,
        "Annual Income (k$)": income.round(1),
        "Spending Score (1-100)": spending.round(0).astype(int),
    })
    return df


# -------------------------------------------------------------------
# STEP 2: PREPROCESS
# -------------------------------------------------------------------
def preprocess(df, features):
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# -------------------------------------------------------------------
# STEP 3: FIND OPTIMAL K (Elbow Method + Silhouette Score)
# -------------------------------------------------------------------
def find_optimal_k(X_scaled, k_range=range(2, 11)):
    inertias = []
    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    # Plot elbow curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(list(k_range), inertias, marker="o")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (WCSS)")

    axes[1].plot(list(k_range), sil_scores, marker="o", color="green")
    axes[1].set_title("Silhouette Score by k")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")

    plt.tight_layout()
    plt.savefig("elbow_silhouette.png", dpi=150)
    plt.close()

    # Pick k with the best silhouette score as the recommended value
    best_k = list(k_range)[int(np.argmax(sil_scores))]
    print(f"[INFO] Suggested optimal k (best silhouette score) = {best_k}")
    return best_k


# -------------------------------------------------------------------
# STEP 4: FIT FINAL MODEL
# -------------------------------------------------------------------
def fit_kmeans(X_scaled, k):
    km = KMeans(n_clusters=k, init="k-means++", random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"[INFO] Final model: k={k}, silhouette score={score:.3f}")
    return km, labels


# -------------------------------------------------------------------
# STEP 5: VISUALIZE CLUSTERS (PCA -> 2D)
# -------------------------------------------------------------------
def visualize_clusters(X_scaled, labels, k):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=60, edgecolor="k", alpha=0.8)
    plt.title(f"Customer Segments (k={k}) - PCA Projection")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig("customer_segments.png", dpi=150)
    plt.close()


# -------------------------------------------------------------------
# STEP 6: PROFILE EACH SEGMENT
# -------------------------------------------------------------------
def profile_segments(df, features):
    summary = df.groupby("Cluster")[features].mean().round(1)
    summary["Count"] = df.groupby("Cluster").size()
    print("\n=== Segment Profiles (mean values) ===")
    print(summary)
    return summary


# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------
def main():
    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

    df = load_data()
    print("=== Sample of loaded data ===")
    print(df.head())

    X_scaled, scaler = preprocess(df, features)

    best_k = find_optimal_k(X_scaled)

    # You can override best_k manually here if you have domain knowledge,
    # e.g. best_k = 5
    km, labels = fit_kmeans(X_scaled, best_k)

    df["Cluster"] = labels

    visualize_clusters(X_scaled, labels, best_k)
    profile_segments(df, features)

    # Save final labeled dataset
    df.to_csv("customer_segments_output.csv", index=False)
    print("\n[INFO] Saved labeled dataset to 'customer_segments_output.csv'")
    print("[INFO] Saved plots: 'elbow_silhouette.png', 'customer_segments.png'")


if __name__ == "__main__":
    main()

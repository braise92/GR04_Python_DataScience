# ------------------------------------------------------
# TP2 - Clustering de données financières (Corrigé)
# ------------------------------------------------------
# 1. Clustering des profils financiers avec K-Means
# 2. Clustering des profils de risque avec regroupement hiérarchique
# 3. Clustering basé sur la corrélation des rendements journaliers
# 4. Clustering direct sur profils de rendements journaliers (entreprises = lignes)
# 5. Évaluation et comparaison des algorithmes : KMeans, Hiérarchique, DBSCAN
# ------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

# 1. Clustering des profils financiers
def preprocess_financial_clustering(file_path):
    df = pd.read_csv(file_path)
    financial_features = ['forwardPE', 'beta', 'priceToBook', 'returnOnEquity']
    columns_needed = ['Name'] + financial_features
    df_filtered = df[columns_needed].dropna()
    names = df_filtered['Name']
    df_features = df_filtered[financial_features]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_features)
    return pd.DataFrame(data_scaled, columns=financial_features), names.reset_index(drop=True)

def elbow_method(data, max_k=10):
    inertias = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    plt.plot(range(1, max_k+1), inertias, marker='o')
    plt.title('Méthode du coude')
    plt.xlabel('K')
    plt.ylabel('Inertie')
    plt.grid(True)
    plt.show()

def do_kmeans_clustering(data, names, n_clusters=3, show_graph=True):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(data)
    df_clustered = pd.DataFrame(data, columns=data.columns)
    df_clustered['Cluster'] = clusters
    df_clustered['Name'] = names
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(data)
    if show_graph:
        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=clusters, cmap='tab10')
        plt.title("t-SNE des clusters financiers")
        plt.grid(True)
        plt.show()
    return df_clustered

# 2. Clustering des profils de risque
def preprocess_risk_clustering(file_path):
    df = pd.read_csv(file_path)
    features = ['debtToEquity', 'currentRatio', 'quickRatio', 'returnOnAssets']
    df_filtered = df[features + ['Name']].dropna()
    names = df_filtered['Name']
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_filtered[features])
    return pd.DataFrame(data_scaled, columns=features), names

def do_hierarchical_clustering(data, names, n_clusters=3):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = model.fit_predict(data)
    df_clustered = pd.DataFrame(data, columns=data.columns)
    df_clustered['Cluster'] = clusters
    df_clustered['Name'] = names.values
    return df_clustered

def plot_dendrogram(data, method='ward'):
    linked = linkage(data, method=method)
    dendrogram(linked, orientation='top', distance_sort='descending')
    plt.title("Dendrogramme des profils de risque")
    plt.grid(True)
    plt.show()

# 3. Clustering sur les corrélations des rendements journaliers
def prepare_returns_data(folder_path):
    filepaths = glob.glob(f"{folder_path}/*.csv")
    returns_dict = {}
    for path in filepaths:
        df = pd.read_csv(path)
        company_name = path.split("/")[-1].replace(".csv", "")
        if "Rendement" in df.columns:
            returns_dict[company_name] = df["Rendement"].reset_index(drop=True)
    returns_df = pd.DataFrame(returns_dict).fillna(method='ffill').fillna(method='bfill')
    return returns_df

def plot_correlation_dendrogram(returns_df):
    corr_matrix = returns_df.corr()
    distance_matrix = 1 - corr_matrix
    linked = linkage(distance_matrix, method='ward')
    dendrogram(linked, labels=returns_df.columns)
    plt.title("Corrélations de rendements journaliers")
    plt.grid(True)
    plt.show()

# 4. Clustering direct sur les profils de rendements journaliers (correctif)
def clustering_on_return_profiles(returns_df):
    returns_df_T = returns_df.T  # entreprises = lignes, jours = colonnes
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns_df_T)
    names = returns_df_T.index

    # KMeans sur profils de rendement
    elbow_method(pd.DataFrame(returns_scaled, index=names))
    kmeans = KMeans(n_clusters=3, random_state=0)
    clusters = kmeans.fit_predict(returns_scaled)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(returns_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=clusters, cmap='tab10')
    plt.title("t-SNE des clusters de profils de rendements (entreprises)")
    plt.grid(True)
    for i, name in enumerate(names):
        plt.annotate(name, (tsne_result[i, 0], tsne_result[i, 1]), fontsize=8)
    plt.show()

    return clusters

# 5. Évaluation et comparaison des algorithmes
def do_dbscan_clustering(data, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data)
    df_clustered = pd.DataFrame(data, columns=data.columns)
    df_clustered['Cluster'] = clusters
    return df_clustered

def evaluate_clustering(data, clustering_labels):
    if len(set(clustering_labels)) > 1:
        return silhouette_score(data, clustering_labels)
    else:
        return -1

def compare_algorithms(data_financial, data_risk, data_returns_T):
    results = []
    algos = {
        "K-Means": lambda data: KMeans(n_clusters=3, random_state=0).fit_predict(data),
        "Hierarchical": lambda data: AgglomerativeClustering(n_clusters=3).fit_predict(data),
        "DBSCAN": lambda data: do_dbscan_clustering(data)['Cluster']
    }
    for name, algo in algos.items():
        for dataset_name, dataset in [("Finance", data_financial), ("Risk", data_risk), ("Returns", data_returns_T)]:
            clusters = algo(dataset)
            score = evaluate_clustering(dataset, clusters)
            results.append([name, dataset_name, score])
    return pd.DataFrame(results, columns=["Algorithm", "Dataset", "Silhouette Score"])

def main():
    file_path = "data/ratios_financiers.csv"
    returns_folder = "data/Companies_Historical_Data/"

    # 1. Clustering profils financiers
    print("→ Clustering des profils financiers...")
    data_financial, names_financial = preprocess_financial_clustering(file_path)
    elbow_method(data_financial)
    do_kmeans_clustering(data_financial, names_financial, n_clusters=3)

    # 2. Clustering profils de risque
    print("\n→ Clustering des profils de risque...")
    data_risk, names_risk = preprocess_risk_clustering(file_path)
    plot_dendrogram(data_risk)
    do_hierarchical_clustering(data_risk, names_risk, n_clusters=3)

    # 3. Clustering corrélations rendements
    print("\n→ Clustering des corrélations de rendements journaliers...")
    data_returns = prepare_returns_data(returns_folder)
    plot_correlation_dendrogram(data_returns)

    # 4. Clustering direct sur les profils de rendements (corrigé)
    print("\n→ Clustering direct sur les profils de rendements (chaque entreprise = une ligne)...")
    clustering_on_return_profiles(data_returns)

    # 5. Évaluation et comparaison
    print("\n→ Comparaison des algorithmes de clustering...")
    data_returns_T = data_returns.T  # entreprises = lignes
    results_df = compare_algorithms(data_financial, data_risk, data_returns_T)
    print(results_df)

if __name__ == "__main__":
    main()

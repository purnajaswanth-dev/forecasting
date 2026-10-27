from sklearn.cluster import KMeans


def build_clusters(sites, n_clusters=5):

    coords = sites[["lat", "Lon"]]

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42
    )

    sites["Cluster"] = model.fit_predict(coords)

    return sites
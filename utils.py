import numpy as np
from scipy import stats


def k_means(data, k):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    new_centroids = np.zeros((k, 4))
    clusters = [[] for _ in range(k)]

    while not np.all(centroids == new_centroids):
        distances = np.sqrt(
            np.sum((data - centroids[:, np.newaxis]) ** 2, axis=2)
        )
        labels = np.argmin(distances, axis=0)

        new_centroids = np.array(
            [data[labels == i].mean(axis=0) for i in range(k)]
        )

        centroids = new_centroids

    for i, label in enumerate(labels):
        clusters[label].append(data[i])

    return (centroids, labels + 1, clusters)


def k_means_predict(data, centroids):
    distances = np.sqrt(
        np.sum(np.square(data - centroids[:, np.newaxis]), axis=2)
    )
    labels = np.argmin(distances, axis=0)
    clusters = [[] for _ in range(len(centroids))]

    for i, label in enumerate(labels):
        clusters[label].append(data[i])

    return (distances, labels + 1, clusters)


def knn(test_data, train_data, k):
    distances = np.sqrt(
        np.sum(
            np.square(
                test_data[:, np.newaxis, :] - train_data[np.newaxis, :, :]
            ),
            axis=2,
        )
    )

    return np.argpartition(distances, k, axis=1)[:, :k]


def speedy_knn(df, centroids, k):
    data = np.concatenate(
        (df.values, np.zeros((df.values.shape[0], 1))), axis=1
    )

    # adding predicted labels to data
    data[:, -1] = k_means_predict(data[:, :4], centroids)[1]

    labels = []

    for sample in data:
        cluster = data[data[:, -1] == sample[-1]]
        distance = np.sqrt(
            np.sum(
                np.square(cluster[:, :4] - sample[:4]),
                axis=1,
            )
        )

        nearest_samples = (
            cluster
            if len(cluster) <= k + 1
            else cluster[np.argpartition(distance, k + 1)[: k + 1]]
        )

        labels.append(stats.mode(nearest_samples[:, -2]).mode)

    return np.array(labels, dtype="int16")
